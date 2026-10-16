"""
数据层：ZMQ 接收 / 数据中枢 Hub / 录制 Recorder / 回放 Session + Player

所有显示数据都经过 Hub：
  live   : ZMQ 线程 + 主程序输出 ──► Hub ──► 界面
                                     └──► Recorder（原始数据落盘）
  replay : Session 文件 ──► Player ──► Hub ──► 界面（同一套解析和绘制代码）

录制目录结构（session_YYYYmmdd_HHMMSS/）：
  meta.json        配置、起止时间
  events.jsonl     {"t": 秒, "k": log|state|jev|input|typing|proc, "d": ..., "img": jev 图像序号}
  cam{i}.mjpg/.idx 相机原始 JPEG 依次拼接；idx 每条 <t:f64, offset:u64, len:u32>
  jev.mjpg/.idx    JEV 输入图像
"""
import base64
import bisect
import json
import os
import queue
import struct
import threading
import time
from collections import deque

import zmq
from PySide6.QtGui import QImage

IDX = struct.Struct("<dQI")


# ============================================================
# 图像字节提取：兼容 原始 JPEG / base64 JPEG / JSON 包裹 / multipart
# ============================================================
def _is_img(b):
    return b[:3] == b"\xff\xd8\xff" or b[:4] == b"\x89PNG"


def _b64(s):
    try:
        b = base64.b64decode(s, validate=False)
    except Exception:
        return None
    return b if _is_img(b) else None


def _extract_img_bytes_from_obj(obj):
    if isinstance(obj, (str, bytes)):
        b_in = obj.encode("ascii", "ignore") if isinstance(obj, str) else obj
        if _is_img(b_in):
            return b_in
        if len(b_in) > 100:
            b = _b64(b_in)
            if b:
                return b
        return None
    elif isinstance(obj, dict):
        for key in ("images", "image", "data", "frame", "rgb"):
            if key in obj:
                res = _extract_img_bytes_from_obj(obj[key])
                if res is not None:
                    return res
        for v in obj.values():
            res = _extract_img_bytes_from_obj(v)
            if res is not None:
                return res
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            res = _extract_img_bytes_from_obj(v)
            if res is not None:
                return res
    return None


def find_image_bytes(parts):
    for data in reversed(parts):
        if not data:
            continue
        if _is_img(data):
            return data
        s = data.strip()
        # 1. Check if payload contains JSON structure
        if b"{" in s and b"}" in s:
            first_b = s.find(b"{")
            last_b = s.rfind(b"}")
            try:
                obj = json.loads(s[first_b : last_b + 1])
                b = _extract_img_bytes_from_obj(obj)
                if b:
                    return b
            except Exception:
                pass
        # 2. Check topic-prefixed raw bytes or base64
        if b" " in s[:80]:
            after_space = s.split(b" ", 1)[1].strip()
            if _is_img(after_space):
                return after_space
            b = _b64(after_space)
            if b:
                return b
        # 3. Direct base64 decode
        b = _b64(s)
        if b:
            return b
    return None


# ============================================================
# Hub
# ============================================================
class Slot:
    def __init__(self):
        self.value, self.seq, self.last, self.stamps = None, 0, -1e18, deque()


class Hub:
    def __init__(self, n_cams, clock=time.time):
        self.clock = clock
        self.lock = threading.Lock()
        self.cams = [Slot() for _ in range(n_cams)]
        self.state = Slot()
        self.q = deque()
        self.recorder = None
        self._state_rec_t = -1e18

    def now(self):
        return self.clock()

    def _touch(self, slot, v, t):
        with self.lock:
            slot.value, slot.last = v, t
            slot.seq += 1
            slot.stamps.append(t)

    def frame(self, i, jpeg, t=None):
        t = self.now() if t is None else t
        img = QImage.fromData(jpeg)
        if img.isNull():
            return False
        # 机载推流端采用 OpenCV 编码（原始为 BGR），调换红蓝通道恢复正常真实色彩
        img = img.rgbSwapped()
        self._touch(self.cams[i], img, t)
        rec = self.recorder
        if rec:
            rec.frame(i, t, jpeg)
        return True

    def state_msg(self, obj, t=None):
        t = self.now() if t is None else t
        self._touch(self.state, obj, t)
        rec = self.recorder
        if rec and t - self._state_rec_t >= 1 / 30:   # 状态最多按 30Hz 录制
            self._state_rec_t = t
            rec.event(t, "state", obj)

    def event(self, kind, data, t=None, img=None):
        t = self.now() if t is None else t
        rec = self.recorder
        if rec:
            rec.event(t, kind, data, img)
        self.q.append((t, kind, data, img))

    def drain(self):
        out = []
        while True:
            try:
                out.append(self.q.popleft())
            except IndexError:
                return out

    def latest(self, slot):
        with self.lock:
            return slot.value, slot.seq

    def rate(self, slot):
        now = self.now()
        with self.lock:
            while slot.stamps and not (0 <= now - slot.stamps[0] <= 1.0):
                slot.stamps.popleft()
            return len(slot.stamps)

    def alive(self, slot, window=1.5):
        return 0 <= self.now() - slot.last < window

    def reset(self):
        with self.lock:
            for s in self.cams + [self.state]:
                s.value, s.last = None, -1e18
                s.stamps.clear()
                s.seq += 1
        self.q.clear()
        self._state_rec_t = -1e18


# ============================================================
# ZMQ 订阅线程
# ============================================================
class SubThread(threading.Thread):
    """latest_only=True：积压时只处理最新一条（相机/状态）；False：逐条处理（JEV 判定）"""

    def __init__(self, ctx, addr, handler, latest_only=True, conflate=False):
        super().__init__(daemon=True)
        self.ctx, self.addr, self.handler, self.latest_only, self.conflate = ctx, addr, handler, latest_only, conflate
        self.running = True
        self.error = None

    def run(self):
        sock = self.ctx.socket(zmq.SUB)
        if self.conflate:
            try:
                sock.setsockopt(zmq.CONFLATE, 1)
            except Exception:
                pass
        sock.setsockopt(zmq.RCVHWM, 4 if self.latest_only else 100)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.connect(self.addr)
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        while self.running:
            if not poller.poll(200):
                continue
            msgs = []
            while True:
                try:
                    msgs.append(sock.recv_multipart(zmq.NOBLOCK))
                except zmq.Again:
                    break
            if self.latest_only:
                msgs = msgs[-1:]
            for parts in msgs:
                try:
                    self.handler(parts)
                except Exception as e:
                    self.error = f"{type(e).__name__}: {e}"
        sock.close()


# ============================================================
# 录制
# ============================================================
class _Store:
    def __init__(self, base):
        self.data = open(base + ".mjpg", "ab")
        self.idx = open(base + ".idx", "ab")
        self.n = 0

    def add(self, t, b):
        off = self.data.tell()
        self.data.write(b)
        self.idx.write(IDX.pack(t, off, len(b)))
        self.n += 1
        return self.n - 1

    def flush(self):
        self.data.flush()
        self.idx.flush()

    def close(self):
        self.data.close()
        self.idx.close()


class Recorder:
    def __init__(self, root, meta, n_cams, record_fps=0):
        root = os.path.expanduser(root)
        ts = time.strftime("session_%Y%m%d_%H%M%S")
        candidate = os.path.join(root, ts)
        count = 1
        while os.path.exists(candidate):
            candidate = os.path.join(root, f"{ts}_{count}")
            count += 1
        self.dir = candidate
        os.makedirs(self.dir, exist_ok=True)
        self.t0 = time.time()
        self.min_dt = 1.0 / record_fps if record_fps else 0.0
        self.last = [-1e18] * n_cams
        self.q = queue.Queue(maxsize=900)
        self.dropped = 0
        self.meta = dict(meta, version=1, start_wall=self.t0,
                         start_local=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.t0)))
        self._write_meta()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _write_meta(self):
        with open(os.path.join(self.dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def _put(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def frame(self, i, t, jpeg):
        if t - self.last[i] < self.min_dt:
            return
        self.last[i] = t
        self._put(("f", i, t - self.t0, jpeg))

    def event(self, t, kind, data, img=None):
        self._put(("e", t - self.t0, kind, data, img))

    def _run(self):
        stores = {}
        ev = open(os.path.join(self.dir, "events.jsonl"), "a", encoding="utf-8")

        def store(name):
            if name not in stores:
                stores[name] = _Store(os.path.join(self.dir, name))
            return stores[name]

        last_flush = time.time()
        while True:
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                item = ()
            if item is None:
                break
            if item and item[0] == "f":
                _, i, t, jpeg = item
                store(f"cam{i}").add(t, jpeg)
            elif item:
                _, t, kind, data, img = item
                rec = {"t": round(t, 4), "k": kind, "d": data}
                if img:
                    rec["img"] = store("jev").add(t, img)
                ev.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if time.time() - last_flush > 0.5:
                ev.flush()
                for s in stores.values():
                    s.flush()
                last_flush = time.time()
        ev.close()
        for s in stores.values():
            s.close()

    def close(self):
        try:
            self.q.put(None, timeout=5)
        except queue.Full:
            pass
        self.th.join(timeout=15)
        self.meta.update(end_wall=time.time(), duration=round(time.time() - self.t0, 2),
                         dropped_items=self.dropped)
        self._write_meta()


# ============================================================
# 回放
# ============================================================
class FrameStore:
    def __init__(self, base):
        self.f = open(base + ".mjpg", "rb")
        raw = open(base + ".idx", "rb").read()
        raw = raw[: len(raw) // IDX.size * IDX.size]   # 容忍异常退出导致的截断
        self.t, self.off, self.len = [], [], []
        for t, o, n in IDX.iter_unpack(raw):
            self.t.append(t)
            self.off.append(o)
            self.len.append(n)

    def index_at(self, t):
        return bisect.bisect_right(self.t, t) - 1

    def read(self, i):
        self.f.seek(self.off[i])
        return self.f.read(self.len[i])


class Session:
    def __init__(self, path):
        self.dir = os.path.expanduser(path)
        with open(os.path.join(self.dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.events = []
        with open(os.path.join(self.dir, "events.jsonl"), encoding="utf-8") as f:
            for line in f:
                try:
                    self.events.append(json.loads(line))
                except Exception:
                    pass
        self.events.sort(key=lambda e: e["t"])
        n = len(self.meta.get("cameras", [])) or 3
        self.cams = []
        for i in range(n):
            base = os.path.join(self.dir, f"cam{i}")
            self.cams.append(FrameStore(base) if os.path.exists(base + ".idx") else None)
        base = os.path.join(self.dir, "jev")
        self.jev = FrameStore(base) if os.path.exists(base + ".idx") else None
        ends = [self.events[-1]["t"] if self.events else 0.0]
        ends += [c.t[-1] for c in self.cams if c and c.t]
        self.duration = max(ends)


class Player:
    def __init__(self, session: Session, hub: Hub):
        self.s, self.hub = session, hub
        self.t, self.playing, self.speed = 0.0, False, 1.0
        self.cursor = 0
        self.cam_idx = [-1] * len(session.cams)
        self._wall = None
        hub.clock = lambda: self.t

    def toggle(self):
        if not self.playing and self.t >= self.s.duration - 1e-3:
            self.seek(0.0)
        self.playing = not self.playing
        self._wall = None

    def seek(self, t):
        self.t = min(max(0.0, t), self.s.duration)
        self.hub.reset()
        self.cursor = 0
        self.cam_idx = [-1] * len(self.s.cams)
        self._advance(self.t, catchup=True)

    def tick(self):
        """实时回放：按墙钟推进"""
        now = time.monotonic()
        if self.playing and self._wall is not None:
            self.t = min(self.s.duration, self.t + (now - self._wall) * self.speed)
            if self.t >= self.s.duration:
                self.playing = False
        self._wall = now
        self._advance(self.t)

    def set_time(self, t):
        """导出：逐帧推进（只能向前）"""
        self.t = t
        self._advance(t)

    def _advance(self, t, catchup=False):
        ev = self.s.events
        end = bisect.bisect_right([e["t"] for e in ev[self.cursor:]], t) + self.cursor if catchup else None
        last_img = None
        if catchup:  # 跳转时只读取最后一张 JEV 图，其余跳过
            for j in range(end - 1, self.cursor - 1, -1):
                if "img" in ev[j]:
                    last_img = j
                    break
        while self.cursor < len(ev) and ev[self.cursor]["t"] <= t:
            e = ev[self.cursor]
            j = self.cursor
            self.cursor += 1
            if e["k"] == "state":
                self.hub.state_msg(e["d"], t=e["t"])
                continue
            img = None
            if "img" in e and self.s.jev and (not catchup or j == last_img):
                try:
                    img = self.s.jev.read(e["img"])
                except Exception:
                    img = None
            self.hub.event(e["k"], e["d"], t=e["t"], img=img)
        for i, fs in enumerate(self.s.cams):
            if not fs:
                continue
            j = fs.index_at(t)
            if j >= 0 and j != self.cam_idx[i]:
                self.cam_idx[i] = j
                self.hub.frame(i, fs.read(j), t=fs.t[j])
