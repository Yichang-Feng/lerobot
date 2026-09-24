#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unitree G1 三路相机 ZMQ 推流服务 (强制 640x480@30fps MJPG 版)
针对腕部相机掉帧问题，使用 v4l2-ctl 底层强制锁定 MJPG 格式。
"""
import base64, json, os, re, signal, subprocess, threading, time
import cv2, numpy as np, zmq

try:
    import pyrealsense2 as rs
    HAS_RS = True
except ImportError:
    HAS_RS = False

# ============ 本机固化配置 ============
FPS = 30
JPEG_QUALITY = 80
CAP_W, CAP_H = 640, 480  # 直接采集并输出 640x480，完美对齐 LeRobot 配置

CAMERAS = [
    {
        "name": "observation.images.global_view",
        "port": 5556, "source": "realsense",
        "rs_serial": "254522074934",
    },
    {
        "name": "observation.images.left_wrist",
        "port": 5557, "source": "opencv",
        "serial": "JR0002", "vid_pid": "0002:0002",
    },
    {
        "name": "observation.images.right_wrist",
        "port": 5558, "source": "opencv",
        "serial": "JR0001", "vid_pid": "0001:0001",
    },
]
# ======================================

def _sys_read(path):
    try:
        with open(path) as f: return f.read().strip()
    except OSError: return ""

def enumerate_video_devices():
    """遍历 /sys/class/video4linux 寻找相机"""
    devices = []
    root = "/sys/class/video4linux"
    if not os.path.isdir(root): return devices
    for entry in sorted(os.listdir(root), key=lambda x: int(x.replace("video", ""))):
        if not re.match(r"^video\d+$", entry): continue
        idx = int(entry.replace("video", ""))
        ddir = os.path.join(root, entry)
        info = {"index": idx, "device": f"/dev/video{idx}",
                "name": _sys_read(os.path.join(ddir, "name")),
                "serial": "", "vid_pid": "", "is_metadata": False}
        try: link = os.path.realpath(os.path.join(ddir, "device"))
        except OSError: link = ""
        cur = link
        for _ in range(6):
            if cur and os.path.isfile(os.path.join(cur, "serial")):
                info["serial"] = _sys_read(os.path.join(cur, "serial"))
                vid = _sys_read(os.path.join(cur, "idVendor"))
                pid = _sys_read(os.path.join(cur, "idProduct"))
                if vid and pid: info["vid_pid"] = f"{vid}:{pid}".lower()
                break
            cur = os.path.dirname(cur)
        if "metadata" in info["name"].lower(): info["is_metadata"] = True
        devices.append(info)
    return devices

def find_video_node(devices, serial=None, vid_pid=None, tag=""):
    pool = [d for d in devices if not d["is_metadata"]] or devices
    cands = pool
    if serial: cands = [d for d in cands if d["serial"].lower() == serial.lower()]
    if vid_pid and len(cands) != 1: cands = [d for d in cands if d["vid_pid"] == vid_pid.lower()]
    if not cands:
        print(f"[!] [{tag}] 未找到相机 (serial={serial}, vid_pid={vid_pid})")
        return None
    chosen = min(cands, key=lambda d: d["index"])
    print(f"[+] [{tag}] 定位 -> {chosen['device']} (serial={chosen['serial']})")
    return chosen["device"]

def force_v4l2_mjpg(device_node, w, h):
    """核心修复：使用 v4l2-ctl 在底层强制设置 MJPG 格式，解决 OpenCV 默认 YUYV 导致 5fps 的问题"""
    try:
        subprocess.run(
            ["v4l2-ctl", "-d", device_node, "--set-fmt-video", f"width={w},height={h},pixelformat=MJPG"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2
        )
        print(f"    [{device_node}] v4l2-ctl 已强制设置为 MJPG {w}x{h}")
    except Exception as e:
        print(f"    [!] {device_node} v4l2-ctl 设置失败: {e}")

class CameraStreamer:
    def __init__(self, cfg, device_path, ctx):
        self.cfg, self.device_path, self.ctx = cfg, device_path, ctx
        self._stop = threading.Event()
        self._thread = self._pipe = self._cap = self._sock = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread: self._thread.join(timeout=3)

    def _open_rs(self):
        cfg = self.cfg
        try:
            pipe, c = rs.pipeline(), rs.config()
            if cfg.get("rs_serial"): c.enable_device(cfg["rs_serial"])
            c.enable_stream(rs.stream.color, CAP_W, CAP_H, rs.format.bgr8, FPS)
            pipe.start(c)
            if pipe.wait_for_frames(timeout_ms=5000).get_color_frame():
                print(f"[+] [{cfg['name']}] RealSense 启动成功 {CAP_W}x{CAP_H}@{FPS}fps")
                return pipe
            pipe.stop()
        except Exception as e:
            print(f"[!] [{cfg['name']}] RealSense 启动失败: {e}")
        return None

    def _open_cv(self):
        cfg = self.cfg
        if not self.device_path: return None
        
        # 1. 底层强制 MJPG (解决 5fps 核心逻辑)
        force_v4l2_mjpg(self.device_path, CAP_W, CAP_H)
        
        # 2. OpenCV 打开 (使用 V4L2 后端)
        dev_idx = int(self.device_path.replace("/dev/video", ""))
        cap = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        if not cap.isOpened():
            print(f"[!] [{cfg['name']}] 无法打开 {self.device_path}")
            return None
            
        ret, fr = cap.read()
        if not ret or fr is None:
            print(f"[!] [{cfg['name']}] {self.device_path} 读帧失败")
            cap.release()
            return None
            
        h, w = fr.shape[:2]
        actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
        print(f"[+] [{cfg['name']}] OpenCV 启动成功 (实际 {w}x{h}, 格式 {fourcc_str})")
        if fourcc_str != "MJPG":
            print(f"    ⚠ 警告: 格式仍为 {fourcc_str}，可能无法达到 30fps！")
        return cap

    def _read(self):
        if self._pipe is not None:
            f = self._pipe.wait_for_frames(timeout_ms=1000).get_color_frame()
            return np.asanyarray(f.get_data()) if f else None
        if self._cap is not None:
            ret, fr = self._cap.read()
            return fr if ret else None
        return None

    def _run(self):
        cfg = self.cfg
        self._sock = self.ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 10)
        self._sock.bind(f"tcp://0.0.0.0:{cfg['port']}")

        if cfg["source"] == "realsense" and HAS_RS:
            self._pipe = self._open_rs()
            if not self._pipe:
                self._sock.close(); return
        else:
            self._cap = self._open_cv()
            if not self._cap:
                self._sock.close(); return

        print(f"[+] [{cfg['name']}] ZMQ 推流: tcp://0.0.0.0:{cfg['port']} ({CAP_W}x{CAP_H}@{FPS}fps)")
        interval = 1.0 / FPS
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        
        while not self._stop.is_set():
            t0 = time.time()
            fr = self._read()
            if fr is None:
                time.sleep(0.01); continue
            
            # 兜底：如果因为某些原因采集到的不是 640x480，则强制 resize，确保 LeRobot 接收端 Shape 绝对正确
            if fr.shape[1] != CAP_W or fr.shape[0] != CAP_H:
                fr = cv2.resize(fr, (CAP_W, CAP_H), interpolation=cv2.INTER_AREA)
                
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            ok, buf = cv2.imencode(".jpg", rgb, enc)
            if not ok: continue
            
            payload = {"timestamps": {cfg["name"]: time.time()},
                       "images": {cfg["name"]: base64.b64encode(buf).decode("ascii")}}
            try:
                self._sock.send_string(json.dumps(payload), zmq.NOBLOCK)
            except zmq.Again:
                pass
                
            dt = time.time() - t0
            if dt < interval: time.sleep(interval - dt)
            
        if self._pipe:
            try: self._pipe.stop()
            except: pass
        if self._cap: self._cap.release()
        self._sock.close()
        print(f"[*] [{cfg['name']}] 已停止")

def main():
    devices = enumerate_video_devices()
    if not HAS_RS: print("[!] 未检测到 pyrealsense2")
    print("=" * 70)
    print(" Unitree G1 三路相机 ZMQ 推流 (强制 640x480@30fps MJPG)")
    print("=" * 70)

    ctx = zmq.Context()
    streamers = []
    for cfg in CAMERAS:
        dev_path = None
        if cfg["source"] == "opencv":
            dev_path = find_video_node(devices, cfg.get("serial"), cfg.get("vid_pid"), tag=cfg["name"])
        streamers.append(CameraStreamer(cfg, dev_path, ctx))
    for s in streamers: s.start()

    stop = threading.Event()
    def _sig(*a): stop.set()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    try:
        while not stop.is_set(): time.sleep(0.5)
    finally:
        for s in streamers: s.stop()
        ctx.term()
        print("[+] 所有相机已安全退出")

if __name__ == "__main__":
    main()
