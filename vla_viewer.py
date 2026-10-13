"""
G1 Dex-1 VLA 控制台（PySide6）

集成方式（主程序零改动）：
  1. 相机：直接订阅机载 Orin 的 ZMQ PUB 端口 5555 / 5557 / 5558（PUB 支持多个订阅者，不影响主程序）
  2. 机器人状态：订阅 6001 端口（模式 / 关节广播）
  3. 主程序：界面用 QProcess 启动 run_dex1_water_bottle_transfer_hold.sh，
     - stdout/stderr 实时显示在「程序输出」（保留 ANSI 颜色）
     - 按钮 s / d / r / 回车 / q 直接写入主程序 stdin，等同于在终端里敲键

运行：
  pip install PySide6 pyzmq
  python vla_viewer.py                 # 连接真机（参数见 CONFIG 或命令行）
  python vla_viewer.py --mock          # 本地模拟（先另开终端运行 python mock_robot.py）
"""
import argparse
import base64
import html
import json
import os
import re
import sys
import threading
import time
from collections import deque

# 自动处理 Conda/虚拟环境下 Qt xcb 插件与系统 libxkbcommon 的 ABI 冲突（避免段错误）
if __name__ == "__main__" and sys.argv and sys.argv[0] != "-c":
    _env_lib = os.path.join(sys.prefix, "lib")
    _current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if os.path.isdir(_env_lib) and _env_lib not in _current_ld.split(":") and not os.environ.get("_VLA_REEXEC"):
        os.environ["_VLA_REEXEC"] = "1"
        os.environ["LD_LIBRARY_PATH"] = f"{_env_lib}:{_current_ld}".strip(":")
        os.execv(sys.executable, [sys.executable] + sys.argv)

import zmq
from PySide6.QtCore import Qt, QTimer, QProcess, QProcessEnvironment
from PySide6.QtGui import QImage, QPixmap, QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QPlainTextEdit, QLineEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QSplitter, QFrame, QCheckBox,
)

def _detect_conda_sh():
    for candidate in [
        "~/miniforge3/etc/profile.d/conda.sh",
        "~/miniconda3/etc/profile.d/conda.sh",
        "~/anaconda3/etc/profile.d/conda.sh",
        "/opt/conda/etc/profile.d/conda.sh",
    ]:
        if os.path.exists(os.path.expanduser(candidate)):
            return candidate
    return "~/miniforge3/etc/profile.d/conda.sh"

_conda_sh = _detect_conda_sh()
_project_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 配置（也可用命令行参数覆盖）
# ============================================================
DEFAULT_ROBOT_IP = "192.168.31.248"

CONFIG = dict(
    robot_ip=DEFAULT_ROBOT_IP,
    cameras=[
        ("头部相机 · base_0_rgb", 5556),
        ("左腕相机 · left_wrist_0_rgb", 5557),
        ("右腕相机 · right_wrist_0_rgb", 5558),
    ],
    state_port=6001,
    mode_port=6000,
    # 主程序启动命令（通过 bash -lc 执行，可以 source conda）
    # --display_data=false：关闭 Rerun，画面统一由本界面显示
    main_cmd=(
        f"source {_conda_sh} && conda activate lerobot && cd {_project_dir} && "
        "./run_dex1_water_bottle_transfer_hold.sh "
        f"--robot_ip={DEFAULT_ROBOT_IP} "
        "--policy.path=outputs/train/pi05_lora_5090_g1_pick_put_dex1_subtasks/merged_model "
        "--valen_evaluator=true --valen_ip=0.0.0.0 --valen_port=5559 "
        "--display_data=false"
    ),
)

# 主程序键盘命令（文档 6.2 节）
QUICK_CMDS = [
    ("开始抓取  s", "s", "旁路 JEV，直接开始 Phase 0 桌1抓取"),
    ("到达桌2  d", "d", "旁路 JEV，解除 Phase 2 阻塞，接管桌2放置"),
    ("彻底重置  r", "r", "打断轮询，Phase 归零，机械臂回位"),
    ("查询状态  ↵", "", "输出当前阶段、夹爪弧度、肩部角度"),
]

UI_FPS = 30


# ============================================================
# 图像解码：兼容 原始 JPEG / base64 JPEG / JSON 包裹 / multipart
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
    """递归从 JSON 对象（如 {'images': {'cam_name': '<base64>'}}）中提取图像数据"""
    if isinstance(obj, str):
        if len(obj) > 100:
            b = _b64(obj.encode())
            if b:
                return b
        return None
    elif isinstance(obj, dict):
        if "images" in obj and isinstance(obj["images"], dict):
            for v in obj["images"].values():
                b = _extract_img_bytes_from_obj(v)
                if b:
                    return b
        for v in obj.values():
            b = _extract_img_bytes_from_obj(v)
            if b:
                return b
    elif isinstance(obj, list):
        for v in obj:
            b = _extract_img_bytes_from_obj(v)
            if b:
                return b
    return None


def find_image_bytes(parts):
    for data in reversed(parts):
        if _is_img(data):
            return data
        s = data.strip()
        if s.startswith(b"{"):
            try:
                obj = json.loads(s)
            except Exception:
                obj = None
            if obj is not None:
                b = _extract_img_bytes_from_obj(obj)
                if b:
                    return b
        b = _b64(s)
        if b:
            return b
        if b" " in s[:64]:  # "topic <base64>" 形式
            b = _b64(s.split(b" ", 1)[1])
            if b:
                return b
    return None


# ============================================================
# ZMQ 接收线程：只保留最新一帧，界面定时拉取
# ============================================================
class Receiver(threading.Thread):
    def __init__(self, ctx, addr):
        super().__init__(daemon=True)
        self.ctx, self.addr = ctx, addr
        self.lock = threading.Lock()
        self.stamps = deque()
        self.running = True
        self.error = None
        self.last_rx = 0.0

    def rate(self):
        now = time.time()
        with self.lock:
            while self.stamps and now - self.stamps[0] > 1.0:
                self.stamps.popleft()
            return len(self.stamps)

    def handle(self, parts):
        raise NotImplementedError

    def run(self):
        sock = self.ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 4)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.connect(self.addr)
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        while self.running:
            if not poller.poll(200):
                continue
            parts = None
            while True:  # 取走所有积压，只处理最新一条
                try:
                    parts = sock.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
            if parts is None:
                continue
            try:
                self.handle(parts)
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
        sock.close()


class CameraReceiver(Receiver):
    def __init__(self, ctx, addr):
        super().__init__(ctx, addr)
        self.image = None

    def handle(self, parts):
        data = find_image_bytes(parts)
        if data is None:
            p = parts[-1]
            self.error = f"无法解析图像（{len(p)} bytes，开头 {p[:12]!r}）"
            return
        img = QImage.fromData(data)
        if img.isNull():
            self.error = "图像解码失败"
            return
        # 相机源端采用 OpenCV BGR 编码，调换红蓝通道恢复正常 RGB 色彩
        img = img.rgbSwapped()
        with self.lock:
            self.image = img
            self.stamps.append(time.time())
        self.last_rx = time.time()
        self.error = None

    def latest(self):
        with self.lock:
            return self.image


class StateReceiver(Receiver):
    """6001 状态广播；格式未定，尽量从 JSON 里找模式字段"""

    def __init__(self, ctx, addr):
        super().__init__(ctx, addr)
        self.state = None

    def handle(self, parts):
        raw = parts[-1]
        try:
            obj = json.loads(raw)
        except Exception:
            obj = {"_raw_bytes": len(raw)}
        with self.lock:
            self.state = obj
            self.stamps.append(time.time())
        self.last_rx = time.time()

    def latest(self):
        with self.lock:
            return self.state


def parse_mode_string(raw_str):
    """规范化模式名称 (VLA / NAV / GAMEPAD)"""
    s = str(raw_str).strip().lower()
    if "vla" in s:
        return "VLA"
    if "nav" in s or "daohang" in s:
        return "NAV"
    if "gamepad" in s or "manual" in s or "teleop" in s or "shoubing" in s or "joystick" in s:
        return "GAMEPAD"
    return s.upper() if s else "UNKNOWN"


class ModeReceiver(Receiver):
    """6000 模式广播监听（PUB/SUB: {'state':'gamepad'/'vla'/'nav'} 等）"""

    def __init__(self, ctx, addr):
        super().__init__(ctx, addr)
        self.mode = None

    def handle(self, parts):
        for part in reversed(parts):
            try:
                s = part.decode("utf-8", "ignore").strip()
            except Exception:
                continue
            if not s:
                continue
            if "{" in s and "}" in s:
                try:
                    first_b = s.find("{")
                    last_b = s.rfind("}")
                    obj = json.loads(s[first_b : last_b + 1])
                    if isinstance(obj, dict):
                        for k in ("state", "mode", "robot_mode", "status", "action"):
                            if k in obj:
                                m = parse_mode_string(str(obj[k]))
                                with self.lock:
                                    self.mode = m
                                    self.stamps.append(time.time())
                                self.last_rx = time.time()
                                return
                except Exception:
                    pass
            m = parse_mode_string(s)
            if m != "UNKNOWN":
                with self.lock:
                    self.mode = m
                    self.stamps.append(time.time())
                self.last_rx = time.time()
                return

    def latest(self):
        with self.lock:
            return self.mode


# ============================================================
# ANSI 彩色输出 → HTML
# ============================================================
ANSI_RE = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")
ANSI_COLORS = {30: "#8b9097", 31: "#ff6b6b", 32: "#5fd38d", 33: "#f2c94c",
               34: "#5aa9ff", 35: "#d08bff", 36: "#4fd1d9", 37: "#e6e7ea"}


def _esc(s):
    return html.escape(s).replace(" ", "&nbsp;")


def ansi_to_html(s):
    out, pos, color, bold, open_span = [], 0, None, False, False
    for m in ANSI_RE.finditer(s):
        out.append(_esc(s[pos:m.start()]))
        pos = m.end()
        if m.group(2) != "m":
            continue
        codes = [int(c) for c in m.group(1).split(";") if c.isdigit()] or [0]
        for c in codes:
            if c == 0:
                color, bold = None, False
            elif c == 1:
                bold = True
            elif c == 22:
                bold = False
            elif c == 39:
                color = None
            elif 30 <= c <= 37:
                color = ANSI_COLORS[c]
            elif 90 <= c <= 97:
                color = ANSI_COLORS[c - 60]
        if open_span:
            out.append("</span>")
            open_span = False
        if color or bold:
            st = (f"color:{color};" if color else "") + ("font-weight:700;" if bold else "")
            out.append(f'<span style="{st}">')
            open_span = True
    out.append(_esc(s[pos:]))
    if open_span:
        out.append("</span>")
    return "".join(out)


def strip_ansi(s):
    return ANSI_RE.sub("", s)


# ============================================================
# 界面组件
# ============================================================
class CameraView(QFrame):
    def __init__(self, title, addr):
        super().__init__()
        self.setObjectName("card")
        t = QLabel(title)
        t.setObjectName("cardTitle")
        self.info = QLabel("连接中…")
        self.info.setObjectName("muted")
        head = QHBoxLayout()
        head.addWidget(t)
        head.addStretch()
        head.addWidget(self.info)
        self.view = QLabel(f"等待画面\n{addr}")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setObjectName("camView")
        self.view.setMinimumSize(240, 180)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 8)
        lay.setSpacing(6)
        lay.addLayout(head)
        lay.addWidget(self.view, 1)

    def update_from(self, rx: CameraReceiver):
        img = rx.latest()
        stale = time.time() - rx.last_rx > 1.0
        if rx.error:
            new_info = f"⚠ {rx.error}"
        elif img is None:
            new_info = "无信号"
        else:
            new_info = ("⚠ 已中断  " if stale else "") + f"{img.width()}×{img.height()}  {rx.rate()} fps"
        if self.info.text() != new_info:
            self.info.setText(new_info)
        if img is not None:
            pix = QPixmap.fromImage(img).scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.view.setPixmap(pix)


class Badge(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setObjectName("badge")
        self._level = "neutral"

    def set(self, text, level="neutral"):
        if self.text() == text and self._level == level:
            return
        self.setText(text)
        if self._level != level:
            self._level = level
            self.setProperty("level", level)
            self.style().unpolish(self)
            self.style().polish(self)


class MainWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._refresh_count = 0
        self.setWindowTitle("G1 Dex-1 VLA Console")
        self.resize(1440, 900)

        # ---- ZMQ 接收 ----
        self.ctx = zmq.Context.instance()
        ip = cfg["robot_ip"]
        self.cam_rx = [CameraReceiver(self.ctx, f"tcp://{ip}:{p}") for _, p in cfg["cameras"]]
        self.state_rx = StateReceiver(self.ctx, f"tcp://{ip}:{cfg['state_port']}")
        self.mode_rx = ModeReceiver(self.ctx, f"tcp://{ip}:{cfg.get('mode_port', 6000)}")
        self.current_mode = "未知"
        for r in self.cam_rx + [self.state_rx, self.mode_rx]:
            r.start()

        # ---- 相机区 ----
        self.cams = [CameraView(n, f"tcp://{ip}:{p}") for n, p in cfg["cameras"]]
        cam_row = QWidget()
        g = QGridLayout(cam_row)
        g.setContentsMargins(0, 0, 0, 0)
        g.setSpacing(10)
        for i, c in enumerate(self.cams):
            g.addWidget(c, 0, i)

        # ---- 状态条 ----
        self.b_proc = Badge()
        self.b_mode = Badge()
        self.b_phase = Badge()
        self.b_state = Badge()
        status_row = QHBoxLayout()
        for lbl, b in (("主程序", self.b_proc), ("机器人模式", self.b_mode),
                       ("任务阶段", self.b_phase), ("6001 状态流", self.b_state)):
            l = QLabel(lbl)
            l.setObjectName("muted")
            status_row.addWidget(l)
            status_row.addWidget(b)
            status_row.addSpacing(14)
        status_row.addStretch()

        # ---- 控制区 ----
        ctrl = QGroupBox("控制")
        self.btn_start = QPushButton("▶ 启动主程序")
        self.btn_start.setObjectName("primary")
        self.btn_stop = QPushButton("■ 退出主程序 (q)")
        self.btn_stop.setObjectName("danger")
        prow = QHBoxLayout()
        prow.addWidget(self.btn_start)
        prow.addWidget(self.btn_stop)

        self.cmd_btns = []
        qgrid = QGridLayout()
        qgrid.setSpacing(8)
        for i, (label, key, tip) in enumerate(QUICK_CMDS):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setObjectName("cmd")
            b.clicked.connect(lambda _=False, k=key: self.send_line(k))
            qgrid.addWidget(b, i // 2, i % 2)
            self.cmd_btns.append(b)

        self.line = QLineEdit()
        self.line.setObjectName("mainInput")
        self.line.setPlaceholderText("向主程序输入指令 (s / d / r / 回车查询)...")
        self.line.setMinimumHeight(56)

        self.btn_send = QPushButton("发 送 ↵")
        self.btn_send.setObjectName("btnSend")
        self.btn_send.setMinimumHeight(56)
        self.btn_send.setToolTip("向主程序 stdin 发送指令（快捷键：回车）")

        lrow = QHBoxLayout()
        lrow.addWidget(self.line, 1)
        lrow.addWidget(self.btn_send)

        self.robot_info = QPlainTextEdit()
        self.robot_info.setReadOnly(True)
        self.robot_info.setObjectName("mono")
        self.robot_info.setMaximumBlockCount(30)
        self.robot_info.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.robot_info.setFixedHeight(170)
        self.robot_info.setPlainText("—")

        safety = QLabel("硬件急停：手柄双摇杆同时按下（Damp）")
        safety.setObjectName("warn")

        cl = QVBoxLayout(ctrl)
        cl.setSpacing(10)
        cl.addLayout(prow)
        cl.addLayout(qgrid)
        cl.addLayout(lrow)
        sep = QLabel("机器人状态 (6001)")
        sep.setObjectName("muted")
        cl.addWidget(sep)
        cl.addWidget(self.robot_info)
        cl.addWidget(safety)

        # ---- 输出区 ----
        outbox = QGroupBox("程序输出")
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(5000)
        self.output.setObjectName("console")
        f = QFont("DejaVu Sans Mono")
        f.setStyleHint(QFont.Monospace)
        f.setPointSize(10)
        self.output.setFont(f)
        self.chk_scroll = QCheckBox("自动滚动")
        self.chk_scroll.setChecked(True)
        self.btn_clear = QPushButton("清空")
        orow = QHBoxLayout()
        orow.addWidget(self.chk_scroll)
        orow.addStretch()
        orow.addWidget(self.btn_clear)
        ol = QVBoxLayout(outbox)
        ol.addWidget(self.output, 1)
        ol.addLayout(orow)

        bottom = QSplitter(Qt.Horizontal)
        bottom.addWidget(ctrl)
        bottom.addWidget(outbox)
        bottom.setSizes([430, 1010])

        bottom_wrap = QWidget()
        bw = QVBoxLayout(bottom_wrap)
        bw.setContentsMargins(0, 4, 0, 0)
        bw.addLayout(status_row)
        bw.addWidget(bottom, 1)

        main_split = QSplitter(Qt.Vertical)
        main_split.addWidget(cam_row)
        main_split.addWidget(bottom_wrap)
        main_split.setSizes([380, 520])

        root = QWidget()
        rl = QVBoxLayout(root)
        rl.setContentsMargins(12, 12, 12, 12)
        rl.addWidget(main_split)
        self.setCentralWidget(root)

        # ---- 主程序子进程 ----
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")    # Python 不缓冲输出
        env.insert("FORCE_COLOR", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self.on_proc_output)
        self.proc.stateChanged.connect(self.on_proc_state)
        self.proc.finished.connect(self.on_proc_finished)
        self._buf = ""
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.timeout.connect(self.flush_partial)

        self.btn_start.clicked.connect(self.start_program)
        self.btn_stop.clicked.connect(self.stop_program)
        self.btn_send.clicked.connect(self.send_input)
        self.line.returnPressed.connect(self.send_input)
        self.btn_clear.clicked.connect(self.output.clear)

        self.phase = None
        self.on_proc_state(QProcess.NotRunning)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(int(1000 / UI_FPS))
        self.log_sys(f"相机订阅：{', '.join(r.addr for r in self.cam_rx)}")
        self.log_sys(f"状态订阅：{self.state_rx.addr}")

    # ---------------- 刷新 ----------------
    def refresh(self):
        self._refresh_count += 1
        for v, r in zip(self.cams, self.cam_rx):
            v.update_from(r)

        # 状态面板更新节流到 ~10 Hz (每 3 帧一次)，避免高频刷新导致界面组件闪烁
        if self._refresh_count % 3 == 0:
            st = self.state_rx.latest()
            hz = self.state_rx.rate()
            # 模式识别优先级：6000端口最新接收 > 主程序日志识别 > 历史模式
            mode_from_rx = self.mode_rx.latest()
            if mode_from_rx and mode_from_rx != "UNKNOWN":
                self.current_mode = mode_from_rx

            mode_disp = self.current_mode or "未知"
            if mode_disp == "VLA":
                self.b_mode.set("VLA", "ok")
            elif mode_disp in ("GAMEPAD", "NAV"):
                self.b_mode.set(mode_disp, "info")
            else:
                self.b_mode.set(mode_disp, "neutral")

            if st is None:
                self.b_state.set("无数据", "bad")
                formatted = f"当前模式: {mode_disp}\n6001 状态流: 等待数据…"
                if self.robot_info.toPlainText() != formatted:
                    self.robot_info.setPlainText(formatted)
                return

            self.b_state.set(f"{hz} Hz", "ok" if hz > 0 else "bad")
            formatted = self.format_state(st, current_mode=mode_disp)
            if self.robot_info.toPlainText() != formatted:
                self.robot_info.setPlainText(formatted)

    @staticmethod
    def format_state(st, current_mode="未知"):
        if not isinstance(st, dict):
            return f"当前模式: {current_mode}\n{str(st)[:400]}"
        lines = [f"当前模式: {current_mode}"]
        for k, v in st.items():
            # 过滤顶层 wireless 字段，不输出冗余无意义长字符串
            if "wireless" in str(k).lower():
                continue
            if k == "data" and isinstance(v, dict):
                lines.append("data:")
                for sub_k, sub_v in v.items():
                    # 过滤掉原始 wireless_remote 冗余长串信号
                    if "wireless" in str(sub_k).lower():
                        continue
                    if sub_k == "motor_state" and isinstance(sub_v, list):
                        q_list = [f"{m.get('q', 0):.2f}" for m in sub_v if isinstance(m, dict)]
                        lines.append(f"  motor q({len(q_list)}): [{', '.join(q_list[:8])}, …]")
                    elif isinstance(sub_v, (list, dict)):
                        lines.append(f"  {sub_k}: {str(sub_v)[:60]}")
                    else:
                        lines.append(f"  {sub_k}: {sub_v}")
                continue
            if isinstance(v, list) and len(v) > 8:
                v = "[" + ", ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in v[:8]) + f", … ({len(v)})]"
            elif isinstance(v, float):
                v = f"{v:.3f}"
            lines.append(f"{k}: {v}")
        return "\n".join(lines[:12])

    # ---------------- 主程序 ----------------
    def start_program(self):
        if self.proc.state() != QProcess.NotRunning:
            return
        self.log_sys(f"启动：{self.cfg['main_cmd']}")
        self.proc.start("bash", ["-lc", self.cfg["main_cmd"]])

    def stop_program(self):
        if self.proc.state() == QProcess.NotRunning:
            return
        self.send_line("q")
        # 3 秒内没优雅退出则强制结束
        QTimer.singleShot(3000, lambda: self.proc.state() != QProcess.NotRunning and self.proc.terminate())
        QTimer.singleShot(5000, lambda: self.proc.state() != QProcess.NotRunning and self.proc.kill())

    def send_line(self, text):
        if self.proc.state() != QProcess.Running:
            self.log_sys("主程序未运行，命令未发送")
            return
        # 安全清洗：去除前后不可见字符与控制字符，保留正常单行命令输入
        clean_text = "".join(ch for ch in str(text) if ch.isprintable() or ch == ' ').strip()
        try:
            self.proc.write((clean_text + "\n").encode("utf-8"))
            display_text = clean_text if clean_text else "↵ (查询状态)"
            self.append_html(f'<span style="color:#5aa9ff; font-size:14px; font-weight:800;">&gt;&nbsp;{_esc(display_text)}</span>')
        except Exception as e:
            self.log_sys(f"发送异常: {e}")

    def send_input(self):
        t = self.line.text()
        self.line.clear()
        self.send_line(t)
        self.line.setFocus()

    def on_proc_output(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        self._buf += data.replace("\r\n", "\n").replace("\r", "\n")
        *lines, self._buf = self._buf.split("\n")
        if lines:
            html_chunks = []
            for ln in lines:
                plain = strip_ansi(ln)
                m = re.search(r"Phase\s*(\d)", plain)
                if m and m.group(1) != self.phase:
                    self.phase = m.group(1)
                    self.b_phase.set(f"Phase {self.phase}", "info")
                # 动态从主程序输出捕获手柄/底盘模式切换
                m_mode = re.search(r"模式切换:.*?->.*?\b(VLA|GAMEPAD|NAV)\b", plain, re.IGNORECASE)
                if m_mode:
                    self.current_mode = m_mode.group(1).upper()
                ts = time.strftime("%H:%M:%S")
                html_chunks.append(f'<span style="color:#5d636b;">{ts}</span>&nbsp;&nbsp;{ansi_to_html(ln)}')
            for chunk in html_chunks:
                self.output.appendHtml(chunk)
            if self.chk_scroll.isChecked():
                self.output.moveCursor(QTextCursor.End)
        if self._buf:  # input() 提示符等不带换行的内容，稍后补刷
            self._flush_timer.start(200)

    def flush_partial(self):
        if self._buf:
            self.append_program_line(self._buf)
            self._buf = ""

    def append_program_line(self, ln):
        plain = strip_ansi(ln)
        m = re.search(r"Phase\s*(\d)", plain)
        if m and m.group(1) != self.phase:
            self.phase = m.group(1)
            self.b_phase.set(f"Phase {self.phase}", "info")
        m_mode = re.search(r"模式切换:.*?->.*?\b(VLA|GAMEPAD|NAV)\b", plain, re.IGNORECASE)
        if m_mode:
            self.current_mode = m_mode.group(1).upper()
        self.append_html(ansi_to_html(ln))

    def on_proc_state(self, s):
        running = s != QProcess.NotRunning
        self.b_proc.set("运行中" if running else "未运行", "ok" if running else "neutral")
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        for b in self.cmd_btns + [self.btn_send]:
            b.setEnabled(running)
        self.line.setEnabled(running)

    def on_proc_finished(self, code, status):
        self.flush_partial()
        self.log_sys(f"主程序已退出（code={code}）")
        self.phase = None
        self.b_phase.set("—", "neutral")

    # ---------------- 日志 ----------------
    def append_html(self, body):
        ts = time.strftime("%H:%M:%S")
        self.output.appendHtml(f'<span style="color:#5d636b;">{ts}</span>&nbsp;&nbsp;{body}')
        if self.chk_scroll.isChecked():
            self.output.moveCursor(QTextCursor.End)

    def log_sys(self, text):
        self.append_html(f'<span style="color:#8b9097;">[viewer]&nbsp;{_esc(text)}</span>')

    def closeEvent(self, e):
        if self.proc.state() != QProcess.NotRunning:
            self.proc.write(b"q\n")
            if not self.proc.waitForFinished(3000):
                self.proc.kill()
                self.proc.waitForFinished(1000)
        for r in self.cam_rx + [self.state_rx, self.mode_rx]:
            r.running = False
        super().closeEvent(e)


STYLE = """
QWidget { background:#1e1f22; color:#dcdde1; font-size:13px; }
QGroupBox { border:1px solid #3a3c42; border-radius:8px; margin-top:14px; padding:12px 10px 10px 10px; font-weight:600; }
QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 4px; color:#9aa0a6; }
#card { background:#26282c; border:1px solid #3a3c42; border-radius:8px; }
#cardTitle { background:transparent; font-weight:600; }
#muted { background:transparent; color:#8b9097; font-size:12px; }
#warn { background:transparent; color:#e0a040; font-size:12px; }
#mono { background:#141518; border:1px solid #2e3035; border-radius:6px; padding:6px; color:#b8bcc2;
        font-family:"DejaVu Sans Mono", monospace; font-size:12px; }
#camView { background:#101113; border-radius:4px; color:#666; }
QPlainTextEdit, QLineEdit { background:#141518; border:1px solid #3a3c42; border-radius:6px; padding:6px; }
QPushButton { background:#33363c; border:1px solid #44474e; border-radius:6px; padding:8px 12px; }
QPushButton:hover { background:#3d4047; }
QPushButton:disabled { color:#5d636b; background:#2a2c30; border-color:#33363c; }
QPushButton#primary { background:#2f6feb; border-color:#2f6feb; color:white; font-weight:600; }
QPushButton#primary:hover { background:#4280f0; }
QPushButton#primary:disabled { background:#24324d; border-color:#24324d; color:#7a8597; }
QPushButton#danger { background:#8f2a2a; border-color:#a33; color:white; font-weight:600; }
QPushButton#danger:disabled { background:#3a2626; border-color:#3a2626; color:#7a6464; }
QPushButton#cmd { text-align:left; padding:9px 12px; }
#badge { border-radius:10px; padding:2px 10px; font-size:12px; font-weight:600; background:#33363c; color:#b8bcc2; }
#badge[level="ok"] { background:#1f4d33; color:#7ee2a8; }
#badge[level="bad"] { background:#4d2323; color:#ff9a9a; }
#badge[level="info"] { background:#1f3a5c; color:#8fc1ff; }
#mainInput { background:#14171e; border:2px solid #388bfd; border-radius:9px; padding:12px 18px;
            font-size:22px; font-weight:800; color:#ffffff; }
#mainInput:focus { border:2px solid #79c0ff; background:#181d26; }
#mainInput:disabled { background:#141518; border:1px solid #33363c; color:#5d636b; }
#btnSend { background:#1f6feb; border:1px solid #388bfd; border-radius:9px; padding:12px 24px;
          font-size:20px; font-weight:800; color:#ffffff; min-width:96px; }
#btnSend:hover { background:#388bfd; }
#btnSend:pressed { background:#1158c7; }
#btnSend:disabled { background:#23272e; border-color:#33363c; color:#5d636b; }
QSplitter::handle { background:transparent; }
QCheckBox { background:transparent; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", default=CONFIG["robot_ip"])
    ap.add_argument("--cmd", default=None, help="主程序启动命令（覆盖 CONFIG）")
    ap.add_argument("--mock", action="store_true", help="连接本地 mock_robot.py，并启动 mock_main.py")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    cfg["robot_ip"] = args.robot_ip
    if args.robot_ip != CONFIG["robot_ip"] and not args.cmd:
        cfg["main_cmd"] = cfg["main_cmd"].replace(f"--robot_ip={CONFIG['robot_ip']}", f"--robot_ip={args.robot_ip}")
    if args.mock:
        here = os.path.dirname(os.path.abspath(__file__))
        cfg["robot_ip"] = "127.0.0.1"
        cfg["main_cmd"] = f"python3 -u {here}/mock_main.py"
    if args.cmd:
        cfg["main_cmd"] = args.cmd

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow(cfg)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
