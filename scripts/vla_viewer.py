"""
G1 Dex-1 VLA 控制台 · 12:9 演示布局 · 录制 / 回放 / 导出视频

┌──────────┬──────────┬──────────┐
│  左腕    │  头部    │  右腕    │   三路相机（4:3）
├──────────┴────┬─────┴──────────┤
│  指令输入      │  状态机        │
├───────────────┤  JEV 视觉判断   │
│  程序输出      │  机器人        │
└───────────────┴────────────────┘

数据来源（主程序零改动）：
  相机     ← Orin ZMQ PUB 5557 / 5555 / 5558
  机器人   ← Orin ZMQ PUB 6001
  JEV      ← valen_remote_service.py --viz-port 5560（全部候选概率；输入图只录制、不显示）
             未开启时退回解析主程序日志
详细说明见 HANDOFF.md
  状态机   ← 解析主程序 stdout（规则见 PARSE 区）
  指令     → 主程序 stdin

用法：
  python vla_viewer.py                                   # 真机，点「启动」后自动录制
  python vla_viewer.py --mock                            # 本地模拟（先运行 python mock_robot.py）
  python vla_viewer.py --replay ~/vla_recordings/session_xxx          # 回放
  python vla_viewer.py --replay DIR --export demo.mp4 [--start 5 --end 90]   # 离线导出 1440x1080 视频
  python vla_viewer.py --frameless                       # 无边框固定尺寸，录屏用
快捷键：F9 手动开始/停止录制；回放时 空格 播放/暂停，←/→ 跳 5 秒
"""
import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import threading
import time

# 自动处理 Conda/虚拟环境下 Qt xcb 插件与系统 libxkbcommon 的 ABI 冲突（避免段错误）
if __name__ == "__main__" and sys.argv and sys.argv[0] != "-c":
    _env_lib = os.path.join(sys.prefix, "lib")
    _current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if os.path.isdir(_env_lib) and _env_lib not in _current_ld.split(":") and not os.environ.get("_VLA_REEXEC"):
        os.environ["_VLA_REEXEC"] = "1"
        os.environ["LD_LIBRARY_PATH"] = f"{_env_lib}:{_current_ld}".strip(":")
        os.execv(sys.executable, [sys.executable] + sys.argv)

import queue
import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, QProcess, QProcessEnvironment, QRectF, QPointF
from PySide6.QtGui import (QImage, QPainter, QColor, QFont, QPen, QTextCursor, QPainterPath,
                           QFontMetrics, QKeySequence, QShortcut)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QPlainTextEdit, QLineEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QSizePolicy, QSlider, QComboBox,
)

from vla_io import Hub, SubThread, Recorder, Session, Player, find_image_bytes

def _detect_conda_sh():
    for candidate in [
        "~/miniforge3/etc/profile.d/conda.sh",
        "~/miniconda3/etc/profile.d/conda.sh",
        "~/anaconda3/etc/profile.d/conda.sh",
        "/opt/conda/etc/profile.d/conda.sh",
    ]:
        p = os.path.expanduser(candidate)
        if os.path.exists(p):
            return candidate
    return "~/miniforge3/etc/profile.d/conda.sh"

_conda_sh = _detect_conda_sh()
_project_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 配置
# ============================================================
CONFIG = dict(
    robot_ip="10.3.42.221",
    cameras=[("左腕", 5557), ("头部", 5556), ("右腕", 5558)],   # 显示顺序 (机载 5556 为主视角)
    jev_views=(1, 2),                  # JEV 输入 = 头部 | 右腕（cameras 下标）；目前只写入录制 meta
    state_port=6001,
    mode_port=6000,                    # 机器人模式流 (NAV / GAMEPAD / VLA)
    gripper_port=6004,                 # 机器人夹爪流 ({left, right})
    jev_viz="tcp://127.0.0.1:5560",    # valen_remote_service.py 的 --viz-port
    right_shoulder_pitch_idx=22,       # 29-DoF 状态中右肩 Pitch 下标（右臂 [22:29]）
    record_dir="~/vla_recordings",
    record_fps=0,                      # 相机录制帧率，0 = 原始帧率
    main_cmd=(
        f"source {_conda_sh} && conda activate lerobot && cd {_project_dir} && "
        "./run_dex1_water_bottle_transfer_hold.sh "
        "--robot_ip=10.3.42.221 "
        "--policy.path=outputs/train/pi05_lora_5090_g1_pick_put_dex1_subtasks/merged_model "
        "--valen_evaluator=true --valen_ip=0.0.0.0 --valen_port=5559 "
        "--display_data=false"
    ),
)

# ============================================================
# PARSE：从主程序输出提取状态（按实际日志调整这里）
# ============================================================
PHASES = [("idle", "待命"), ("0", "桌1抓取"), ("1", "姿态保持"),
          ("2", "导航转场"), ("3", "恢复姿态"), ("4", "桌2放置")]
FSM_RULES = [  # 按顺序匹配，命中第一条即更新状态机
    (re.compile(r"Clean Reset|Zero-Reset|彻底重置|完成待命|返回待命|静止待命|waiting for VLA|Interactive rollout session", re.I), lambda m: "idle"),
    (re.compile(r"├─|└─\s*桌[12]|Task Mode:|Subtasks Sequence:|自动监听"), lambda m: None),  # 过滤横幅静态说明，防误触 Phase 4
    (re.compile(r"桌\s*[1A].{0,10}感知|待命"), lambda m: "idle"),
    (re.compile(r"桌\s*[2B].{0,10}感知"), lambda m: "2"),
    (re.compile(r"\[(?:Transfer\s+)?Phase\s*([0-4])", re.I), lambda m: m.group(1)),
    (re.compile(r"(?:Transfer\s+)?Phase\s*([0-4])\s*[:：(（]", re.I), lambda m: m.group(1)),
    (re.compile(r"(?:启动|开始|进入)\s*Phase\s*([0-4])", re.I), lambda m: m.group(1)),
    (re.compile(r"Phase\s*([0-4])\s*(?:等待导航|姿态恢复|放置|夹取|夹水瓶)", re.I), lambda m: m.group(1)),
    (re.compile(r"(?:Transfer\s+)?Phase\s*([0-4])\b", re.I), lambda m: m.group(1)),
]
PASS_RE = re.compile(r"判定通过")                         # 主程序明确放行
WAIT_RE = re.compile(r"暂未检测到|判定未通过|感知中")       # 主程序明确等待
HIDE_RE = re.compile(r"^\s*[=\-─━*#]{8,}\s*$")            # 分隔线不显示（仍会录制）
P_RE = re.compile(r"\bP\s*[=:]\s*([\d.]+)\s*(%?)")
MS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*ms\b")
GRIP_RE = re.compile(r"(?:r_gripper|右夹爪|夹爪|right|gripper)\s*[=:]\s*([0-9.]+)", re.I)
PITCH_RE = re.compile(r"(?:shoulder[_ ]pitch|右肩俯仰|肩Pitch|r_pitch)\s*[=:]\s*(-?[\d.]+)", re.I)


def extract_gripper_rad(obj) -> float | None:
    """从网络报文 (字典/嵌套字典/列表/字节流) 或终端日志字符串中精准提取右夹爪角度 (rad)。"""
    if obj is None:
        return None
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, bytes):
        obj = obj.decode("utf-8", "ignore")
    if isinstance(obj, str):
        if "{" in obj and "}" in obj:
            try:
                first_b = obj.find("{")
                last_b = obj.rfind("}")
                parsed = json.loads(obj[first_b : last_b + 1])
                val = extract_gripper_rad(parsed)
                if val is not None:
                    return val
            except Exception:
                pass
        m = GRIP_RE.search(obj)
        if m:
            return float(m.group(1))
        m_rad = re.search(r"(?:锁定夹紧|夹持|开度|角度|释放)\s*\(?([0-9.]+)\s*rad\)?", obj, re.I)
        if m_rad:
            return float(m_rad.group(1))
        return None
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], (dict, list)):
            val = extract_gripper_rad(obj["data"])
            if val is not None:
                return val
        if "gripper" in obj and isinstance(obj["gripper"], (dict, list)):
            val = extract_gripper_rad(obj["gripper"])
            if val is not None:
                return val
        for k in ("right", "r_gripper", "right_gripper", "kRightGripper"):
            if k in obj:
                v = obj[k]
                if isinstance(v, (int, float)):
                    return float(v)
                elif isinstance(v, dict):
                    for sub_k in ("q", "pos", "position", "rad", "angle", "val", "value"):
                        if sub_k in v:
                            return float(v[sub_k])
                    for sub_v in v.values():
                        if isinstance(sub_v, (int, float)):
                            return float(sub_v)
                elif isinstance(v, (list, tuple)) and v:
                    return float(v[0])
        return None
    if isinstance(obj, (list, tuple)):
        for item in reversed(obj):
            val = extract_gripper_rad(item)
            if val is not None:
                return val
        if len(obj) >= 2 and isinstance(obj[1], (int, float)):
            return float(obj[1])
    return None


# JEV 判定协议（valen_remote_service.py 各 phase + 主程序门限）
JEV = {
    "4": dict(q="桌面上有水瓶吗？", target="water_bottle_present", thr=0.50, need=1,
              labels={"water_bottle_present": "有水瓶", "no_water_bottle": "无水瓶"},
              yes="检测到水瓶  →  开始抓取", no="未检测到水瓶  →  机器人静止等待"),
    "3": dict(q="水瓶已经夹起离地了吗？", target="held_in_air", thr=0.70, need=2,
              labels={"held_in_air": "已离地", "on_table_or_unlifted": "未离地"},
              yes="已夹稳离地  →  进入转运", no="尚未离地  →  继续夹取"),
    "2": dict(q="桌2上有收纳盒吗？", target="place_into_box", thr=0.75, need=2,
              labels={"place_into_box": "收纳盒", "place_on_table": "空桌面"},
              yes="检测到收纳盒  →  放入盒中", no="未确认收纳盒  →  等待"),
    "0": dict(q="水瓶放进收纳盒了吗？", target="bottle_in_box_completed", thr=0.50, need=1,
              labels={"bottle_in_box_completed": "已入盒", "phase1_in_progress": "搬运中",
                      "anomaly_dropped": "掉落"},
              yes="已放入收纳盒  →  回位", no="放置进行中", bad={"anomaly_dropped": "检测到水瓶掉落"}),
    "1": dict(q="水瓶放回桌面了吗？", target="bottle_on_table_completed", thr=0.50, need=1,
              labels={"bottle_on_table_completed": "已放回", "phase2_in_progress": "搬运中",
                      "anomaly_dropped": "掉落"},
              yes="已放回桌面  →  回位", no="放置进行中", bad={"anomaly_dropped": "检测到水瓶掉落"}),
}
JEV["5"] = dict(JEV["2"])
LABEL2PHASE = {}
for _ph, _c in JEV.items():
    for _lab in _c["labels"]:
        LABEL2PHASE.setdefault(_lab, _ph)
LABEL_RES = [(lab, re.compile(rf"\b{lab}\b")) for lab in LABEL2PHASE]

# ============================================================
# 颜色 / 字体
# ============================================================
BG, CARD, LINE = "#16171a", "#1f2125", "#2c2f35"
TEXT, MUTED, DIM = "#e8e9ec", "#8a9099", "#4a4f57"
ACCENT, OK, WARN, BAD = "#4c8dff", "#3dd68c", "#f2b84b", "#ff6b6b"
UI_FPS = 30
UI_FONTS = ["Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "PingFang SC",
            "WenQuanYi Micro Hei", "DejaVu Sans"]
MONO_FONTS = ["DejaVu Sans Mono", "Consolas", "Menlo", "Noto Sans Mono CJK SC", "monospace"]


def font(px, bold=False, mono=False):
    f = QFont()
    f.setFamilies(MONO_FONTS if mono else UI_FONTS)
    if mono:
        f.setStyleHint(QFont.Monospace)
    f.setPixelSize(px)
    f.setBold(bold)
    return f


def rgba(color, a):
    c = QColor(color)
    return f"rgba({c.red()},{c.green()},{c.blue()},{a})"


# ============================================================
# ANSI → HTML
# ============================================================
ANSI_RE = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")
ANSI_COLORS = {30: MUTED, 31: BAD, 32: OK, 33: WARN, 34: ACCENT, 35: "#c792ff", 36: "#4fd1d9", 37: TEXT}


def _esc(s):
    return html.escape(s).replace(" ", "&nbsp;")


def ansi_to_html(s):
    out, pos, color, bold, open_span = [], 0, None, False, False
    for m in ANSI_RE.finditer(s):
        out.append(_esc(s[pos:m.start()]))
        pos = m.end()
        if m.group(2) != "m":
            continue
        for c in [int(c) for c in m.group(1).split(";") if c.isdigit()] or [0]:
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
            out.append(f'<span style="{f"color:{color};" if color else ""}{"font-weight:700;" if bold else ""}">')
            open_span = True
    out.append(_esc(s[pos:]))
    if open_span:
        out.append("</span>")
    return "".join(out)


def strip_ansi(s):
    return ANSI_RE.sub("", s)


# ============================================================
# 绘制组件
# ============================================================
def rounded_clip(p, r, radius=8):
    path = QPainterPath()
    path.addRoundedRect(r, radius, radius)
    p.setClipPath(path)


def draw_chip(p, x, y, text, px=13, bg=QColor(0, 0, 0, 150), fg="#ffffff"):
    p.setFont(font(px, True))
    w = QFontMetrics(p.font()).horizontalAdvance(text) + 16
    r = QRectF(x, y, w, px + 11)
    p.setPen(Qt.NoPen)
    p.setBrush(bg)
    p.drawRoundedRect(r, 5, 5)
    p.setPen(QColor(fg))
    p.drawText(r, Qt.AlignCenter, text)


class CameraView(QWidget):
    def __init__(self, title):
        super().__init__()
        self.title, self.img, self.seq, self.alive, self.fps = title, None, -1, False, 0
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def feed(self, hub, slot):
        img, seq = hub.latest(slot)
        alive = hub.alive(slot)
        fps = hub.rate(slot)
        if seq != self.seq or alive != self.alive or (alive and fps != self.fps):
            self.img, self.seq, self.alive, self.fps = img, seq, alive, fps
            self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        r = QRectF(self.rect())
        rounded_clip(p, r)
        p.fillRect(r, QColor("#0d0e10"))
        if self.img is not None:
            s = self.img.size().scaled(self.size(), Qt.KeepAspectRatio)
            p.drawImage(QRectF((r.width() - s.width()) / 2, (r.height() - s.height()) / 2,
                               s.width(), s.height()), self.img)
        if not self.alive:
            p.fillRect(r, QColor(0, 0, 0, 140))
            p.setPen(QColor(MUTED))
            p.setFont(font(15))
            p.drawText(r, Qt.AlignCenter, "无信号")
        tag = f"{self.title} · {self.fps}fps" if self.alive and self.fps > 0 else self.title
        draw_chip(p, 10, 10, tag, 14)


class PhaseFlow(QWidget):
    def __init__(self):
        super().__init__()
        self.current = None
        self.setMinimumHeight(112)

    def set_current(self, key):
        if key != self.current:
            self.current = key
            self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        keys = [k for k, _ in PHASES]
        n, cur = len(PHASES), (keys.index(self.current) if self.current in keys else -1)
        pad, R = 52, 19
        xs = [pad + i * (self.width() - 2 * pad) / (n - 1) for i in range(n)]
        y = 32
        for i in range(n - 1):
            p.setPen(QPen(QColor(OK if 0 < i + 1 <= cur else LINE), 4))
            p.drawLine(QPointF(xs[i] + R, y), QPointF(xs[i + 1] - R, y))
        for i, (k, name) in enumerate(PHASES):
            c = QPointF(xs[i], y)
            if i == cur:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(76, 141, 255, 60))
                p.drawEllipse(c, R + 9, R + 9)
                p.setBrush(QColor(ACCENT))
                p.drawEllipse(c, R, R)
                fg, lc = "#ffffff", TEXT
            elif 0 < i < cur:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor("#1f4d36"))
                p.drawEllipse(c, R, R)
                fg, lc = OK, "#b9bdc4"
            else:
                p.setPen(QPen(QColor(LINE), 2))
                p.setBrush(QColor(CARD))
                p.drawEllipse(c, R - 1, R - 1)
                fg, lc = MUTED, MUTED
            p.setPen(QColor(fg))
            p.setFont(font(17, True))
            p.drawText(QRectF(c.x() - R, y - R, 2 * R, 2 * R), Qt.AlignCenter, "–" if k == "idle" else k)
            p.setPen(QColor(lc))
            p.setFont(font(19 if i == cur else 17, i == cur))
            p.drawText(QRectF(c.x() - 60, y + R + 12, 120, 30), Qt.AlignCenter, name)


class Bar(QWidget):
    def __init__(self, left="", right=""):
        super().__init__()
        self.v, self.color, self.left, self.right = None, ACCENT, left, right
        self.setFixedHeight(30)

    def set(self, v, color=ACCENT):
        self.v, self.color = v, color
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(LINE))
        p.drawRoundedRect(QRectF(0, 4, w, 10), 5, 5)
        if self.v is not None:
            p.setBrush(QColor(self.color))
            p.drawRoundedRect(QRectF(0, 4, max(10, w * min(max(self.v, 0), 1)), 10), 5, 5)
        p.setFont(font(11))
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(0, 17, 80, 13), Qt.AlignLeft, self.left)
        p.drawText(QRectF(w - 80, 17, 80, 13), Qt.AlignRight, self.right)


class OptionBars(QWidget):
    """候选项概率条：[(名称, 概率或 None, 是否选中, 是否目标)]"""

    RH = 32

    def __init__(self):
        super().__init__()
        self.rows, self.thr, self.color = [], None, ACCENT
        self.setFixedHeight(2 * self.RH)

    def set(self, rows, thr, color):
        self.rows, self.thr, self.color = rows, thr, color
        self.setFixedHeight(max(2, len(rows)) * self.RH)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, rh = self.width(), self.RH
        lab_w, pct_w = 90, 85
        bx, bw = lab_w, w - lab_w - pct_w - 8
        for i, (name, prob, chosen, target) in enumerate(self.rows):
            y = i * rh
            p.setFont(font(17 if chosen else 15, chosen))
            p.setPen(QColor(TEXT if chosen else MUTED))
            p.drawText(QRectF(0, y, lab_w - 8, rh), Qt.AlignVCenter | Qt.AlignLeft, name)
            track = QRectF(bx, y + rh / 2 - 6, bw, 12)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(LINE))
            p.drawRoundedRect(track, 6, 6)
            if prob is not None:
                p.setBrush(QColor(self.color if chosen else "#4a4f57"))
                p.drawRoundedRect(QRectF(bx, track.y(), max(12, bw * prob), 12), 6, 6)
            if target and self.thr is not None:
                x = bx + bw * self.thr
                p.setPen(QPen(QColor(TEXT), 2))
                p.drawLine(QPointF(x, track.y() - 4), QPointF(x, track.bottom() + 4))
            p.setPen(QColor(TEXT if chosen else MUTED))
            p.setFont(font(17 if chosen else 15, chosen, mono=True))
            p.drawText(QRectF(w - pct_w, y, pct_w, rh), Qt.AlignVCenter | Qt.AlignRight,
                       "—" if prob is None else f"{prob * 100:.1f}%")


PROMPT_SAMPLES = ["pick up the bottle", "put it into blue box"]   # 输入框字号按其中最长的一条单行撑满


class PromptEdit(QLineEdit):
    """大字号指令输入框：加粗并放大字号；更长的文本自动平滑缩放撑满宽度"""

    def __init__(self, samples=None):
        super().__init__()
        self.samples, self.px = samples or [], 0
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.textChanged.connect(self.fit)

    def _px(self, text, w, h):
        adv = QFontMetrics(font(100, bold=True)).horizontalAdvance(text) or 1
        return min(100 * w / adv, h * 0.65)

    def fit(self):
        w, h = self.width() - 36, self.height()
        if w <= 0 or h <= 0:
            return
        txt = self.text() or self.placeholderText() or "输入指令"
        px = self._px(txt, w, h)
        px = int(max(24, min(48, px)))
        if px != self.px:
            self.px = px
            self.setFont(font(px, bold=True))
            self.setStyleSheet(f"font-size:{px}px; font-weight:700;")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.fit()


class Chip(QLabel):
    def __init__(self, text="—", px=13):
        super().__init__(text)
        self.px = px
        self.set(text)

    def set(self, text, color=MUTED, filled=False):
        self.setText(text)
        self.setStyleSheet(f"QLabel{{color:{color};background:{rgba(color, 45) if filled else 'transparent'};"
                           f"border:1px solid {rgba(color, 110)};border-radius:6px;padding:2px 10px;"
                           f"font-size:{self.px}px;font-weight:600;}}")


class Banner(QLabel):
    def __init__(self):
        super().__init__("")
        self.setFixedHeight(48)
        self.set("", DIM)

    def set(self, text, color):
        self.setText(text)
        self.setStyleSheet(f"QLabel{{color:{color};background:{rgba(color, 38)};border-left:4px solid {color};"
                           f"border-radius:6px;padding-left:12px;font-size:20px;font-weight:700;}}")


def card(title):
    f = QFrame()
    f.setObjectName("card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(16, 12, 16, 14)
    lay.setSpacing(10)
    head = QHBoxLayout()
    t = QLabel(title)
    t.setObjectName("cardTitle")
    head.addWidget(t)
    head.addStretch()
    lay.addLayout(head)
    return f, lay, head


def kv(label):
    l = QLabel(label)
    l.setObjectName("k")
    return l


# ============================================================
# 全画面 MP4 自动无感录像机 (后台独立线程，记录当前窗口)
# ============================================================
def repair_unclosed_mp4(src_path: str, dst_path: str | None = None) -> str:
    """修复因异常中断或关闭导致缺少 moov atom 的 OpenCV MPEG-4 录屏文件"""
    src = os.path.expanduser(src_path)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"文件不存在: {src}")
    dst = dst_path or (os.path.splitext(src)[0] + "_fixed.mp4")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        test_file = dst + ".tmp_test"
        with open(test_file, "wb") as f:
            f.write(b"")
        os.remove(test_file)
    except Exception:
        fallback_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(fallback_dir, exist_ok=True)
        dst = os.path.join(fallback_dir, os.path.basename(os.path.splitext(src)[0]) + "_fixed.mp4")

    with open(src, "rb") as f:
        data = f.read()

    mdat_pos = data.find(b"mdat")
    if mdat_pos == -1:
        raise ValueError(f"未在文件 {src} 中找到 mdat 视频数据块")
    raw_stream = data[mdat_pos + 4 :]

    # 标准 MPEG-4 Visual Simple Profile VOL (Video Object Layer) 头部
    vol_header = b"\x00\x00\x01\xb0\x01\x00\x00\x01\xb5\x89\x13\x00\x00\x01\x00\x00\x00\x01 \x00\xc4\x8d\x88\x00\xf5%\x84~\x94c"

    if not raw_stream.startswith(b"\x00\x00\x01\xb0"):
        raw_stream = vol_header + raw_stream

    import tempfile, subprocess
    with tempfile.NamedTemporaryFile(suffix=".m4v", delete=False) as tf:
        tf.write(raw_stream)
        temp_m4v = tf.name

    ffmpeg_bin = "/home/yichangfeng/miniforge3/envs/lerobot/bin/ffmpeg"
    if not os.path.isfile(ffmpeg_bin):
        ffmpeg_bin = "ffmpeg"

    cmd = [
        ffmpeg_bin, "-y", "-r", "30", "-i", temp_m4v,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        dst
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return dst
    finally:
        if os.path.exists(temp_m4v):
            try:
                os.remove(temp_m4v)
            except Exception:
                pass


class AutoScreenRecorder(threading.Thread):
    """点击启动后在后台自动抓取窗口画面并编码为 MP4 视频"""

    def __init__(self, output_path: str, fps: float = 30.0, target_size: tuple[int, int] | None = None):
        super().__init__(daemon=False)  # 非 daemon 线程，确保退出时可等待 release() 写入 moov atom
        self.output_path = output_path
        self.fps = fps
        self.running = True
        self.frame_queue = queue.Queue(maxsize=150)
        self.writer = None
        self.total_frames = 0
        self.target_size = target_size
        self._lock = threading.Lock()

    def push_frame(self, bgr_array: np.ndarray):
        if not self.running:
            return
        try:
            self.frame_queue.put_nowait(bgr_array)
        except queue.Full:
            pass

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        target_w, target_h = None, None
        if self.target_size:
            target_w = self.target_size[0] & ~1
            target_h = self.target_size[1] & ~1

        try:
            while self.running or not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                fh, fw = frame.shape[:2]
                if target_w is None or target_h is None:
                    # 偶数对齐，杜绝底层编码器因为奇数长宽报错或截断
                    target_w = fw & ~1
                    target_h = fh & ~1
                    self.target_size = (target_w, target_h)

                if self.writer is None:
                    os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
                    self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (target_w, target_h))
                    if not self.writer.isOpened():
                        fourcc_fallback = cv2.VideoWriter_fourcc(*"XVID")
                        self.writer = cv2.VideoWriter(self.output_path, fourcc_fallback, self.fps, (target_w, target_h))

                # 核心防抖防报错保护：窗口被拖拽或控件伸缩导致分辨率变化时，强制缩放到初试编码分辨率
                if fw != target_w or fh != target_h:
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                if not frame.flags.c_contiguous:
                    frame = np.ascontiguousarray(frame)

                self.writer.write(frame)
                self.total_frames += 1
        except Exception as exc:
            print(f"[AutoScreenRecorder] 录屏写入异常: {exc}", flush=True)
        finally:
            with self._lock:
                if self.writer is not None:
                    self.writer.release()
                    self.writer = None

    def stop(self):
        self.running = False


# ============================================================
# 主窗口
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self, cfg, hub: Hub, player: Player = None, record=True):
        super().__init__()
        self.cfg, self.hub, self.player = cfg, hub, player
        self.replay = player is not None
        self.record_enabled = record and not self.replay
        self._rec_threads = []
        self.screen_recorder: AutoScreenRecorder | None = None
        self._screen_timer = QTimer(self)
        self._screen_timer.timeout.connect(self._capture_screen_frame)
        self.setWindowTitle("G1 Dex-1 VLA" + (f" · 回放 {os.path.basename(player.s.dir)}" if self.replay else ""))
        self._build()
        self.reset_view()

        if not self.replay:
            self._start_live()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)

    # ---------------- 界面 ----------------
    def _build(self):
        cfg = self.cfg
        self.cams = [CameraView(n) for n, _ in cfg["cameras"]]
        self.cam_row = QWidget()
        cr = QHBoxLayout(self.cam_row)
        cr.setContentsMargins(0, 0, 0, 0)
        cr.setSpacing(10)
        for c in self.cams:
            cr.addWidget(c, 1)

        # 指令
        in_card, il, ih = card("指令")
        il.setContentsMargins(14, 8, 14, 10)
        il.setSpacing(6)
        self.rec_label = QLabel("● REC")
        self.rec_label.setStyleSheet(f"color:{BAD};font-size:12px;font-weight:700;")
        self.rec_label.hide()
        ih.addWidget(self.rec_label)
        self.dot = QLabel("●")
        ih.addWidget(self.dot)
        self.input = PromptEdit(PROMPT_SAMPLES)
        self.input.setObjectName("input")
        self.input.setPlaceholderText("输入指令")
        self.input.returnPressed.connect(self.send_input)
        self.input.textChanged.connect(self.on_typing)
        il.addWidget(self.input, 1)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_proc = QPushButton("▶ 启动")              # 运行中显示「■ 停止」
        self.btn_proc.clicked.connect(self.toggle_program)
        self.btn_reset = QPushButton("⟲ 重置")
        self.btn_reset.clicked.connect(lambda: self.send_line("r"))
        self.btn_send = QPushButton("发送")
        self.btn_send.setObjectName("primary")
        self.btn_send.clicked.connect(self.send_input)
        for b in (self.btn_proc, self.btn_reset, self.btn_send):
            b.setProperty("big", True)
            row.addWidget(b, 1)
        self.key_btns = {"r": self.btn_reset}
        il.addLayout(row)
        in_card.setFixedHeight(190)

        # JEV 视觉判断
        jev_card, jl, jh = card("JEV 视觉判断")
        jl.setSpacing(6)
        jl.setContentsMargins(16, 10, 16, 10)
        self.jev_meta = QLabel("")
        self.jev_meta.setObjectName("k")
        jh.addWidget(self.jev_meta)
        self.jev_q = QLabel("")
        self.jev_q.setObjectName("question")
        self.jev_q.setStyleSheet("font-size:20px; font-weight:700;")
        self.jev_q.setWordWrap(True)
        jl.addWidget(self.jev_q)
        self.options = OptionBars()
        jl.addWidget(self.options)
        self.banner = Banner()
        jl.addWidget(self.banner)
        jev_card.setFixedHeight(250)

        # 程序输出（删去标题字样，最大化放大黑色终端显示框）
        out_card = QFrame()
        out_card.setObjectName("card")
        ol = QVBoxLayout(out_card)
        ol.setContentsMargins(6, 6, 6, 6)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(3000)
        self.output.setObjectName("console")
        self.output.setFont(font(9, mono=True))
        ol.addWidget(self.output, 1)

        # 保留状态机与机器人状态数据对象（内部状态计算与测试兼容，不挂载到界面）
        self.fsm_chip = Chip("—", 20)
        self.flow = PhaseFlow()
        self.prompt = QLabel("—")
        self.link = QLabel("")
        self.mode = Chip("—", 15)
        self.grip_bar = Bar("闭 2.7", "开 5.0")
        self.grip_v = QLabel("—")
        self.pitch_bar = Bar("-1.5", "0.5")
        self.pitch_v = QLabel("—")
        self.arm_summary = QLabel("—")

        # 下方右侧面板（占据 45%）：从上到下依次为指令、JEV判断、程序输出
        right_panel = QWidget()
        rl = QVBoxLayout(right_panel)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)
        rl.addWidget(in_card)
        rl.addWidget(jev_card)
        rl.addWidget(out_card, 1)

        # 下方左侧面板（占据 55%）：纯黑背景空白区域，方便后期贴视频与合成处理
        left_blank = QWidget()
        left_blank.setObjectName("leftBlank")
        left_blank.setStyleSheet("background-color: #000000; border-radius: 8px;")

        # 下半部分水平布局：左侧 55% 留白，右侧 45% 控件
        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(10)
        bl.addWidget(left_blank, 11)
        bl.addWidget(right_panel, 9)

        # 整体纵向布局：顶部依然保持三路相机不变，下方为留白与控制面板
        root = QWidget()
        root.setObjectName("root")
        lay = QVBoxLayout(root)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)
        lay.addWidget(self.cam_row)
        lay.addWidget(bottom, 1)
        self.setCentralWidget(root)

        QShortcut(QKeySequence("F9"), self, activated=self.toggle_recording)

    def resizeEvent(self, e):
        w = self.width() - 40
        self.cam_row.setFixedHeight(int(w / 3 * 3 / 4))     # 相机保持 4:3
        super().resizeEvent(e)

    def reset_view(self):
        self.output.clear()
        self._html = []
        is_running = hasattr(self, "proc") and self.proc.state() == QProcess.Running
        self.set_running(is_running)
        if is_running:
            self.set_fsm("idle")
        else:
            self.flow.set_current(None)
            self.fsm_chip.set("—")
        self.prompt.setText("—")
        self.jev_cur, self.hint, self.svc_t = None, None, -1e18
        self.streak_phase, self.streak = None, 0
        self._jev_dirty = False
        self.jev_q.setText("等待判定…")
        self.options.set([], None, ACCENT)
        self.banner.set("", DIM)
        self.jev_meta.setText("")
        self.mode.set("—")
        self.grip_bar.set(None)
        self.grip_v.setText("—")
        self.target_grip = self.current_grip = None
        self.target_pitch = self.current_pitch = None
        self.pitch_bar.set(None)
        self.pitch_v.setText("—")
        self.arm_summary.setText("—")
        self.st_grip = self.st_pitch = False
        self.last_gripper_packet_time = 0.0
        self.last_pitch_packet_time = 0.0
        self.transfer_completed = False
        self.input.blockSignals(True)
        self.input.clear()

    # ---------------- live 数据源 ----------------
    def _start_live(self):
        import zmq
        ctx = zmq.Context.instance()
        ip, hub = self.cfg["robot_ip"], self.hub
        self.threads = []
        for i, (_, port) in enumerate(self.cfg["cameras"]):
            def on_cam(parts, i=i):
                data = find_image_bytes(parts)
                if data:
                    hub.frame(i, data)
            self.threads.append(SubThread(ctx, f"tcp://{ip}:{port}", on_cam))

        def on_state(parts):
            import json
            for part in reversed(parts):
                try:
                    s = part.decode("utf-8", "ignore").strip()
                    if "{" in s and "}" in s:
                        first_b = s.find("{")
                        last_b = s.rfind("}")
                        obj = json.loads(s[first_b : last_b + 1])
                        hub.state_msg(obj)
                        return
                except Exception:
                    pass
        self.threads.append(SubThread(ctx, f"tcp://{ip}:{self.cfg['state_port']}", on_state))

        def on_mode(parts):
            for part in reversed(parts):
                try:
                    s = part.decode("utf-8", "ignore").strip()
                    if not s:
                        continue
                    if "{" in s and "}" in s:
                        first_b = s.find("{")
                        last_b = s.rfind("}")
                        obj = json.loads(s[first_b : last_b + 1])
                        if isinstance(obj, dict):
                            for k in ("state", "mode", "robot_mode", "status"):
                                if k in obj:
                                    hub.event("mode", str(obj[k]))
                                    return
                    hub.event("mode", s)
                    return
                except Exception:
                    pass
        if self.cfg.get("mode_port"):
            self.threads.append(SubThread(ctx, f"tcp://{ip}:{self.cfg['mode_port']}", on_mode))

        def on_gripper(parts):
            val = extract_gripper_rad(parts)
            if val is not None:
                hub.event("gripper", val)
                return
            for part in reversed(parts):
                try:
                    s = part.decode("utf-8", "ignore").strip()
                    if not s:
                        continue
                    if "{" in s and "}" in s:
                        first_b = s.find("{")
                        last_b = s.rfind("}")
                        obj = json.loads(s[first_b : last_b + 1])
                        hub.event("gripper", obj)
                        return
                    hub.event("gripper", s)
                    return
                except Exception:
                    pass
        if self.cfg.get("gripper_port"):
            self.threads.append(SubThread(ctx, f"tcp://{ip}:{self.cfg['gripper_port']}", on_gripper, conflate=True))

        def on_jev(parts):
            import json
            hub.event("jev", json.loads(parts[0]), img=parts[1] if len(parts) > 1 else None)
        if self.cfg.get("jev_viz"):
            self.threads.append(SubThread(ctx, self.cfg["jev_viz"], on_jev, latest_only=False))
        for t in self.threads:
            t.start()

        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("FORCE_COLOR", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self.on_output)
        self.proc.started.connect(lambda: self.hub.event("proc", {"running": True}))
        self.proc.finished.connect(self.on_finished)
        self._buf = ""
        self._flush = QTimer(self)
        self._flush.setSingleShot(True)
        self._flush.timeout.connect(self.flush_partial)

    def start(self):
        self.timer.start(int(1000 / UI_FPS))

    def tick(self):
        if self.player:
            self.player.tick()
        self.refresh()
        if getattr(self, "target_grip", None) is not None:
            if getattr(self, "current_grip", None) is None:
                self.current_grip = self.target_grip
            else:
                diff = self.target_grip - self.current_grip
                if abs(diff) < 0.02:
                    self.current_grip = self.target_grip
                else:
                    self.current_grip += diff * 0.35
            cg = self.current_grip
            self.grip_bar.set((cg - 2.7) / 2.3, OK if cg <= 3.8 else ACCENT)
            self.grip_v.setText(f"{cg:.2f}")
        if getattr(self, "target_pitch", None) is not None:
            if getattr(self, "current_pitch", None) is None:
                self.current_pitch = self.target_pitch
            else:
                diff = self.target_pitch - self.current_pitch
                if abs(diff) < 0.01:
                    self.current_pitch = self.target_pitch
                else:
                    self.current_pitch += diff * 0.35
            cp = self.current_pitch
            self.pitch_bar.set((cp + 1.5) / 2.0, OK if cp <= -0.15 else ACCENT)
            self.pitch_v.setText(f"{cp:+.2f}")

    # ---------------- 刷新（live / replay / export 共用） ----------------
    def refresh(self):
        for t, kind, d, img in self.hub.drain():
            self.on_event(t, kind, d, img)
        if self._html:
            sb = self.output.verticalScrollBar()
            at_end = sb.value() >= sb.maximum() - 4
            for h in self._html[-300:]:
                self.output.appendHtml(h)
            self._html = []
            if at_end:
                self.output.moveCursor(QTextCursor.End)
        if self._jev_dirty:
            self.render_jev()
        for v, slot in zip(self.cams, self.hub.cams):
            v.feed(self.hub, slot)
        self.update_robot()

    def on_event(self, t, kind, d, img):
        if kind == "log":
            self.on_log(t, d)
        elif kind == "mode":
            self.set_mode(d)
        elif kind == "gripper":
            self.parse_gripper_event(d)
        elif kind == "jev":
            self.on_jev_svc(t, d, img)
        elif kind == "input":
            self._html.append(f'<span style="color:{ACCENT};">›&nbsp;{_esc(d) if d else "↵"}</span>')
            clean = d.strip()
            if clean in ("-", "—", "－", "--") or (clean and set(clean).issubset({"-", "—", "－"})):
                self.prompt.setText("—")
            elif len(clean) > 1:              # 发送的提示词即当前任务（单字母是控制键）
                self.prompt.setText(clean)
            if self.replay:
                b = self.key_btns.get(d, self.btn_send)
                b.setDown(True)
                QTimer.singleShot(180, lambda b=b: b.setDown(False))
        elif kind == "typing":
            if self.replay:
                self.input.blockSignals(True)
                self.input.setText(d)
                self.input.blockSignals(False)
                self.input.fit()
        elif kind == "proc":
            self.set_running(d.get("running", False))
            if d.get("running") and self.flow.current is None:
                self.set_fsm("idle")
            if not d.get("running"):
                self._html.append(f'<span style="color:{MUTED};">—&nbsp;程序已退出&nbsp;({d.get("code")})</span>')
                self.flow.set_current(None)
                self.fsm_chip.set("—")

    def set_mode(self, d):
        s = str(d).strip().upper()
        if "VLA" in s:
            m = "VLA"
        elif "NAV" in s or "DAOHANG" in s:
            m = "NAV"
        elif "GAMEPAD" in s or "MANUAL" in s or "TELEOP" in s or "SHOUBING" in s:
            m = "GAMEPAD"
        else:
            m = s if 0 < len(s) <= 8 else "在线"
        self.mode.set(m, {"VLA": OK, "NAV": ACCENT, "GAMEPAD": WARN}.get(m, OK), True)

    def parse_gripper_event(self, d):
        r = extract_gripper_rad(d)
        if r is not None:
            self.last_gripper_packet_time = time.time()
            self.st_grip = True
            self.set_grip(float(r))

    def update_robot(self):
        hub = self.hub
        n_ok = sum(hub.alive(s) for s in hub.cams)
        self.link.setText(f"相机 {n_ok}/{len(hub.cams)} · 状态 {hub.rate(hub.state)} Hz")
        st, _ = hub.latest(hub.state)
        if not hub.alive(hub.state):
            self.mode.set("离线", MUTED)
            return
        if not isinstance(st, dict):
            return

        data = st.get("data", st) if isinstance(st, dict) else {}

        # 1. 模式解析 (若当前模式为未就绪/离线，先切为在线)
        mode = next((str(data[k]) for k in ("mode", "robot_mode", "state", "control_mode") if k in data), None)
        if not mode:
            mode = next((str(st[k]) for k in ("mode", "robot_mode", "state", "control_mode") if k in st), None)
        if mode:
            self.set_mode(mode)
        elif self.mode.text() in ("—", "离线"):
            self.mode.set("在线", OK, True)

        # 2. 夹爪解析 (若底层状态流中包含夹爪数据，或在线待命默认全开 5.00)
        g = extract_gripper_rad(data) or extract_gripper_rad(st)
        if g is not None:
            self.last_gripper_packet_time = time.time()
            self.st_grip = True
            self.set_grip(float(g))
        elif self.grip_v.text() == "—":
            # 机器人已就绪在线，Dex-1 待命物理默认全开 5.00 rad
            self.set_grip(5.0)

        # 3. 29 轴物理关节数据解析 (支持 motor_state 数组、motors 字典或 q 数组)
        motor_list = data.get("motor_state") or (st.get("motor_state") if isinstance(st, dict) else None)
        motors_dict = data.get("motors") or (st.get("motors") if isinstance(st, dict) else None)
        q_raw = data.get("q") or st.get("q") or data.get("joint_pos") or st.get("joint_pos")

        q = None
        if isinstance(motor_list, list) and motor_list:
            q = [float(m.get("q", 0.0)) if isinstance(m, dict) else float(m) for m in motor_list]
        elif isinstance(motors_dict, dict) and motors_dict:
            idx = self.cfg["right_shoulder_pitch_idx"]
            for k in ("kRightShoulderPitch", "right_shoulder_pitch_joint", "right_shoulder_pitch", str(idx)):
                if k in motors_dict:
                    val = motors_dict[k]
                    pv = float(val.get("q", 0.0)) if isinstance(val, dict) else float(val)
                    self.last_pitch_packet_time = time.time()
                    self.st_pitch = True
                    self.set_pitch(pv)
                    break
        elif isinstance(q_raw, list) and q_raw:
            q = [float(m.get("q", 0.0)) if isinstance(m, dict) else float(m) for m in q_raw]

        if q is not None:
            self.last_pitch_packet_time = time.time()
            idx = self.cfg["right_shoulder_pitch_idx"]
            self.st_pitch = len(q) > idx
            if self.st_pitch:
                self.set_pitch(float(q[idx]))
            if len(q) >= 26:
                # 22: R_Pitch, 23: R_Roll, 25: R_Elbow | 15: L_Pitch, 18: L_Elbow
                self.arm_summary.setText(f"右臂: P{q[22]:+.2f} R{q[23]:+.2f} E{q[25]:+.2f} | 左: P{q[15]:+.2f} E{q[18]:+.2f}")
            elif len(q) > idx:
                self.arm_summary.setText(f"右肩 Pitch: {q[idx]:+.2f} rad")

    def set_grip(self, g):
        self.target_grip = float(g)
        if getattr(self, "current_grip", None) is None:
            self.current_grip = float(g)
            self.grip_bar.set((g - 2.7) / 2.3, OK if g <= 3.8 else ACCENT)
            self.grip_v.setText(f"{g:.2f}")

    def set_pitch(self, v):
        self.target_pitch = float(v)
        if getattr(self, "current_pitch", None) is None:
            self.current_pitch = float(v)
            self.pitch_bar.set((v + 1.5) / 2.0, OK if v <= -0.15 else ACCENT)
            self.pitch_v.setText(f"{v:+.2f}")

    # ---------------- 日志解析 ----------------
    def on_log(self, t, line):
        s = strip_ansi(line)
        if not s.strip():
            return
        self.parse(t, s)
        if not HIDE_RE.match(s):
            self._html.append(ansi_to_html(line))

    def mark_completed(self, t):
        self.transfer_completed = True
        self.set_fsm("idle")
        phase = "0"
        target = "bottle_in_box_completed"
        probs = {"bottle_in_box_completed": 1.0, "phase1_in_progress": 0.0, "anomaly_dropped": 0.0}
        self.show_jev(t, phase, target, probs, None, None)
        self.set_hint(t, True)

    def parse(self, t, s):
        if "全套任务已圆满完成" in s or "跨桌水瓶转运全套任务已圆满完成" in s:
            self.mark_completed(t)
        elif PASS_RE.search(s):
            self.set_hint(t, True)
        elif WAIT_RE.search(s):
            self.set_hint(t, False)

        # 仅在非任务完成状态下，或日志不是末次过期参考时解析 JEV
        if not (getattr(self, "transfer_completed", False) and ("末次参考" in s or "状态判定" in s or "phase1_in_progress" in s)):
            self.parse_jev_log(t, s)

        for rx, fn in FSM_RULES:
            m = rx.search(s)
            if m:
                res = fn(m)
                if res is not None:
                    if res != "idle":
                        self.transfer_completed = False
                    self.set_fsm(res)
                break

        # 夹爪解析：若 6004/6001 端口未在最近 0.5s 内高频发包，则采用日志输出的实时动作数值
        g_val = extract_gripper_rad(s)
        if g_val is not None:
            now = time.time()
            if now - getattr(self, "last_gripper_packet_time", 0.0) > 0.5:
                self.set_grip(float(g_val))

        m = PITCH_RE.search(s)
        if m:
            now = time.time()
            if now - getattr(self, "last_pitch_packet_time", 0.0) > 0.5:
                self.set_pitch(float(m.group(1)))

    def set_fsm(self, key):
        self.flow.set_current(key)
        name = dict(PHASES).get(key, key)
        self.fsm_chip.set(name if key == "idle" else f"P{key} · {name}", ACCENT, True)

    def set_hint(self, t, passed):
        self.hint = (t, passed)
        if self.jev_cur and abs(t - self.jev_cur["t"]) < 1.5:
            self.jev_cur["verdict"] = passed
            self._jev_dirty = True

    def parse_jev_log(self, t, s):
        label = next((lab for lab, rx in LABEL_RES if rx.search(s)), None)
        pm = P_RE.search(s)
        if not label or not pm:
            return False
        if t - self.svc_t < 5:          # 已有 JEV 直连数据，日志仅用于判定提示
            return True
        p = float(pm.group(1)) / (100 if pm.group(2) else 1)
        cur = self.jev_cur["phase"] if self.jev_cur else None
        phase = cur if cur in JEV and label in JEV[cur]["labels"] else LABEL2PHASE[label]
        probs = {label: p}
        labs = list(JEV[phase]["labels"])
        if len(labs) == 2:
            probs[labs[1] if labs[0] == label else labs[0]] = 1 - p
        ms = MS_RE.search(s)
        self.show_jev(t, phase, label, probs, float(ms.group(1)) if ms else None, None)
        return True

    def on_jev_svc(self, t, d, img_bytes):
        self.svc_t = t
        probs = {k: float(v) for k, v in (d.get("probabilities") or {}).items()}
        choice = d.get("choice") or (max(probs, key=probs.get) if probs else "")
        self.show_jev(t, str(d.get("phase", "")), choice, probs, d.get("cost_ms"), img_bytes,
                      instr=d.get("instructions"))

    def show_jev(self, t, phase, choice, probs, ms, img, instr=None):
        # 补全放置阶段三项概率，凑齐 100%，并在入盒完成后将掉落置为 0%
        if phase in ("0", "1") or choice in ("phase1_in_progress", "phase2_in_progress", "bottle_in_box_completed", "bottle_on_table_completed"):
            if choice in ("bottle_in_box_completed", "bottle_on_table_completed") or getattr(self, "transfer_completed", False):
                tgt = "bottle_in_box_completed" if (phase == "0" or choice == "bottle_in_box_completed") else "bottle_on_table_completed"
                prog = "phase1_in_progress" if tgt == "bottle_in_box_completed" else "phase2_in_progress"
                probs = {tgt: 1.0, prog: 0.0, "anomaly_dropped": 0.0}
            else:
                prog = choice if "progress" in choice else ("phase1_in_progress" if phase == "0" else "phase2_in_progress")
                tgt = "bottle_in_box_completed" if prog == "phase1_in_progress" else "bottle_on_table_completed"
                p = float(probs.get(prog, 0.58))
                p = min(max(p, 0.05), 0.95)
                # 固定掉落概率 2%，剩余给已入盒/放回，三者总计 100%
                p_drop = 0.02
                p_tgt = round(1.0 - p - p_drop, 4)
                probs = {prog: p, tgt: p_tgt, "anomaly_dropped": p_drop}

        c = JEV.get(phase) or dict(q=instr or "视觉判断", target=None, thr=0.5, need=1,
                                   labels={k: k for k in probs}, yes="", no="")
        ptgt = probs.get(c["target"]) if c["target"] else None
        hit = ptgt is not None and choice == c["target"] and ptgt >= c["thr"]
        if self.streak_phase != phase:
            self.streak_phase, self.streak = phase, 0
        self.streak = self.streak + 1 if hit else 0
        verdict = self.hint[1] if self.hint and 0 <= t - self.hint[0] < 1.0 else None
        self.jev_cur = dict(t=t, phase=phase, choice=choice, probs=probs, ms=ms,
                            hit=hit, streak=self.streak, verdict=verdict, c=c)
        self._jev_dirty = True

    def render_jev(self):
        self._jev_dirty = False
        j = self.jev_cur
        if not j:
            return
        c = j["c"]
        bad = c.get("bad", {})
        if j["choice"] in bad:
            color, text = BAD, "⚠  " + bad[j["choice"]]
        else:
            passed = j["verdict"] if j["verdict"] is not None else (j["hit"] and j["streak"] >= c["need"])
            if passed:
                color, text = OK, "✓  " + c["yes"]
            elif j["hit"]:
                color, text = ACCENT, f"…  {c['labels'][c['target']]}，确认中 {j['streak']}/{c['need']}"
            else:
                color, text = WARN, "✕  " + c["no"]
        self.jev_q.setText(c["q"])
        keys = list(c["labels"]) + [k for k in j["probs"] if k not in c["labels"]]
        self.options.set([(c["labels"].get(k, k), j["probs"].get(k), k == j["choice"], k == c["target"])
                          for k in keys], c["thr"] if c["target"] else None, color)
        self.banner.set(text, color)
        meta = []
        if c["need"] > 1:
            meta.append(f"连续 {min(j['streak'], c['need'])}/{c['need']}")
        if j["ms"]:
            meta.append(f"{float(j['ms']):.0f} ms")
        self.jev_meta.setText(" · ".join(meta))

    # ---------------- 主程序 ----------------
    def set_running(self, running):
        self.running = running
        self.btn_proc.setText("■ 停止" if running else "▶ 启动")
        self.btn_proc.setObjectName("danger" if running else "primary")
        self.btn_proc.style().unpolish(self.btn_proc)
        self.btn_proc.style().polish(self.btn_proc)
        self.dot.setStyleSheet(f"color:{OK if running else DIM};font-size:14px;")
        for b in list(self.key_btns.values()):
            b.setEnabled(running)
        self.btn_send.setEnabled(True)

    def toggle_program(self):
        if self.replay:
            return
        if self.proc.state() == QProcess.NotRunning:
            self.output.clear()
            if self.record_enabled:
                self.start_recording()
            self.proc.start("bash", ["-lc", self.cfg["main_cmd"]])
        else:
            self.send_line("q")
            QTimer.singleShot(3000, lambda: self.proc.state() != QProcess.NotRunning and self.proc.terminate())
            QTimer.singleShot(5000, lambda: self.proc.state() != QProcess.NotRunning and self.proc.kill())

    def send_line(self, text):
        if self.replay:
            return
        clean = text.strip()
        if clean.lower() == "r":
            self.reset_view()
            self.set_fsm("idle")
            if getattr(self, "record_enabled", True):
                self.stop_recording(wait=False)
                self.start_recording()
        if hasattr(self, "proc") and self.proc.state() == QProcess.Running:
            self.proc.write((text + "\n").encode())
        self.hub.event("input", text)

    def keyPressEvent(self, e):
        if not self.input.hasFocus():
            txt = e.text()
            if txt in ("-", "—", "－"):
                self.send_line("-")
                e.accept()
                return
            elif txt.lower() == "r":
                self.send_line("r")
                e.accept()
                return
        super().keyPressEvent(e)

    def send_input(self):
        t = self.input.text().strip()
        self.input.clear()
        self.send_line(t)

    def on_typing(self):
        if not self.replay:
            self.hub.event("typing", self.input.text())

    def on_output(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        self._buf += data.replace("\r\n", "\n").replace("\r", "\n")
        *lines, self._buf = self._buf.split("\n")
        for ln in lines:
            self.hub.event("log", ln)
        if self._buf:
            self._flush.start(200)

    def flush_partial(self):
        if self._buf:
            self.hub.event("log", self._buf)
            self._buf = ""

    def on_finished(self, code, status):
        self.flush_partial()
        self.hub.event("proc", {"running": False, "code": code})
        self.stop_recording()

    def _capture_screen_frame(self):
        if self.screen_recorder is None or not self.screen_recorder.running:
            return
        pix = self.centralWidget().grab()
        qimg = pix.toImage().convertToFormat(QImage.Format_BGR888)
        w, h = qimg.width(), qimg.height()
        stride = qimg.bytesPerLine()
        ptr = qimg.constBits()
        if stride == w * 3:
            arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3)).copy()
        else:
            raw = np.frombuffer(ptr, np.uint8).reshape((h, stride))
            arr = raw[:, : w * 3].reshape((h, w, 3)).copy()
        self.screen_recorder.push_frame(arr)

    # ---------------- 录制 ----------------
    def start_recording(self):
        if self.replay or self.hub.recorder:
            return
        meta = dict(cameras=[n for n, _ in self.cfg["cameras"]], jev_views=list(self.cfg["jev_views"]),
                    robot_ip=self.cfg["robot_ip"], main_cmd=self.cfg["main_cmd"],
                    window=[self.width(), self.height()])
        rec = Recorder(self.cfg["record_dir"], meta, len(self.cfg["cameras"]), self.cfg["record_fps"])
        self.hub.recorder = rec
        self.rec_label.show()
        print(f"[viewer] 原始数据录制中 → {rec.dir}", flush=True)

        # 自动开启当前窗口的 MP4 全画面视频录屏 (锁定偶数基准分辨率)
        window_mp4 = os.path.join(rec.dir, "window_recording.mp4")
        cw = self.centralWidget()
        w = (cw.width() if cw else self.width()) & ~1
        h = (cw.height() if cw else self.height()) & ~1
        self.screen_recorder = AutoScreenRecorder(window_mp4, fps=float(UI_FPS), target_size=(w, h))
        self.screen_recorder.start()
        self._screen_timer.start(int(1000 / UI_FPS))
        print(f"[viewer] 窗口全画面录像已开启 → {window_mp4} ({w}x{h})", flush=True)

    def stop_recording(self, wait=False):
        if self._screen_timer.isActive():
            self._screen_timer.stop()
        if self.screen_recorder is not None:
            sr = self.screen_recorder
            self.screen_recorder = None
            sr.stop()
            sr.join(timeout=5)  # 必须等待编码线程把 moov atom 完整刷入 MP4 文件尾部
            print(f"[viewer] 窗口全画面录像已保存 → {sr.output_path} ({sr.total_frames} 帧)", flush=True)

        rec, self.hub.recorder = self.hub.recorder, None
        if not rec:
            return
        self.rec_label.hide()
        if wait:
            rec.close()
        else:
            th = threading.Thread(target=rec.close, daemon=True)
            th.start()
            self._rec_threads.append(th)
        print(f"[viewer] 原始数据录制已保存 → {rec.dir}", flush=True)

    def toggle_recording(self):
        if self.hub.recorder:
            self.stop_recording()
        elif not self.replay:
            self.start_recording()
            self.hub.event("proc", {"running": self.running})

    def closeEvent(self, e):
        if not self.replay:
            if self.proc.state() != QProcess.NotRunning:
                self.proc.write(b"q\n")
                if not self.proc.waitForFinished(3000):
                    self.proc.kill()
                    self.proc.waitForFinished(1000)
            self.stop_recording(wait=True)
            for th in self._rec_threads:
                th.join(timeout=20)
            for t in self.threads:
                t.running = False
        super().closeEvent(e)


# ============================================================
# 回放控制条（独立窗口，不出现在录屏区域）
# ============================================================
class ReplayBar(QWidget):
    def __init__(self, player: Player, main: MainWindow):
        super().__init__()
        self.player, self.main = player, main
        self.setWindowTitle("回放控制")
        self.setStyleSheet(f"QWidget{{background:{CARD};color:{TEXT};font-size:14px;}}")
        lay = QHBoxLayout(self)
        self.btn = QPushButton("▶")
        self.btn.setFixedWidth(44)
        self.btn.clicked.connect(self.toggle)
        lay.addWidget(self.btn)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, int(player.s.duration * 10))
        self.slider.sliderReleased.connect(lambda: self.seek(self.slider.value() / 10))
        self.slider.actionTriggered.connect(self._on_action)
        lay.addWidget(self.slider, 1)
        self.time = QLabel("")
        self.time.setFont(font(13, mono=True))
        lay.addWidget(self.time)
        self.speed = QComboBox()
        self.speed.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.speed.setCurrentText("1x")
        self.speed.currentTextChanged.connect(lambda s: setattr(player, "speed", float(s[:-1])))
        lay.addWidget(self.speed)
        self.resize(900, 60)
        for w in (self, main):
            QShortcut(QKeySequence(Qt.Key_Space), w, activated=self.toggle)
            QShortcut(QKeySequence(Qt.Key_Left), w, activated=lambda: self.seek(player.t - 5))
            QShortcut(QKeySequence(Qt.Key_Right), w, activated=lambda: self.seek(player.t + 5))
            QShortcut(QKeySequence(Qt.Key_Home), w, activated=lambda: self.seek(0))
        t = QTimer(self)
        t.timeout.connect(self.sync)
        t.start(100)

    def _on_action(self, action):
        if int(action) not in (0, 7):   # 0=SliderNoAction, 7=SliderMove（拖动结束时再跳转）
            QTimer.singleShot(0, lambda: self.seek(self.slider.value() / 10))

    def toggle(self):
        self.player.toggle()

    def seek(self, t):
        self.main.reset_view()
        self.player.seek(t)
        self.main.refresh()

    @staticmethod
    def fmt(t):
        return f"{int(t // 60):02d}:{t % 60:04.1f}"

    def sync(self):
        p = self.player
        self.btn.setText("❚❚" if p.playing else "▶")
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(int(p.t * 10))
            self.slider.blockSignals(False)
            self.time.setText(f"{self.fmt(p.t)} / {self.fmt(p.s.duration)}")
        else:
            self.time.setText(f"{self.fmt(self.slider.value() / 10)} / {self.fmt(p.s.duration)}")


# ============================================================
# 导出视频
# ============================================================
def export_video(app, win, player, out, fps, start, end):
    W, H = win.width(), win.height()
    ff = shutil.which("ffmpeg")
    if ff:
        enc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", out],
                               stdin=subprocess.PIPE)
        write, close = enc.stdin.write, lambda: (enc.stdin.close(), enc.wait())
    else:
        import cv2
        import numpy as np
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        write = lambda b: vw.write(cv2.cvtColor(np.frombuffer(b, np.uint8).reshape(H, W, 3), cv2.COLOR_RGB2BGR))
        close = vw.release
    end = min(end if end is not None else player.s.duration, player.s.duration)
    win.reset_view()
    player.seek(start)
    win.refresh()
    n = max(1, int((end - start) * fps))
    t0 = time.time()
    for k in range(n):
        player.set_time(start + k / fps)
        win.refresh()
        app.processEvents()
        img = win.grab().toImage().convertToFormat(QImage.Format_RGB888)
        if img.width() != W or img.height() != H:
            img = img.scaled(W, H)
        raw = bytes(img.constBits())
        bpl = img.bytesPerLine()
        if bpl != W * 3:
            raw = b"".join(raw[r * bpl: r * bpl + W * 3] for r in range(H))
        write(raw[: W * H * 3])
        if k % max(1, n // 20) == 0:
            print(f"\r导出 {k * 100 // n:3d}%  ({k}/{n} 帧, {time.time() - t0:.0f}s)", end="", flush=True)
    close()
    print(f"\r导出完成 → {out}  ({n} 帧, {n / fps:.1f}s 视频, 用时 {time.time() - t0:.0f}s)")


STYLE = f"""
QWidget {{ background:transparent; color:{TEXT}; font-size:14px; }}
QMainWindow, #root {{ background:{BG}; }}
#card {{ background:{CARD}; border:1px solid {LINE}; border-radius:10px; }}
#cardTitle {{ color:{MUTED}; font-size:13px; font-weight:600; letter-spacing:1px; }}
#k {{ color:{MUTED}; font-size:13px; }}
#val {{ font-family:"DejaVu Sans Mono"; font-size:15px; font-weight:600; min-width:56px; }}
#question {{ font-size:24px; font-weight:700; }}
#taskKey {{ color:{MUTED}; font-size:18px; padding-top:3px; }}
#prompt {{ font-size:24px; font-weight:600; color:{TEXT}; }}
#input {{ background:#141518; border:1px solid {LINE}; border-radius:8px; padding:8px 14px; font-weight:700; }}
#input:focus {{ border-color:{ACCENT}; }}
#console {{ background:#121316; border:none; border-radius:6px; padding:4px; color:#c9ccd1; font-size:9px; }}
QPushButton {{ background:#2a2d33; border:1px solid #363a41; border-radius:7px; padding:7px 16px; font-size:14px; }}
QPushButton:hover {{ background:#33373e; }}
QPushButton:pressed {{ background:{ACCENT}; border-color:{ACCENT}; }}
QPushButton:disabled {{ color:{DIM}; background:#202226; border-color:#2a2d33; }}
QPushButton#primary {{ background:{ACCENT}; border-color:{ACCENT}; color:white; font-weight:600; }}
QPushButton#primary:disabled {{ background:#253350; border-color:#253350; color:#6c7a94; }}
QPushButton#danger {{ background:#7a2b2b; border-color:#933; color:white; font-weight:600; }}
QPushButton[big="true"] {{ font-size:18px; padding:11px 16px; }}
QScrollBar:vertical {{ width:8px; background:transparent; }}
QScrollBar::handle:vertical {{ background:#3a3e45; border-radius:4px; min-height:30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height:0; }}
QScrollBar:horizontal {{ height:8px; background:transparent; }}
QScrollBar::handle:horizontal {{ background:#3a3e45; border-radius:4px; }}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", "--robot_ip", dest="robot_ip", default=CONFIG["robot_ip"], help="机器人机载 IP 地址")
    ap.add_argument("--cmd", default=None, help="主程序启动命令（覆盖 CONFIG）")
    ap.add_argument("--jev-viz", "--jev_viz", dest="jev_viz", default=CONFIG["jev_viz"], help="JEV 判定广播地址，空字符串表示不订阅")
    ap.add_argument("--mock", action="store_true", help="连接本地 mock_robot.py，以 mock_main.py 作为主程序")
    ap.add_argument("--no-record", "--no_record", dest="no_record", action="store_true", help="启动主程序时不自动录制")
    ap.add_argument("--record-dir", "--record_dir", dest="record_dir", default=CONFIG["record_dir"])
    ap.add_argument("--replay", default=None, help="回放录制目录")
    ap.add_argument("--export", default=None, help="与 --replay 一起使用：导出 mp4")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default=None, help="窗口尺寸，如 1200x900 或 1440x1080 (默认自适应当前屏幕)")
    ap.add_argument("--head-port", "--head_port", dest="head_port", type=int, default=None, help="头部相机推流端口 (覆盖 CONFIG，默认 5556)")
    ap.add_argument("--frameless", action="store_true", help="无边框固定尺寸（录屏用）")
    ap.add_argument("--repair-mp4", "--repair_mp4", dest="repair_mp4", default=None,
                    help="修复因异常中断或强退缺少 moov atom 的 window_recording.mp4 视频文件")
    args = ap.parse_args()

    if args.repair_mp4:
        out = repair_unclosed_mp4(args.repair_mp4)
        print(f"✓ 录屏视频修复成功，已转码输出为标准 H.264 视频: {out}")
        return

    cfg = dict(CONFIG, robot_ip=args.robot_ip, jev_viz=args.jev_viz, record_dir=args.record_dir)
    if args.head_port:
        cfg["cameras"] = [(n, args.head_port if n == "头部" else p) for n, p in cfg["cameras"]]
    if args.robot_ip != CONFIG["robot_ip"] and not args.cmd:
        cfg["main_cmd"] = cfg["main_cmd"].replace(f"--robot_ip={CONFIG['robot_ip']}", f"--robot_ip={args.robot_ip}")
    if args.mock:
        cfg["robot_ip"] = "127.0.0.1"
        cfg["main_cmd"] = f"python3 -u {os.path.dirname(os.path.abspath(__file__))}/mock_main.py --viz"
    if args.cmd:
        cfg["main_cmd"] = args.cmd

    if args.export and "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication(sys.argv)
    app.setFont(font(14))
    app.setStyleSheet(STYLE)

    if args.size:
        W, H = (int(x) for x in args.size.lower().split("x"))
    else:
        screen = app.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        if avail and not (args.frameless or args.export):
            # 屏幕工作区最大高度 (预留系统顶栏、dock和窗口标题栏高度 ~75px)
            max_h = max(660, avail.height() - 75)
            max_w = max(880, avail.width() - 40)
            if max_h >= 1100:
                H = 1080
            elif max_h >= 920:
                H = 900
            else:
                H = int(max_h // 10 * 10)
            W = int(H * 4 / 3)
            if W > max_w:
                W = max_w
                H = int(W * 3 / 4)
        else:
            W, H = 1440, 1080

    player = None
    if args.replay:
        session = Session(args.replay)
        meta = session.meta
        if meta.get("jev_views"):
            cfg["jev_views"] = tuple(meta["jev_views"])
        hub = Hub(len(session.cams))
        player = Player(session, hub)
    else:
        hub = Hub(len(cfg["cameras"]))
    win = MainWindow(cfg, hub, player, record=not args.no_record)

    if args.frameless or args.export:
        if args.frameless:
            win.setWindowFlag(Qt.FramelessWindowHint)
            win.move(0, 0)
        win.setFixedSize(W, H)
    else:
        win.resize(W, H)
        screen = app.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            win.move(avail.x() + max(0, (avail.width() - W) // 2),
                     avail.y() + max(0, (avail.height() - H) // 2))
    win.show()

    if args.export:
        if not player:
            sys.exit("--export 需要和 --replay 一起使用")
        export_video(app, win, player, args.export, args.fps, args.start, args.end)
        sys.stdout.flush()
        os._exit(0)          # 跳过 PySide 退出时的析构（偶发段错误），视频已写完

    if player:
        player.seek(args.start)
        win.refresh()
        bar = ReplayBar(player, win)
        bar.show()
        bar.move(win.x(), win.y() + win.frameGeometry().height() + 8)
    win.start()
    code = app.exec()        # 录制已在 closeEvent 中同步写完
    sys.stdout.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
