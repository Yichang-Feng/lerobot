#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unitree G1 独立电机 DDS-ZMQ 桥接服务 (无需在机器人端克隆 LeRobot 完整仓库)
功能：
1. 接收上位机 ZMQ PULL (端口 6000) 的动作指令 -> 转发给 G1 底层 DDS (rt/lowcmd)
2. 读取 G1 底层 DDS 状态 (rt/lowstate) -> 通过 ZMQ PUB (端口 6001) 广播给上位机
"""
from __future__ import annotations
import base64
import contextlib
import json
import threading
import time
from typing import Any

import zmq
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.utils.crc import CRC

kTopicLowCommand_Debug = "rt/lowcmd"
kTopicLowState = "rt/lowstate"
LOWCMD_PORT = 6000
LOWSTATE_PORT = 6001
NUM_MOTORS = 35


def lowstate_to_dict(msg: hg_LowState) -> dict[str, Any]:
    """将 LowState SDK 消息序列化为 JSON 兼容字典"""
    motor_states = []
    for i in range(NUM_MOTORS):
        temp = msg.motor_state[i].temperature
        avg_temp = float(sum(temp) / len(temp)) if isinstance(temp, list) else float(temp)
        motor_states.append({
            "q": float(msg.motor_state[i].q),
            "dq": float(msg.motor_state[i].dq),
            "tau_est": float(msg.motor_state[i].tau_est),
            "temperature": avg_temp,
        })
    return {
        "motor_state": motor_states,
        "imu_state": {
            "quaternion": [float(x) for x in msg.imu_state.quaternion],
            "gyroscope": [float(x) for x in msg.imu_state.gyroscope],
            "accelerometer": [float(x) for x in msg.imu_state.accelerometer],
            "rpy": [float(x) for x in msg.imu_state.rpy],
            "temperature": float(msg.imu_state.temperature),
        },
        "wireless_remote": base64.b64encode(bytes(msg.wireless_remote)).decode("ascii"),
        "mode_machine": int(msg.mode_machine),
    }


def dict_to_lowcmd(data: dict[str, Any]) -> hg_LowCmd:
    """将字典反序列化重构为 LowCmd SDK 消息对象"""
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = data.get("mode_pr", 0)
    cmd.mode_machine = data.get("mode_machine", 0)
    for i, motor_data in enumerate(data.get("motor_cmd", [])):
        cmd.motor_cmd[i].mode = motor_data.get("mode", 0)
        cmd.motor_cmd[i].q = motor_data.get("q", 0.0)
        cmd.motor_cmd[i].dq = motor_data.get("dq", 0.0)
        cmd.motor_cmd[i].kp = motor_data.get("kp", 0.0)
        cmd.motor_cmd[i].kd = motor_data.get("kd", 0.0)
        cmd.motor_cmd[i].tau = motor_data.get("tau", 0.0)
    return cmd


def state_forward_loop(lowstate_sub: ChannelSubscriber, lowstate_sock: zmq.Socket, state_period: float, shutdown_event: threading.Event):
    """读取 DDS 状态并通过 ZMQ 广播到 6001 端口"""
    last_state_time = 0.0
    while not shutdown_event.is_set():
        msg = lowstate_sub.Read()
        if msg is None:
            continue
        now = time.time()
        if now - last_state_time >= state_period:
            state_dict = lowstate_to_dict(msg)
            payload = json.dumps({"topic": kTopicLowState, "data": state_dict}).encode("utf-8")
            with contextlib.suppress(zmq.Again):
                lowstate_sock.send(payload, zmq.NOBLOCK)
            last_state_time = now


def cmd_forward_loop(lowcmd_sock: zmq.Socket, lowcmd_pub_debug: ChannelPublisher, crc: CRC):
    """从 6000 端口接收 ZMQ 指令并通过 DDS 下发给电机"""
    while True:
        try:
            payload = lowcmd_sock.recv()
        except zmq.ContextTerminated:
            break
        msg_dict = json.loads(payload.decode("utf-8"))
        topic = msg_dict.get("topic", "")
        cmd_data = msg_dict.get("data", {})
        cmd = dict_to_lowcmd(cmd_data)
        cmd.crc = crc.Crc(cmd)
        if topic == kTopicLowCommand_Debug:
            lowcmd_pub_debug.Write(cmd)


def main():
    print("[*] 正在初始化 G1 电机底层 DDS 通信...")
    ChannelFactoryInitialize(0)

    # 释放机载默认运动控制器的占用，以便接管控制
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    status, result = msc.CheckMode()
    while result is not None and "name" in result and result["name"]:
        print(f"[*] 正在释放机载服务: {result.get('name')}")
        msc.ReleaseMode()
        status, result = msc.CheckMode()
        time.sleep(1.0)

    crc = CRC()
    lowcmd_pub_debug = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
    lowcmd_pub_debug.Init()
    lowstate_sub = ChannelSubscriber(kTopicLowState, hg_LowState)
    lowstate_sub.Init()

    ctx = zmq.Context.instance()
    lowcmd_sock = ctx.socket(zmq.PULL)
    lowcmd_sock.bind(f"tcp://0.0.0.0:{LOWCMD_PORT}")

    lowstate_sock = ctx.socket(zmq.PUB)
    lowstate_sock.bind(f"tcp://0.0.0.0:{LOWSTATE_PORT}")

    shutdown_event = threading.Event()
    t_state = threading.Thread(
        target=state_forward_loop,
        args=(lowstate_sub, lowstate_sock, 0.002, shutdown_event),
        daemon=True
    )
    t_state.start()

    print(f"[+] 电机桥接服务已启动！")
    print(f"    - 监听上位机指令端口: {LOWCMD_PORT} (ZMQ PULL)")
    print(f"    - 广播电机状态端口: {LOWSTATE_PORT} (ZMQ PUB)")

    try:
        cmd_forward_loop(lowcmd_sock, lowcmd_pub_debug, crc)
    except KeyboardInterrupt:
        print("\n[*] 正在停止电机桥接服务...")
    finally:
        shutdown_event.set()
        ctx.term()


if __name__ == "__main__":
    main()
