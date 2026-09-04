#!/usr/bin/env python
import time
import sys
from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config
from lerobot.teleoperators.unitree_g1 import UnitreeG1Teleoperator, UnitreeG1TeleoperatorConfig

def main():
    print("\n" + "="*75)
    print(" Unitree G1 遥控数据流实时链路诊断工具")
    print(" 启动后请推动手柄左摇杆（前后），查看第 1 到第 5 阶段哪个环节没有变化")
    print("="*75 + "\n")

    robot_cfg = UnitreeG1Config(
        is_simulation=True,
        locomotion_mode="walk",
        cameras={},
        controller="GrootLocomotionController"
    )
    teleop_cfg = UnitreeG1TeleoperatorConfig()

    robot = UnitreeG1(robot_cfg)
    teleop = UnitreeG1Teleoperator(teleop_cfg)

    teleop.connect()
    robot.connect()

    sim_bridge = robot.sim_env.simulator.unitree_bridge
    js = sim_bridge.joystick

    print(f"[手柄检测] 识别名称: {js.get_name() if js else '未检测到手柄'}")
    print("开始监听数据链路（按 Ctrl+C 退出）...\n")
    print(f"{'1.硬件轴(LY)':<14} | {'2.仿真桥(LY)':<14} | {'3.遥控动作(LY)':<16} | {'4.运控指令(CMD)':<22} | {'5.运控输出'}")
    print("-" * 80)

    try:
        while True:
            obs = robot.get_observation()
            teleop.send_feedback(obs)
            action = teleop.get_action()
            robot.send_action(action)

            # 1. 直接读取硬件轴
            hw_ly = round(js.get_axis(1), 2) if js else 0.0
            # 2. 仿真桥接层
            bridge_ly = round(sim_bridge.wireless_controller.ly, 2)
            # 3. Teleoperator 输出动作中的 remote.ly
            teleop_ly = round(action.get("remote.ly", 0.0), 2)
            # 4. GR00T 控制器实际接收到的速度指令 [vx, vy, yaw_rate]
            ctrl_cmd = [round(float(x), 2) for x in robot.controller.cmd] if robot.controller else []
            # 5. 运控输出的关节数量
            ctrl_out = len(robot.controller_output)

            print(f"\r{hw_ly:+14.2f} | {bridge_ly:+14.2f} | {teleop_ly:+16.2f} | {str(ctrl_cmd):<22} | {ctrl_out} 个关节指令", end="", flush=True)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n诊断结束。")
    finally:
        robot.disconnect()
        teleop.disconnect()

if __name__ == "__main__":
    main()
