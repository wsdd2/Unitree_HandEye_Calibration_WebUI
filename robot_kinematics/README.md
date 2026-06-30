# Robot Kinematics

本目录按机器人运动学的两个方向拆分：

- `joint_to_pose`：正运动学 FK，从 URDF 关节值计算 link 位姿。
- `pose_to_joint`：逆运动学 IK，从目标 link 位姿反求关节值。

两个方向都采用通用 URDF 思路实现，不绑定某一个具体 G1 文件。只要机器人
URDF 符合标准 `link` / `joint` 定义，就可以用于 Unitree G1、H 系列或其它
机器人模型。

当前实现定位：

- FK 是确定性计算，适合实时查询相机、手腕、手掌等 link 的坐标和四元数。
- IK 是数值迭代基础版，适合离线验证和单目标 link 求解；实机操控前仍需加入
  轨迹平滑、速度限制、碰撞检测和稳定性约束。

## Unitree SDK2 实机状态接入

`unitree_sdk2_bridge.py` 用于在 WSL2 Ubuntu 22.04 中通过
`unitree_sdk2_python` 订阅宇树机器人 `rt/lowstate`，把电机索引转换成 URDF
关节名，然后直接传给本目录的 FK/IK 计算。

安装 SDK 后，在 WSL2 里先确认机器人 WiFi 对应的网卡名：

```bash
ip addr
```

只查看当前实机关节状态：

```bash
python ./robot_kinematics/unitree_sdk2_bridge.py \
  --mode state \
  --network-interface eth0 \
  --pretty
```

把实机关节状态传入 FK，计算目标 link 位姿：

```bash
python ./robot_kinematics/unitree_sdk2_bridge.py \
  --mode fk \
  --network-interface eth0 \
  --urdf ./unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf \
  --base-link pelvis \
  --target d435_link,left_wrist_yaw_link,right_wrist_yaw_link \
  --pretty
```

把实机关节状态作为 IK 初值，求解左腕目标位姿：

```bash
python ./robot_kinematics/unitree_sdk2_bridge.py \
  --mode ik \
  --network-interface eth0 \
  --urdf ./unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf \
  --base-link pelvis \
  --target-link left_wrist_yaw_link \
  --target-xyz 0.20 0.18 0.05 \
  --target-rpy 0.0 0.0 0.0 \
  --active-joint waist_yaw_joint,waist_roll_joint,waist_pitch_joint,left_shoulder_pitch_joint,left_shoulder_roll_joint,left_shoulder_yaw_joint,left_elbow_joint,left_wrist_roll_joint,left_wrist_pitch_joint,left_wrist_yaw_joint \
  --pretty
```

低层位置命令发布默认关闭。只有确认机器人处于安全测试条件、低层控制模式正确、
关节映射和目标值都已核对后，才使用 `--publish-solution` 和
`--confirm-low-level-control`。
