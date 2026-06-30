# Pose To Joint：笛卡尔位姿转关节参数

本目录实现的是逆运动学 IK：给定一个目标 link 的笛卡尔位姿，反求一组关节值，使该 link 尽量到达目标位置和姿态。

当前实现文件是 `ik_urdf.py`。它复用了 `../joint_to_pose/fk_urdf.py` 中的纯 Python URDF 正运动学，不依赖 ROS、Pinocchio 或 numpy。算法是通用数值 IK，适合先做验证、离线计算和理解流程。

## 与正解的关系

正解 FK 是：

```text
joint_values + URDF -> target_link pose
```

逆解 IK 是：

```text
target_link pose + URDF -> joint_values
```

当前 IK 的内部流程是：

```text
初始关节 q
  -> 调用 FK 得到当前末端位姿
  -> 计算当前位姿和目标位姿的误差
  -> 用数值微分估计雅可比 J
  -> 用阻尼最小二乘求 delta_q
  -> 更新 q 并重复迭代
```

## 算法说明

当前实现使用单目标 link 的阻尼最小二乘法：

```text
delta_q = J^T * inv(J * J^T + lambda^2 * I) * error
```

其中：

- `q`：待求的关节向量。
- `error`：6 维位姿误差，前三维是位置误差，后三维是姿态误差。
- `J`：目标 link 对开放关节的 6xN 雅可比矩阵。
- `lambda`：阻尼系数，对应命令行参数 `--damping`。

姿态误差使用旋转矩阵的 rotation vector 表示，输入目标姿态可以是四元数 `x y z w`，也可以是 `roll pitch yaw`。

## 输入参数

### `--urdf`

必填。URDF 文件路径。

G1 29DoF 推荐优先使用：

```text
robots/g1/g1_29dof_rev_1_0.urdf
```

### `--target-link`

必填。要反求的目标 link 名称。

示例：

```powershell
--target-link left_wrist_yaw_link
--target-link d435_link
--target-link right_hand_palm_link
```

注意：IK 只能改变开放关节来移动目标 link。如果目标 link 是相机 `d435_link`，而相机通过 fixed joint 固定在 `torso_link` 上，则实际可动关节通常只有腰部或更上游关节。

### `--target-xyz`

可选，和 `--target-pose` 二选一。目标位置，单位米。

示例：

```powershell
--target-xyz 0.20 0.15 0.10
```

### `--target-quat-xyzw`

可选。目标姿态四元数，顺序为 `x y z w`。

示例：

```powershell
--target-quat-xyzw 0.0 0.0 0.0 1.0
```

如果没有提供 `--target-quat-xyzw`，可以用 `--target-rpy`。

### `--target-rpy`

可选。目标姿态 roll、pitch、yaw，单位弧度。

示例：

```powershell
--target-rpy 0.0 0.0 0.0
```

如果 `--target-quat-xyzw` 和 `--target-rpy` 都没传，脚本默认目标姿态为零姿态。

### `--target-pose`

可选。用 JSON 文件输入目标位姿。

推荐格式：

```json
{
  "position_xyz": [0.20, 0.15, 0.10],
  "orientation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]
}
```

也支持 RPY：

```json
{
  "position_xyz": [0.20, 0.15, 0.10],
  "orientation_rpy": [0.0, 0.0, 0.0]
}
```

### `--initial-joints`

可选。IK 初始关节值 JSON 文件。格式和正解脚本一致：

```json
{
  "joints": {
    "left_shoulder_pitch_joint": 0.1,
    "left_elbow_joint": 0.3
  }
}
```

IK 是迭代法，初值会影响是否收敛、收敛到哪组解、以及是否落在关节限位内。实际机器人操控时，通常应使用当前实机关节状态作为初值。

### `--joint`

可选。命令行临时提供或覆盖初始关节值。

示例：

```powershell
--joint left_shoulder_pitch_joint=0.1 `
--joint left_elbow_joint=0.3
```

### `--active-joint`

可选。指定哪些关节允许 IK 改变。可以重复传入，也可以逗号分隔。

示例：

```powershell
--active-joint waist_yaw_joint `
--active-joint left_shoulder_pitch_joint,left_shoulder_roll_joint,left_elbow_joint
```

如果不传，脚本会自动寻找从 `base-link` 到 `target-link` 链路上的 active joints。对 `left_wrist_yaw_link` 来说，通常会包含腰部和左臂关节。

实际使用建议显式指定开放关节，尤其是类人机器人：

- 单臂末端 IK：通常开放目标侧手臂，必要时加腰部。
- 相机姿态 IK：通常只开放腰部或躯干相关关节。
- 全身 IK：当前脚本不是完整全身约束 IK，不建议直接开放所有腿部关节去做实机操控。

### `--base-link`

可选。指定输出坐标系挂在哪个 link 上。不填时默认使用 URDF 根 link，G1 通常是 `pelvis`。

### `--base-xyz` / `--base-rpy` / `--base-quat-xyzw`

可选。与正解脚本一致，用于提供 `world -> base-link` 的位姿。

如果目标位姿是世界系下的 `world -> target_link`，就必须传入当前 `world -> pelvis`，否则 IK 会把目标当成 `pelvis` 坐标系下的目标。

### 迭代与收敛参数

- `--max-iterations`：最大迭代次数，默认 `200`。
- `--tolerance-position`：位置误差阈值，单位米，默认 `1e-4`。
- `--tolerance-orientation`：姿态误差阈值，单位弧度，默认 `1e-3`。
- `--damping`：阻尼系数，默认 `1e-3`。更大更稳但可能更慢。
- `--step-scale`：每次更新关节的步长比例，默认 `0.5`。
- `--finite-difference-step`：数值雅可比微分步长，默认 `1e-6`。
- `--position-weight`：位置误差权重，默认 `1.0`。
- `--orientation-weight`：姿态误差权重，默认 `0.5`。
- `--ignore-limits`：默认 IK 会按 URDF 关节限位裁剪结果；加这个参数后不裁剪。
- `--pretty`：格式化 JSON 输出。

## 输出字段

典型输出：

```json
{
  "urdf": "robots/g1/g1_29dof_rev_1_0.urdf",
  "root_links": ["pelvis"],
  "base_link": "pelvis",
  "success": true,
  "message": "Converged",
  "iterations": 32,
  "target_link": "left_wrist_yaw_link",
  "active_joints": [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint"
  ],
  "joint_values": {
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_elbow_joint": 0.6
  },
  "final_position_error_norm": 0.00001,
  "final_orientation_error_norm": 0.0001,
  "final_error_norm": 0.00005
}
```

字段解释：

- `success`：是否达到收敛阈值。
- `message`：收敛或失败原因。
- `iterations`：实际迭代次数。
- `target_link`：本次求解的目标 link。
- `active_joints`：本次允许 IK 改变的关节。
- `joint_values`：求出的关节值，只包含 active joints。
- `final_position_error_norm`：最终位置误差范数，单位米。
- `final_orientation_error_norm`：最终姿态误差范数，单位弧度。
- `final_error_norm`：加权后的总误差范数。

如果 `success=false`，仍会输出最后一次迭代得到的 `joint_values`，但不能直接认为已经到达目标。

## Example 1：运行 G1 左腕 IK 示例

# 从本仓库根目录运行示例：

```powershell
python .\robot_kinematics\pose_to_joint\example_g1_left_arm_ik.py
```

这个示例会：

1. 使用一组已知左臂关节值。
2. 调用 FK 生成 `left_wrist_yaw_link` 的目标位姿。
3. 从零初值开始调用 IK 反求关节值。
4. 输出 IK 是否收敛和最终误差。

这是最稳妥的验证方式，因为目标位姿一定来自同一个 URDF，理论上可达。

## Example 2：直接给目标 xyz 和四元数

```powershell
python .\robot_kinematics\pose_to_joint\ik_urdf.py `
  --urdf .\unitree_ros\robots\g1_description\g1_29dof_rev_1_0.urdf `
  --target-link left_wrist_yaw_link `
  --target-xyz 0.20 0.15 0.10 `
  --target-quat-xyzw 0.0 0.0 0.0 1.0 `
  --active-joint waist_yaw_joint `
  --active-joint waist_roll_joint `
  --active-joint waist_pitch_joint `
  --active-joint left_shoulder_pitch_joint `
  --active-joint left_shoulder_roll_joint `
  --active-joint left_shoulder_yaw_joint `
  --active-joint left_elbow_joint `
  --active-joint left_wrist_roll_joint `
  --active-joint left_wrist_pitch_joint `
  --active-joint left_wrist_yaw_joint `
  --pretty
```

如果目标不可达，或者初值太差，可能输出 `success=false`。

## Example 3：使用目标位姿 JSON

创建 `target_pose.json`：

```json
{
  "position_xyz": [0.20, 0.15, 0.10],
  "orientation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]
}
```

运行：

```powershell
python .\robot_kinematics\pose_to_joint\ik_urdf.py `
  --urdf .\unitree_ros\robots\g1_description\g1_29dof_rev_1_0.urdf `
  --target-link left_wrist_yaw_link `
  --target-pose .\target_pose.json `
  --pretty
```

这里没有显式指定 `--active-joint`，脚本会自动使用 `pelvis -> left_wrist_yaw_link` 链上的 active joints。

## Example 4：使用当前关节状态作为初值

创建 `current_joints.json`：

```json
{
  "joints": {
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.1,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.3,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0
  }
}
```

运行：

```powershell
python .\robot_kinematics\pose_to_joint\ik_urdf.py `
  --urdf .\unitree_ros\robots\g1_description\g1_29dof_rev_1_0.urdf `
  --target-link left_wrist_yaw_link `
  --target-pose .\target_pose.json `
  --initial-joints .\current_joints.json `
  --pretty
```

实机控制时，应优先使用当前机器人关节状态作为初值，而不是全零初值。

## 重要限制

- 当前实现是单目标 link IK，不是完整全身 QP IK。
- 没有碰撞检测、重心约束、足底接触约束和动力学约束。
- 类人机器人存在冗余自由度，同一个目标位姿可能有多组关节解。
- 逆解结果依赖初值、开放关节集合、关节限位和阻尼参数。
- 对实机下发前，必须做关节限位、速度限位、轨迹平滑和安全检查。
- 如果目标位姿在世界系下，必须提供正确的 `world -> pelvis`，否则坐标系会错。
