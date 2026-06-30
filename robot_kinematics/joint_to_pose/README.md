# Joint To Pose：关节参数转笛卡尔位姿

本目录实现的是正运动学 FK：给定一个 URDF 文件和一组关节值，计算指定 link 在某个基准坐标系下的笛卡尔位姿。

当前实现文件是 `fk_urdf.py`，它是纯 Python 标准库实现，不依赖 ROS、Pinocchio 或 numpy。只要 URDF 中的 `link`、`joint`、`origin`、`axis` 写法符合标准，就可以用于 Unitree G1、H 系列或其它机器人。

## 计算逻辑

脚本会从 URDF 解析出机器人运动学树：

- `link`：刚体节点，例如 `pelvis`、`torso_link`、`d435_link`、`left_wrist_yaw_link`。
- `joint`：连接父 link 和子 link 的关节。
- `origin xyz/rpy`：关节相对父 link 的固定安装位姿。
- `axis`：转动或平移轴。
- `limit`：关节上下限，使用 `--clamp-to-limits` 时会用到。

对每个关节，脚本按下面的形式连乘变换：

```text
T_parent_child(q) = T_origin_xyz_rpy * T_joint_motion(q)
```

最终得到：

```text
T_base_target = T_base_link1 * T_link1_link2 * ... * T_linkN_target
```

如果额外传入了机器人根节点在世界系下的位姿，则会输出：

```text
T_world_target = T_world_base * T_base_target
```

## 坐标系说明

对 Unitree G1 的 `g1_29dof_rev_1_0.urdf` 来说，URDF 根节点通常是 `pelvis`。因此：

- 不传 `--base-xyz` / `--base-rpy` 时，输出的是 `pelvis -> target_link` 的相对位姿。
- 传入 `--base-link pelvis` 加 `--base-xyz` 和姿态后，输出的是外部世界系 `world -> target_link` 的位姿。
- URDF 文件里注释掉了 `world -> pelvis` 的 floating base joint，所以真实世界中的绝对位姿必须由外部传入，例如来自定位、里程计、运动捕捉或机器人自身状态估计。

常见目标 link：

- `d435_link`：G1 头部/胸部附近的 D435 相机坐标系。
- `mid360_link`：雷达坐标系，如果 URDF 版本包含它。
- `left_wrist_yaw_link`：左腕 yaw link。
- `right_wrist_yaw_link`：右腕 yaw link。
- `left_rubber_hand` / `right_rubber_hand`：不带灵巧手模型里的橡胶手末端。
- `left_hand_palm_link` / `right_hand_palm_link`：带手模型里的手掌 link。

## 输入参数

### `--urdf`

必填。URDF 文件路径。

G1 29DoF 推荐优先使用：

```text
robots/g1/g1_29dof_rev_1_0.urdf
```

示例：

```powershell
--urdf .\unitree_ros\robots\g1_description\g1_29dof_rev_1_0.urdf
```

### `--target`

必填，至少一个。要查询位姿的目标 link 名称。可以重复传入，也可以用逗号分隔。

示例：

```powershell
--target d435_link
--target left_wrist_yaw_link --target right_wrist_yaw_link
--target d435_link,left_wrist_yaw_link,right_wrist_yaw_link
```

目标必须是 URDF 中存在并且能从 `base_link` 到达的 link。固定关节后的 link 也可以查询，例如相机 link、手掌 link。

### `--joints`

可选。读取一个 JSON 文件，里面保存关节名到关节值的映射。

关节值单位：

- `revolute` / `continuous`：弧度 rad。
- `prismatic`：米 m。

推荐 JSON 格式：

```json
{
  "joints": {
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_elbow_joint": 0.6
  }
}
```

也支持直接传一个对象：

```json
{
  "waist_yaw_joint": 0.0,
  "left_shoulder_pitch_joint": 0.2,
  "left_elbow_joint": 0.6
}
```

如果某个 active joint 没有提供，脚本会默认它为 `0.0`，并在输出的 `missing_joint_values_defaulted_to_zero` 中列出来。

### `--joint`

可选。命令行临时指定单个关节值，格式是 `关节名=数值`。可以重复使用。

示例：

```powershell
--joint left_shoulder_pitch_joint=0.2 `
--joint left_elbow_joint=0.6
```

如果同时使用 `--joints` 和 `--joint`，命令行中的 `--joint` 会覆盖 JSON 文件里的同名关节值。

### `--base-link`

可选。指定输出坐标系挂在哪个 link 上。不填时默认使用 URDF 的根 link。

对 G1 29DoF 来说默认是：

```text
pelvis
```

一般建议显式写出：

```powershell
--base-link pelvis
```

### `--base-xyz`

可选。表示外部输出坐标系到 `base-link` 的平移，单位米。

示例：

```powershell
--base-xyz 1.0 2.0 0.75
```

含义是 `base-link`，例如 `pelvis`，在外部世界系中的位置是 `x=1.0, y=2.0, z=0.75`。

### `--base-rpy`

可选。表示外部输出坐标系到 `base-link` 的姿态，使用 roll、pitch、yaw，单位弧度。

示例：

```powershell
--base-rpy 0.0 0.0 1.57
```

如果没有传 `--base-quat-xyzw`，脚本会使用 `--base-rpy`。如果两者都不传，base 姿态默认为零。

### `--base-quat-xyzw`

可选。表示外部输出坐标系到 `base-link` 的姿态，四元数顺序为 `x y z w`。

示例：

```powershell
--base-quat-xyzw 0.0 0.0 0.7071 0.7071
```

如果同时传了 `--base-rpy` 和 `--base-quat-xyzw`，脚本优先使用 `--base-quat-xyzw`。

### `--quat-order`

可选。控制输出四元数顺序。

可选值：

- `xyzw`：默认，ROS 常用顺序。
- `wxyz`：部分数学库或机器人 SDK 常用顺序。

示例：

```powershell
--quat-order xyzw
--quat-order wxyz
```

### `--clamp-to-limits`

可选。把输入关节值裁剪到 URDF 中的 `limit lower/upper` 范围内。

示例：

```powershell
--clamp-to-limits
```

注意：默认不裁剪。默认行为更适合调试，因为如果上游传入了异常关节值，可以直接在结果里暴露出来。

### `--list-active-joints`

可选。只输出 URDF 中所有 active joint 名称，然后退出，不计算目标 link 位姿。

示例：

```powershell
python .\robot_kinematics\joint_to_pose\fk_urdf.py `
  --urdf .\unitree_ros\robots\g1_description\g1_29dof_rev_1_0.urdf `
  --list-active-joints
```

这个命令适合用来核对 Unitree SDK 或日志里的关节顺序是否能映射到 URDF joint name。

### `--pretty`

可选。让 JSON 输出更易读。不给这个参数时，输出会是一行紧凑 JSON，更适合程序读取。

## 输出字段

脚本输出 JSON。典型结构如下：

```json
{
  "urdf": "robots/g1/g1_29dof_rev_1_0.urdf",
  "root_links": ["pelvis"],
  "base_link": "pelvis",
  "missing_joint_values_defaulted_to_zero": [
    "left_hip_pitch_joint",
    "left_hip_roll_joint"
  ],
  "targets": {
    "d435_link": {
      "position_xyz": [0.05366, 0.01753, 0.47387],
      "orientation_quat_xyzw": [0.0, 0.403545, 0.0, 0.91496],
      "transform_matrix": [
        [0.674302, 0.0, 0.738455, 0.05366],
        [0.0, 1.0, 0.0, 0.01753],
        [-0.738455, 0.0, 0.674302, 0.47387],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  }
}
```

字段解释：

- `urdf`：本次加载的 URDF 文件路径。
- `root_links`：URDF 中没有父 joint 的根 link。G1 通常是 `pelvis`。
- `base_link`：本次输出位姿所使用的基准 link。
- `missing_joint_values_defaulted_to_zero`：没有输入、被默认设为 `0.0` 的 active joints。
- `targets`：每个目标 link 的计算结果。
- `position_xyz`：目标 link 原点在输出坐标系中的位置，单位米。
- `orientation_quat_xyzw`：目标 link 相对输出坐标系的姿态四元数，顺序为 `x, y, z, w`。
- `orientation_quat_wxyz`：如果使用 `--quat-order wxyz`，输出字段会变成这个，顺序为 `w, x, y, z`。
- `transform_matrix`：4x4 齐次变换矩阵，表示 `T_output_target`。

`transform_matrix` 的含义：

```text
[ R00 R01 R02 X ]
[ R10 R11 R12 Y ]
[ R20 R21 R22 Z ]
[  0   0   0  1 ]
```

其中左上角 `3x3` 是旋转矩阵，最后一列前三个数是位置 `X, Y, Z`。

## Example 1：G1 零位下查询相机和左右腕

# 从本仓库根目录运行示例：

```bash
python robot_kinematics/joint_to_pose/example_g1_zero_pose.py \
  --urdf robots/g1/g1_29dof_rev_1_0.urdf
```

```powershell
python .\robot_kinematics\joint_to_pose\fk_urdf.py `
  --urdf .\unitree_ros\robots\g1_description\g1_29dof_rev_1_0.urdf `
  --target d435_link `
  --target left_wrist_yaw_link `
  --target right_wrist_yaw_link `
  --pretty
```

这个例子没有提供任何关节值，所以所有 active joints 都按 `0.0` 计算。输出的是 `pelvis -> d435_link`、`pelvis -> left_wrist_yaw_link`、`pelvis -> right_wrist_yaw_link`。

这适合做第一步 sanity check：确认 URDF 能解析、link 名称正确、坐标方向大致合理。

也可以运行示例脚本：

```powershell
python .\robot_kinematics\joint_to_pose\example_g1_zero_pose.py
```

## Example 2：用 JSON 输入一组关节值

先创建一个关节值文件，例如 `my_joint_values.json`：

```json
{
  "joints": {
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.1,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0
  }
}
```

再运行：

```powershell
python .\robot_kinematics\joint_to_pose\fk_urdf.py `
  --urdf .\unitree_ros\robots\g1_description\g1_29dof_rev_1_0.urdf `
  --joints .\my_joint_values.json `
  --target left_wrist_yaw_link `
  --pretty
```

这个例子会输出左腕末端在 `pelvis` 坐标系下的位姿。没有写入 JSON 的腿部、右臂等关节仍然默认是 `0.0`。

## Example 3：命令行临时覆盖关节值

如果只是临时测试几个关节，不想新建 JSON 文件，可以直接用 `--joint`：

```powershell
python .\robot_kinematics\joint_to_pose\fk_urdf.py `
  --urdf .\unitree_ros\robots\g1_description\g1_29dof_rev_1_0.urdf `
  --joint left_shoulder_pitch_joint=0.2 `
  --joint left_shoulder_roll_joint=0.1 `
  --joint left_elbow_joint=0.6 `
  --target left_wrist_yaw_link `
  --pretty
```

这个命令适合快速测试某个关节变化对末端位姿的影响。

## Example 4：输出世界坐标下的相机位姿

如果已知 `pelvis` 在世界系中的位姿，可以传入 base pose：

```powershell
python .\robot_kinematics\joint_to_pose\fk_urdf.py `
  --urdf .\unitree_ros\robots\g1_description\g1_29dof_rev_1_0.urdf `
  --base-link pelvis `
  --base-xyz 1.0 2.0 0.75 `
  --base-rpy 0.0 0.0 1.57 `
  --target d435_link `
  --pretty
```

这个输出不再只是 `pelvis -> d435_link`，而是：

```text
world -> d435_link
```

其中：

```text
T_world_d435 = T_world_pelvis * T_pelvis_d435
```

如果外部系统给的是四元数姿态，可以改用：

```powershell
--base-quat-xyzw 0.0 0.0 0.7071 0.7071
```

## Example 5：切换其它 URDF 模型

算法不绑定 G1 的某一个 URDF。只要换 `--urdf` 路径即可。

例如使用带手版本：

```powershell
python .\robot_kinematics\joint_to_pose\fk_urdf.py `
  --urdf .\unitree_ros\robots\g1_description\g1_29dof_with_hand_rev_1_0.urdf `
  --target left_hand_palm_link `
  --target right_hand_palm_link `
  --pretty
```

例如使用锁腰版本：

```powershell
python .\robot_kinematics\joint_to_pose\fk_urdf.py `
  --urdf .\unitree_ros\robots\g1_description\g1_29dof_lock_waist_rev_1_0.urdf `
  --target left_wrist_yaw_link `
  --pretty
```

使用不同 URDF 时，要同步检查实际机器人型号、`mode_machine`、是否带手、是否锁腰，否则关节数量和 link 名称可能不一致。

## 与实机数据对接时的注意事项

- Unitree SDK 或日志中的关节数组通常是固定顺序数组，而本脚本需要 `joint_name -> value` 映射。实际接入时建议先建立一张关节顺序映射表。
- 关节角单位必须确认是 rad。如果上游给的是 degree，需要先转成 rad。
- 四元数顺序必须确认。ROS 常用 `x, y, z, w`，有些 SDK 或数学库使用 `w, x, y, z`。
- 没有真实 `T_world_pelvis` 时，不能声称输出是世界坐标，只能说是相对 `pelvis` 的坐标。
- URDF 中的 `fixed` joint 不会出现在 active joint 列表里，但 fixed joint 后面的 link 仍可作为 `--target` 查询。
- 当前脚本不把 floating joint 当作普通标量关节处理。移动底座或 humanoid 根节点位姿应通过 `--base-xyz` 和姿态参数传入。
