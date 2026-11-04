# Vicon2Smpl

从 Vicon 光学动捕的**带标签 marker** 实时求解 SMPL（也支持 SMPL-H / SMPL-X 的身体部分），并实时可视化。

路线是"离线标定 + 在线逐帧优化"：

```
          离线（每个被试一次，约 1 分钟）                       在线（每帧 ~2 ms）
 ┌──────────────────────────────────────────┐      ┌──────────────────────────────────────────┐
 │ T-pose + ROM 录制（record.py 或 C3D）     │      │ Vicon DataStream ──► 最新帧（后台线程）  │
 │   A. T-pose 拟合（多朝向假设）            │      │   按标签映射 → 遮挡 mask                 │
 │   B. marker → 最近体表顶点（自动对应）    │ ───► │   LM 求解 θ、trans（解析雅可比）         │
 │   C. 跟踪整段 ROM，挑 40 个差异最大的姿态 │ calib│   Huber 鲁棒 + 时间平滑 + 姿态先验       │
 │   D. 联合优化 β、marker 偏移、各帧姿态    │ .npz │   丢失自动重初始化                       │
 │   E. 反蒙皮重新分配顶点，重复 C-D         │      │ ──► aitviewer / UDP / AMASS 格式 .npz    │
 └──────────────────────────────────────────┘      └──────────────────────────────────────────┘
```

## 效果（合成数据测试）

由于 SMPL 模型文件有授权限制、测试环境也没有 Vicon 硬件，仓库自带一个**与 SMPL 同构的合成人体**（24 关节树、胶囊几何、形状/姿态 blendshape）和 marker 模拟器（1.5 mm 噪声、突发遮挡、标签互换）来做端到端测试。数字反映的是算法本身，不代表真实数据上的精度（真实数据还有软组织形变、marker 晃动等）。

| 测试 | 结果 |
|---|---|
| 已知标定（oracle）在线求解 | MPJPE 1.6 mm |
| 全自动标定（不给 marker→顶点 对应）后，在**未见过的新动作**上 | MPJPE 6–8.5 mm（3 个随机种子） |
| 求解耗时（2.1 GHz Xeon，单线程，numba） | 中位数 2.2 ms，p99 ≈ 4–7 ms |
| 无 numba（纯 numpy） | 中位数 ≈ 5 ms |
| CLI 集成测试：C3D 标定 → 100 Hz 实时回放 + 可视化 + UDP | 1498/1500 帧被求解，MPJPE 6.2 mm |

## 安装

```bash
pip install -r requirements.txt
pip install aitviewer --no-deps                       # 只用它的 remote 客户端
pip install "<Vicon DataStream SDK>/<平台>/Python/vicon_dssdk"   # 官方 SDK 自带的 Python 包
```

aitviewer 本体依赖 `numpy<2`、`scipy<1.8`，建议**另开一个环境**跑查看器：

```bash
conda create -n aitv python=3.10 && conda activate aitv && pip install aitviewer
python -m aitviewer.server        # 打开一个空窗口，等待推送
```

SMPL 模型文件从 [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) / [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) 下载，按 `smplx` 包的目录约定放置：

```
body_models/
  smpl/SMPL_NEUTRAL.pkl  (SMPL_MALE.pkl, SMPL_FEMALE.pkl)
  smplx/SMPLX_NEUTRAL.npz
```

## 使用

### 1. 录一段标定数据

被试先**静止 T-pose 2–3 秒**，然后做 30–60 秒 ROM：手臂画圈、屈肘、深蹲、抬腿、躯干扭转/侧屈、转头、踮脚。

```bash
python record.py --host 192.168.1.10:801 --subject Actor1 --out calib_take.npz   # Ctrl+C 结束
```

也可以直接用 Nexus / Shogun 导出的 **C3D**（默认按 Z-up、毫米处理；Y-up 的文件加 `--up-axis y`）。

### 2. 标定

```bash
python calibrate.py --rec calib_take.npz --model-path ./body_models --model-type smpl \
    --gender neutral --tpose 0.3:2.5 --out actor1_calib.npz
```

- `--tpose`：开头静止姿态的时间窗（秒）；参数名为兼容旧调用保留。
- `--static-pose t|a`：明确静止段是 T-pose 还是 A-pose。A-pose 可加 `--arm-angle 60`
  指定手臂下垂角的初值；实际关节姿态继续由观测优化。A-pose 的联合标定约束保留已拟合的静止姿态，
  不会把静止帧拉回 T-pose。后续自由活动可用，但未活动到的关节约束可能不足。
- `--facing`：T-pose 时被试朝向（SMPL 坐标轴：Vicon +X → +x，Vicon +Y → −z）。默认 `auto` 会对多个朝向假设各跟踪 10 秒 ROM、选残差最小的；输出里出现 "ambiguous" 时手动指定。
- `--marker-map labels.json`：如果你有 `{marker标签: SMPL顶点id}` 的对应表（例如来自 MoSh++/SOMA 的标记布局），可跳过自动对应。
- `--marker-regions regions.json`：调用方显式提供 `{标签: [允许的顶点id]}`，可在对应身体部位内寻找实际贴点。
  此约束贯穿静止拟合、初次对应和两种重新分配；与固定 `--marker-map` 互斥。区域定义属于调用方先验，
  需要按现场标签和模型核对，不默认把任意名字解释为某个身体部位。
- 输出会列出残差偏大的 marker（贴错位置/标签错误时会在这里暴露）。

### 3. 实时运行

```bash
# 终端 1（aitv 环境）
python -m aitviewer.server
# 终端 2
python run_realtime.py --calib actor1_calib.npz --model-path ./body_models \
    --host 192.168.1.10:801 --subject Actor1 [--udp 127.0.0.1:9763] [--save out.npz]
```

- Shogun 完整人体流使用 `--host <ip>:801`；较新版本的 Always Live 为 `8802`。
  `804` 在较新 Shogun 中是刚体 ObjectTracker 输出，不能用于全身被试。以现场版本和实际标签为准。
- 不连 Vicon 也能调试：`--replay calib_take.npz`（或 `.c3d`）按原帧率回放。
- `--viz none` 关闭可视化；`--spawn-viewer` 自动拉起查看器进程（需同一环境装全 aitviewer）。
- 每 2 秒打印：求解帧率、耗时、marker 残差、可见 marker 数、被跳过的帧数。求解比采集慢时总是处理**最新帧**，不会积压延迟。

## 输出

- **aitviewer**：mesh 在本进程算好再推送（查看器那边不需要 SMPL 文件），marker 用红球显示，独立线程按 `--viz-fps` 发送，不阻塞求解。
- **UDP**（`--udp host:port`）：每帧一个包，小端：`b"VSMP" | u32 version | i64 frame | f64 time | u32 J | f32 rmse | f32[3] trans | f32[J*3] pose(轴角)`。解码见 `vsmpl/outputs.py: UdpPublisher.decode`，Unity/UE 端照此解析即可。
- **`--save out.npz`**：AMASS 风格，`poses (T, J*3)`、`root_orient`、`trans`、`betas`、`gender`、`model_type`、`mocap_framerate`、`frame_numbers`、`rmse`。

坐标系统一为 **SMPL 约定：米、Y 轴向上**（Vicon 的 Z-up 毫米在读入时就转换了：`(x, y, z) → (x, z, −y)`）。

## 算法要点

- **marker 模型**：每个 marker = 某个体表顶点（随 β 变化）+ 静止空间里的 3D 偏移，用该顶点的蒙皮权重和姿态 blendshape 驱动（"虚拟顶点"）。在线时 β 固定，只需对几十个点做 LBS。
- **在线求解**：SO(3)^K × R³ 上的 Levenberg–Marquardt，关节旋转用右乘增量 `R ← R·exp(δ)` 更新，数据项雅可比是解析的（`MarkerRig.jacobian`，已对数值差分验证）。代价 = Huber 鲁棒数据项（遮挡 marker 直接 mask，离预测 >10 cm 的视为标签错误/鬼点剔除）+ 朝匀速预测的时间平滑项 + 很弱的静止姿态先验。每帧以上一帧外推为初值迭代 4 次。连续 10 帧残差 >5 cm 判定丢失，用躯干 marker 做 Kabsch 刚体对齐后重新初始化。
- **不可观关节**自动冻结在静止姿态：没有 marker 影响的关节（例如没贴手部 marker 时的 SMPL-X 手指、下颌）不参与优化。
- **标定中的两个关键设计**：
  - T-pose 几乎前后对称，单帧几何分不清朝向；运动不对称（膝盖只往一边弯），所以用"每个朝向假设跟踪一段 ROM 看残差"来判别。
  - 单帧最近顶点对应容易贴错面（例如外踝贴到内踝）；跟踪整段后把观测**反蒙皮**回静止空间取中位数，再重新找顶点（类 EM），对应由运动数据决定。

## marker 集要求与已知限制

- **每个刚体节段需要 ≥3 个不共线约束**（节段上的 marker，加上关节链传来的约束）。只有 2 个 marker 的节段，绕两点连线的旋转不可观——标定时 marker 位置可以整体"转一圈"而残差不变。常见全身标记集（Plug-in Gait、Vicon Production）在脚上有踝/跟/趾，满足要求；手若只贴 1 个 marker，手的姿态主要靠先验。
- `ViconSource` 按 DataStream SDK 的 Python API 编写（`Connect / EnableMarkerData / SetStreamMode / GetFrame / GetSubjectNames / GetMarkerNames / GetMarkerGlobalTranslation`），**尚未在真机上测试**。不同 SDK 版本若函数名或返回值有出入，只需改 `vsmpl/sources.py` 里这一个类。
- 标定的 β 与 marker 沿体表的滑动之间存在固有的可交换性（与 MoSh 相同），合成测试中静止形状的顶点误差约 1 cm；它对关节位置的影响已体现在上表 MPJPE 中。
- 当前只支持单个被试；姿态先验是简单的 L2，没有接 VPoser/GMM，极端姿态下若 marker 稀疏可能出现不自然的扭转（可调 `--lam-temp`、`SolverConfig.lam_prior`）。

## 测试

```bash
python tests/test_core.py        # SO(3) 内核一致性、前向（numpy vs torch）、解析雅可比 vs 数值差分、已知标定求解
python tests/test_pipeline.py 0 1 2   # 全自动标定 + 新动作跟踪，三个随机种子
python tests/test_cli.py         # C3D → calibrate.py → run_realtime.py --replay（假 viewer 服务器 + UDP + 保存）
VSMPL_NO_NUMBA=1 python tests/test_core.py   # 纯 numpy 路径
```

## 目录

```
record.py            录制 Vicon marker → .npz
calibrate.py         被试标定（β + marker 位置）
run_realtime.py      实时求解 + 可视化 / UDP / 保存
vsmpl/
  body_model.py      numpy 体模型、FK、MarkerRig（前向 + 解析雅可比）
  solver.py          在线 LM 求解器
  calibration.py     离线标定流程
  torch_model.py     可微前向（标定用）
  sources.py         Vicon 实时源、回放、C3D 读取、标签映射、坐标转换
  viz.py             aitviewer 远程推送
  outputs.py         UDP 与 AMASS 格式保存
  rotations.py       SO(3) 工具
  _kernels.py        可选 numba 内核
tests/               合成人体与测试
```

SMPL / SMPL-X 模型及其衍生数据受各自许可证约束，请勿提交到仓库（`.gitignore` 已排除 `body_models/`、`*.pkl`、`*.npz`）。

## 浏览器现场测试

`run_web.py` 提供 Viser 网页入口：被试选择、5 秒准备倒计时、T-pose/ROM 采样、
标定计算、A-pose/T-pose 选择、人体与观测/拟合 marker 对照，以及停止后保存参数。
它独立运行，不调用 Shogun 开拍/停拍、不改正式录制服务或历史原始数据。
每个人、每次重新贴点都应重新标定；手指效果需要用实际贴点与动作检查。

Linux 使用随官方 SDK 编译的只读 C++ 桥接器，因为官方 Python 二进制包主要面向 Windows。
SDK、模型与环境不进入 Git。将以下占位路径换成已经检查过的现场安装位置：

```sh
pip install -r requirements-web.txt
g++ -O2 -std=c++17 native/live_markers.cpp -I<SDK-header-directory> \
  -L<SDK-library-directory> -Wl,-rpath,<SDK-library-directory> \
  -lViconDataStreamSDK_CPP -pthread -o native/live_markers
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run_web.py \
  --source <Shogun-IP>:8802 --native-reader ./native/live_markers \
  --model-path ./body_models --model-type smplx --bind <lab-IP> --port 8792
```

打开 `http://<lab-IP>:8792`。先在 Shogun 确认正确人物和全身标签，再在网页手动启动标定采样。
先选择实际采用的静止姿态，倒计时后保持该姿态 3 秒，其余时间做活动范围动作或自由活动（默认总采样 45 秒）。
采样结束后点击计算标定，完成后再点击实时拟合。首次初始化与 JIT 编译耗时另计。
所有现场测试保存在独立 `runtime/<UTC时间_唯一编号>/`：采样 marker、采样时钟、标定、日志、
拟合序列和 `session.json`。`runtime/status.json` 是当前运行状态，不表示精度验收。

正式数据集接入时，算法结果放入其 `derived/<algorithm>/<run_id>/`，不要替换原始采集入口。
服务只绑定实验室地址或回环地址；外网查看使用已有 SSH 连接转发，不直接暴露控制页。

现场端口依据：[Shogun 1.14 官方手册](https://help.vicon.com/download/attachments/196380086/Getting%20started%20with%20Vicon%20Shogun.pdf)。

已保存的标定候选可用 `run_web.py --calib <subject_calib.npz> --subject <实际被试名>` 加载，
不必重复采样。文件必须含匹配的 `source_subject`；加载后仍需检查真实几何和运动，不自动开始拟合。

新被试或重新贴点时应重新采样和标定。网页加 `--body-regions <已核对区域.json>` 可固定为身体/手腕模式：
保留完整来源采样，冻结本次区域文件，在标定 worker 中仅使用白名单标签，排除手指点。
新人的体型、对应顶点及偏移均重新估计；输出保留新被试名、排除点名及区域快照 SHA256。
缺少配置所需标签或区域不合法时明确失败，不回退到无约束的全点拟合。

### SMPL-X 身体和双手

网页加 `--fit-hands --hand-prior <pose_hand_prior.npz> --hand-components 20`，并保留已核对的
`--body-regions`。完整72点来源用于两阶段标定：先拟合身体，再为同一被试联合优化身体姿态、腕部及手指；
身体体型和52点贴点参数保留，20个手指贴点在正确手指/指段内重新求解。
项目的 `3/MID0` 是近掌点、`6` 是近指尖点，不能直接套用 MoSh 同名 `IDX3` 的定义。

在线使用原生 SMPL-X55 的完整 FK/LBS 和姿态形变；手指由指定学习手形先验的 PCA 系数约束。
手部稀疏点不能独立确定每个指节的所有旋转，因此这些姿态包含先验推断，仍需张握、拇指与捏合近景检查。
全身求解的数据雅可比包含姿态 blendshape 导数；原身体求解器默认保留上游近似路径。

手形先验与模型仍在授权目录，不能提交 Git。采样请求冻结先验路径、SHA256、分量数和区域配置；
输出scope为 `body_hands`，保留55关节原生轴角参数，网页分别显示身体与手指的 marker RMS。
离线手指贴点使用法向0–15mm、初值5mm和每坐标4mm的切向范围，这是优化先验，不能当作实测贴点高度。
