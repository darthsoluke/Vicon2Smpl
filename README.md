# Vicon2Smpl

从 Vicon 光学动捕的**带标签 marker** 实时求解 SMPL 系列人体，并实时可视化。
当前现场网页支持原生 **SMPL-X 身体＋双手 72 点**，包含头颈、手腕和手指；也保留 SMPL / SMPL-H 身体路径。
也可以用 NVIDIA **SOMA** 人体模型（`--model-type soma`，身体＋手腕），直接输出 SOMA 的 77 关节姿态与身份系数，
原理、用法和限制见 [SOMA 后端](docs/SOMA.md)。
网页传入 `--model-profiles <本地配置.json>` 后，可在“人体模型”中切换 SMPL-X 与 SOMA；两者分别标定。

实验室共享部署、精确依赖离线包、单实例启动和健康检查见 [部署说明](docs/DEPLOYMENT.md)。
4090D 的目录、网页、SSH 与用户服务入口见 [4090D 使用说明](docs/4090D.md)。
推荐现场入口是 `run_web.py` 的 Viser 网页；下方 `aitviewer` 是原 CLI 可选路径。

路线是"离线标定 + 在线逐帧优化"：

```
          离线（每个被试/每次重新贴点）                       在线（耗时取决于模型和机器）
 ┌──────────────────────────────────────────┐      ┌──────────────────────────────────────────┐
 │ T-pose + ROM 录制（record.py 或 C3D）     │      │ Vicon DataStream ──► 最新帧（后台线程）  │
 │   A. T-pose 拟合（多朝向假设）            │      │   按标签映射 → 遮挡 mask                 │
 │   B. marker → 最近体表顶点（自动对应）    │ ───► │   LM 求解 θ、trans（解析雅可比）         │
 │   C. 跟踪整段 ROM，挑 40 个差异最大的姿态 │ calib│   Huber 鲁棒 + 时间平滑 + 姿态先验       │
 │   D. 联合优化 β、marker 偏移、各帧姿态    │ .npz │   丢失自动重初始化                       │
 │   E. 反蒙皮重新分配顶点，重复 C-D         │      │ ──► aitviewer / UDP / AMASS 格式 .npz    │
 └──────────────────────────────────────────┘      └──────────────────────────────────────────┘
```

上图概括原 CLI；网页支持 A-pose 初始化、解剖区域约束、双手 PCA 以及 Viser 显示。
下方合成 SMPL 测试耗时不代表原生 SMPL-X 全身双手的现场耗时，输入、求解和网页显示频率分别统计。

## 它是怎么工作的

一句话：先花一两分钟“认识”这个人，搞清楚他的体型、每个 marker 贴在他身上哪里；之后每来一帧，就把人体模型的姿态调到“模型上的点”和“真实 marker”尽量重合。整个过程是纯优化，没有神经网络，也不需要训练数据。

### marker 是长在皮肤上的点

我们把每个 marker 看成粘在人体模型皮肤上的一个点：它挂在某个体表顶点 $i_m$ 上，再加一个小偏移 $o_m$（marker 球的半径，加上贴得不那么正的部分）。在静止姿态下，它的位置是

$$
p_m(\beta) = v_{i_m}(\beta) + o_m,\qquad v(\beta) = \bar v + S\beta
$$

其中 $\beta$ 是体型参数。人动起来以后，这个点跟着它所在那块皮肤一起动，用的就是人体模型本身的线性蒙皮（LBS）：

$$
x_m(\theta, t) = \sum_j w_{i_m j}\,\big(R^w_j\,(p_m - J_j) + c_j\big) + t
$$

$w$ 是这个顶点对各关节的蒙皮权重，$R^w_j$ 和 $c_j$ 是关节 $j$ 在当前姿态 $\theta$ 下的世界旋转和位置（由父关节一层层乘下来：$R^w_j = R^w_{\mathrm{parent}(j)} R_j$），$J_j$ 是静止时的关节位置，$t$ 是整体平移。SMPL 系列还会在 $p_m$ 上加一点随姿态变化的形变，原理一样。

对一个人来说，$\beta$、$i_m$、$o_m$ 都是常量；会变的只有 $\theta$ 和 $t$。**标定**负责求前者，**实时**负责求后者。

Vicon 给的 marker 都带标签（LASI、RFHD……）。标签在这里的作用是“身份证”：同一个标签在所有帧里就是同一个物理点。程序并不理解标签名的含义；贴在哪个部位，是靠区域文件（每个标签允许落在哪片顶点）和数据自己算出来的。

### 标定：先认识这个人

网页上点“开始标定采样”后，被试先在 A-pose 或 T-pose 下静止约 3 秒，然后做 15–90 秒的活动范围动作（抬手、下蹲、转体、踢腿、转头……）。接下来分几步走。

**1. 用静止段粗对齐。** 把静止那几秒取平均，拟合整体朝向、位置、体型和少量姿态，让 marker 尽量贴在模型表面上。这一步还不知道哪个点对应哪个顶点，只看“marker 到最近体表点的距离”；有区域文件时，每个标签只在自己的区域里找。

**2. 判断人朝哪边。** 静止站着几乎前后对称，单看几何分不清朝向；但动作不对称（膝盖只往前弯），所以对最可能的两三个朝向各跟踪 10 秒动作，残差最小的那个就是答案。

**3. 挑代表性的帧，然后一起优化。** 用实时求解器把整段动作跟一遍，只保留所有点都正常、残差小于 3 cm 的帧，再从里面挑出 40 帧彼此差异最大的姿态（加上静止帧）。在这些帧上，体型、所有偏移和每一帧的姿态一起优化：

$$
\min_{\beta,\,o,\,\{\theta_f, t_f\}}\ \sum_{f,m}\rho\big(\lVert x_m(\theta_f, t_f) - y_{fm}\rVert\big)
+ \lambda_\beta\lVert\beta\rVert^2
+ \lambda_n\sum_m\big(n_m^\top o_m - r\big)^2
+ \lambda_\tau\sum_m\lVert o_m^{\parallel}\rVert^2
+ \lambda_a\,\big\lVert\log\big(R^{\mathrm{ref}\top}R^{\mathrm{static}}\big)\big\rVert^2
$$

逐项说一下：$y_{fm}$ 是第 $f$ 帧、标签 $m$ 的观测；$\rho$ 是 Huber 损失，偶尔有个点跳了也不至于把结果带偏；第二项让体型别太离谱；第三项要求偏移沿法向的长度接近 marker 半径 $r$（这个约束很强）；第四项允许 marker 沿皮肤稍微滑一点；最后一项把静止帧钉在 A/T 参考姿态上。最后这一项很关键：没有它，“这一节肢体整体多转一点”和“marker 沿皮肤多滑一点”会互相抵消，谁也说不清。

注意这里虽然也解出了每帧的 $\theta$，但它们只是帮忙估体型和贴点的中间量，标定结束就扔掉，不会存下来。

**4. 让运动数据决定贴点。** 单帧找最近顶点很容易贴错面，比如外踝的点被贴到了内踝。所以我们用刚跟踪出来的姿态，把每一帧的观测“反蒙皮”回静止空间，取中位数，再贴回最近的体表点：

$$
\hat p_m = \operatorname*{median}_f\Big[\Big(\sum_j w_{mj}R^w_{fj}\Big)^{-1}\Big(y_{fm} - t_f - \sum_j w_{mj}A_{fj}\Big)\Big],
\qquad A_{fj} = c_{fj} - R^w_{fj}J_j
$$

$$
i_m = \arg\min_{i\,\in\,\mathrm{region}(m)}\ \lVert v_i + r\,n_i - \hat p_m\rVert
$$

第 3、4 步交替做两轮，有点像 EM。到这里得到的是身体部分（52 个身体加手腕的点），存成 `body_wrist_only`。

**5. 手指（SMPL-X55）。** 身体的体型和贴点冻结不动，只处理 20 个指点。手指自由度太多、每根手指的点太少，所以手的姿态用 PCA 降维，只在常见手势的空间里变化：

$$
\theta_{\mathrm{hand}} = \mu + U z
$$

每个指点的偏移拆成“沿法向”和“沿皮肤”两部分，并且都有边界：

$$
o = h\,n + \tau,\qquad 0 < h < 15\ \mathrm{mm},\qquad \tau \perp n,\ \ |\tau_i| \le 4\ \mathrm{mm}
$$

在 48 帧差异最大的姿态上，先用 Adam 再用 L-BFGS 优化 $z$、$h$、$\tau$（身体姿态只允许微调），然后在各自指段里重新找贴点，同样做两轮。结果存成 `body_hands`。

**6. 正头参考修正（可选）。** 操作者确认一帧“站直、正头、头肩同向”的静止姿态，只重新确定 4 个头部点（LFHD/LBHD/RFHD/RBHD）的贴点，让这一帧解出来的头确实是正的。这个“正头”约束只在标定时存在，实时求解里没有。

最终每个人的标定文件里是：体型 $\beta$、每个点的贴点 $i_m$ 和偏移 $o_m$，以及手部先验的信息。

### 实时：看他现在在做什么

标定好之后，每来一帧，就解一个小的最小二乘问题：

$$
E(\theta, t) =
\sum_{m\ \mathrm{visible}}\rho\big(\lVert x_m(\theta, t) - y_m\rVert\big)
+ \lambda_{\mathrm{temp}}\sum_j\big\lVert\log(\hat R_j^\top R_j)\big\rVert^2 + \lambda_{\mathrm{tr}}\lVert t - \hat t\rVert^2
+ \lambda_p\sum_{j\ne \mathrm{root}}\lVert\log R_j\rVert^2
$$

三项依次是：贴合 marker、别离预测太远、弱先验。

**贴合 marker。** 被遮挡的点直接不算；离预测超过 10 cm 的点多半是标签串了或者鬼点，也直接剔除。

**别离预测太远。** 预测 $\hat R, \hat t$ 是从前两帧按匀速外推的，意思是“照刚才的速度，这一帧大概到这里”：

$$
\hat R_j = R^{(1)}_j\exp\!\Big(g\,\log\big(R^{(0)\top}_j R^{(1)}_j\big)\Big),\qquad \hat t = t^{(1)} + g\,\big(t^{(1)} - t^{(0)}\big)
$$

$g \approx 0.8$，丢帧时按间隔放大。它同时是 LM 的初值，所以每帧迭代 4 次左右就够了。这一项很弱（$\lambda_{\mathrm{temp}} = 10^{-3}$）：关节偏离预测 6° 的代价，只相当于一个 marker 差 3 mm。点看得见的时候，结果几乎完全由 marker 决定；点被挡住时，关节就按刚才的趋势继续走，不会突然跳。

**弱先验。** 把关节轻轻拉回静止姿态（$\lambda_p = 10^{-4}$，根关节不拉）。它的作用是给 marker 测不出来的方向一个默认值，比如只贴两个点的肢体绕连线的转动，让方程始终可解。偏离 29° 也只相当于一个 marker 差 5 mm，正常动作完全不受影响。完全没有 marker 的关节（比如没贴点的手指）干脆冻结，不参与优化。

**怎么解。** 在旋转群上做 Levenberg–Marquardt，每次右乘一个小转动：

$$
(H + \mu\,\mathrm{diag}\,H)\,\delta = -g,\qquad R_j \leftarrow R_j\exp(\delta_j^\wedge),\qquad t \leftarrow t + \delta_t
$$

雅可比是解析推出来的：关节 $k$ 一转，只会带动它子树上的点，

$$
\frac{\partial x_m}{\partial \delta_k} = -\Big[\sum_{j\in\mathrm{sub}(k)} w_{mj}\,q_{mj} - \Big(\sum_{j\in\mathrm{sub}(k)} w_{mj}\Big)c_k\Big]_\times R^w_k,
\qquad \frac{\partial x_m}{\partial t} = I
$$

$q_{mj}$ 是被关节 $j$ 变换后的 marker 点。手指在 PCA 系数 $z$ 上求解，雅可比里还算上了姿态形变的导数。

**跟丢了怎么办。** 连续 10 帧残差超过 5 cm，就用躯干上的点做一次刚体对齐（Kabsch），重新开始跟踪。

**有多快。** 身体部分单帧求解中位数 2.3–3 ms（单线程，numba 加速），也就是三四百帧每秒的能力；100 Hz 采集时回放测试实际解出 96–100 Hz。求解跟不上时程序总是处理最新一帧，不会越积越慢。网页显示是 20 Hz 左右，那只是画面刷新率。全身加双手的速度以现场网页每 2 秒打印的 “solved Hz / solve ms” 为准。

### SOMA 怎么接进来的

SOMA 的每个关节自带一个绑定朝向 $B_j$，公开的姿态参数又是相对另一个 T-pose 朝向 $O_j$ 定义的：

$$
R^{\mathrm{abs}}_j = O_p^\top R^{\mathrm{rel}}_j O_j,\qquad W_j = W_p R^{\mathrm{abs}}_j,\qquad v' = \sum_j w_{vj}\big(W_j B_j^\top (v - b_j) + x_j\big)
$$

看起来和 SMPL 不一样，但只要换一个求解变量 $R'_j = B_p R^{\mathrm{abs}}_j B_j^\top$，记 $S_j = W_j B_j^\top$，就有

$$
S_j = S_p R'_j,\qquad x_j - x_p = S_p\,(b_j - b_p)
$$

这正是 SMPL 的形式：静止网格是 SOMA 的绑定网格（一个 A-pose），静止关节是 $b$。所以上面的标定和实时求解原封不动地复用，只在输出时换回 SOMA 自己的参数：

$$
R^{\mathrm{rel}}_j = C_p\,R'_j\,C_j^\top,\qquad C_j = O_j B_j^\top,\qquad \mathrm{transl} = t + b_{\mathrm{Hips}}
$$

和 SOMA 官方的 `SOMALayer.pose` 逐顶点对比，最大差约 0.001 mm。SOMA 路径目前只拟合身体和手腕，细节见 [SOMA 后端](docs/SOMA.md)。

### 为什么用优化，不用网络

光学动捕的 marker 多、精度到毫米，这种情况下直接优化有几个好处：不需要训练数据；换个人、换个贴法，重新标定就行；每一帧都有 marker 残差可以检查，结果对不对一目了然。代价是 marker 很少或遮挡很严重时，只能靠先验和时间平滑兜底。真遇到这种情况，比较自然的做法是在代价里再加一个学出来的姿态先验，而不是整个换成网络。

### 参考

- 思路主要来自 MoSh / MoSh++：Loper, Mahmood, Black, *MoSh: Motion and Shape Capture from Sparse Markers*, SIGGRAPH Asia 2014；Mahmood et al., *AMASS: Archive of Motion Capture as Surface Shapes*, ICCV 2019。“marker = 体表顶点 + 偏移”和“先求体型与贴点、再逐帧求姿态”都源自这里。在线逐帧 LM、匀速预测、朝向判定、反蒙皮重新找贴点、正头修正是为实时、全自动的现场流程加的，MoSh++ 的软组织形变没有做。
- 人体模型：SMPL（Loper et al., 2015）、SMPL-X（Pavlakos et al., CVPR 2019）、MANO（Romero et al., 2017，手部 PCA）、SOMA（NVIDIA, *SOMA: Unifying Parametric Human Body Models*, arXiv 2603.16858）。
- 旋转上的优化：Solà et al., *A micro Lie theory for state estimation in robotics*, arXiv 1812.01537。
- 容易混淆的同名工作：Ghorbani & Black, *SOMA: Solving Optical Marker-Based MoCap Automatically*（ICCV 2021）解决的是 marker 自动标注，和上面的 NVIDIA SOMA 人体模型只是重名。我们的输入已由 Vicon 标好标签，没有用到它。

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
- `--marker-map labels.json`：如果你有 `{marker标签: SMPL顶点id}` 的对应表（例如来自 MoSh++ 或 MPI 的 SOMA 自动标注方法的标记布局；它与本仓库的 NVIDIA SOMA 人体模型只是重名），可跳过自动对应。
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
python tests/test_soma.py mid    # SOMA 前向与 SOMALayer.pose 逐顶点一致、参数往返、身份线性化
python tests/test_soma_pipeline.py low t 0    # SOMA 人体上的全自动标定 + 新动作跟踪（经 SOMALayer 评估）
python tests/test_soma_cli.py low             # SOMA：calibrate.py → run_realtime.py --replay → 保存/UDP
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
  soma_model.py      SOMA 后端：绑定相对参数化、身份线性化、SOMA 公共参数换算
scripts/convert_regions_to_soma.py   SMPL-X 区域文件 → SOMA 顶点编号
tests/               合成人体与测试（test_soma*.py 需要 py-soma-x）
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
静止参考时头部自然竖直，头与肩朝同一正前方；不能仅凭头带四点的几何认定人体头部朝向。
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

### 正头参考修正

`vsmpl.head_calibration.calibrate_neutral_head` 可用操作者明确确认的站直、正头、头肩同向静止参考，
重新估计同一被试的四个头部贴点。输入必须保留完整72点及逐帧对应的原生55关节姿态和位移；
体型及其余68个贴点保持原值。头部候选区域同时检查位置前后/左右及原生头关节蒙皮权重，
避免只按表面法线符号把前额标签绑定到后侧。

头部竖直和朝向约束仅用于这次参考标定，在线头颈仍自由求解。修正必须再通过不约束头部的原生求解器重放检查；
参考中的世界竖直轴来自 Vicon 配置，并非独立重力测量。旧标定、参考数据和检查报告应分别保留。
网页的“头部近景”使用世界竖直视角，便于检查歪头；当前头部倾角写入运行状态，只作诊断。

命令行入口会保留旧标定，在新目录输出候选和不锁头的重放报告：

```sh
python calibrate_head.py --model-path <SMPL-X模型> \
  --parent-calibration <同一被试的72点标定.npz> \
  --reference <逐帧匹配的正头参考.npz> --subject <实际被试名> \
  --operator-confirmed-neutral --output-dir <新的派生输出目录>
```

参考格式和返回码见 `python calibrate_head.py --help`。数值重放通过仍需真人视觉和独立动作检查，
不能把标定时强制正头的结果作为精度验收。采样、个人标定与重放 NPZ 只保留在服务器。
