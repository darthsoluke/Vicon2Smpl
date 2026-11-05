# SOMA 后端

`--model-type soma` 用 NVIDIA 的 [SOMA](https://github.com/NVlabs/SOMA-X)（`pip install py-soma-x`，Apache-2.0）作为人体模型，
实时输出 **SOMA 自己的参数**：77 个关节的局部旋转（相对 T-pose 关节朝向，`SOMALayer.pose` 的约定）、Hips 位置和
身份系数（默认 SOMA 原生 128 维 PCA 的前 `num_betas` 维）。原生 SOMA 身份不需要 SMPL 授权文件。

## 为什么能直接复用现有求解器

SOMA 的蒙皮是标准 LBS，只是每个关节带一个绑定朝向 `B_j`，公共姿态参数又相对另一个 T-pose 朝向 `O_j`：

```
R_abs[j] = O[p]^T · R_rel[j] · O[j]          W[j] = W[p] · R_abs[j]
v' = Σ_j w_vj ( W[j] B[j]^T (v − b[j]) + x[j] )
```

如果把求解变量换成"相对绑定朝向"的局部旋转 `R'[j] = B[p] R_abs[j] B[j]^T`，那么 `S[j] = W[j] B[j]^T` 满足
`S[j] = S[p] R'[j]`、`x[j] − x[p] = S[p](b[j] − b[p])`。这正是 SMPL 的 FK 与 LBS：静止网格是绑定网格（一个 A-pose），
静止关节是 `b`，没有关节朝向。于是解析雅可比、numba 内核、LM 求解器和整个标定流程都原样可用，
只在输出时换回 SOMA 约定：

```
R_rel[j] = C[p] · R'[j] · C[j]^T,   C[j] = O[j] B[j]^T        transl = trans + b[Hips]
```

`tests/test_soma.py` 用随机体型和随机姿态，把这套前向与 `SOMALayer.pose` 逐顶点比较：最大差 0.001 mm（float32 精度）。

## 身份系数

SOMA 的体型由其身份管线给出（PCA → 骨架拟合 RBF + Kabsch → 重新摆到绑定姿态），对系数基本线性但不严格：
在 0 点线性化时，各维同时取 1σ 的体型误差约 1 cm。因此：

- 标定每一轮开始前调用 `model.prepare_for(betas)`，在当前体型处用 SOMA 自身做中心差分、重新线性化；
- 加载标定建 rig 时也会 `prepare_for(calib betas)`，实时用的网格和骨架与 SOMA 对该身份的输出完全一致。

## 用法

```bash
pip install py-soma-x            # 首次使用会从 HuggingFace 下载资产（约 0.9 GB，含可选的 MHR 后端）
# 标定（--model-path hf 表示用 HuggingFace 缓存；也可以给本地 SOMA 资产目录）
python calibrate.py --rec calib_take.npz --model-type soma --model-path hf --soma-lod mid \
    --num-betas 10 --static-pose a --tpose 0.3:2.5 --out actor1_soma_calib.npz
# 实时
python run_realtime.py --calib actor1_soma_calib.npz --model-path hf --host <ip>:801 --save out.npz
# 网页
python run_web.py --model-type soma --model-path hf --soma-lod mid --source <Shogun-IP>:8802 \
    --native-reader ./native/live_markers --body-regions body_regions_soma.json ...
```

现有的 SMPL-X 区域文件（顶点编号属于 SMPL-X）要先转换成 SOMA 顶点编号：

```bash
python scripts/convert_regions_to_soma.py --regions body_regions_smplx.json --out body_regions_soma.json --soma-lod mid
```

工具利用 SOMA 资产里贴合在 SMPL-X 模板上的 SOMA 网格（`SMPLX/SOMA_wrap.obj`），让每个 SOMA 顶点继承最近 SMPL-X 顶点的标签，
并写一份 `*.conversion.json` 记录来源文件的 SHA256 和每个标签的顶点数。转换后的区域仍需对照现场贴点核对一次。

## 输出

- `--save` / 网页保存的 npz：`poses (T, 77·3)`、`trans`（Hips 位置）、`identity_coeffs (128,)`、`identity_model_type`、
  `soma_lod`、`soma_joint_names`，以及求解器自己的 `solver_poses` / `solver_trans`。还原方法：

  ```python
  from soma import SOMALayer
  layer = SOMALayer(identity_model_type="soma", enable_procedural_transforms=False, device="cpu")
  layer.prepare_identity(torch.tensor(d["identity_coeffs"][None], dtype=torch.float32))
  out = layer.pose(torch.tensor(d["poses"].reshape(-1, 77, 3)), transl=torch.tensor(d["trans"]), apply_correctives=False)
  ```

- UDP 包结构不变，`J = 77`，姿态与平移同上。

## 验证（合成数据，真实 SOMA 人体）

marker 按全身标记集在 SOMA 网格上生成（约 48 个，1.5 mm 噪声、突发遮挡、一次标签互换），真值体型随机。
标定不给 marker→顶点 对应；误差在**没见过的新动作**上、把输出参数送进 `SOMALayer.pose` 后计算（身体 23 个关节）。

| 测试 | 结果 |
|---|---|
| 前向与 `SOMALayer.pose` 一致性（随机体型＋姿态） | 最大顶点差 0.001 mm |
| 自动标定后新动作 MPJPE（low/T 种子 0、1；low/A；mid/T） | 9.0 / 2.3 / 6.7 / 7.1 mm |
| 命令行：`calibrate.py` → `run_realtime.py --replay` → 保存 / UDP | 1485/1500 帧，MPJPE 2.5 mm，UDP 与保存文件逐位一致 |
| 单帧求解（2 核 Xeon，numba） | 中位数 2.3–2.8 ms |
| 标定耗时（low，单独运行） | 约 85 s；mid 网格更慢 |
| 中精度显示网格 | 0.9 ms/帧 |

种子 0 的 low/T 那次只用了 41 个点（候选顶点阈值较严），误差偏大；与 SMPL 测试一样，这些数字只反映算法本身。

## 范围与限制

- 拟合模型是 SOMA 的 **78 关节 LBS rig**（`enable_procedural_transforms=False`）。SOMA 默认的程序化扭转关节和学习的
  姿态矫正（MLP）不在拟合模型里；用默认配置的 `SOMALayer` 渲染同一组参数时，前臂扭转分布和体表矫正会略有不同。
- 当前 SOMA 路径只拟合**身体与手腕**（配合 `--body-regions`）。手指关节没有 marker 时保持绑定姿态；
  SMPL-X 路径里的手指 PCA 先验属于 MANO 关节体系，不能直接用到 SOMA 的手指上，所以网页的 `--fit-hands`
  仍只支持 SMPL-X。
- 求解器的弱姿态先验朝向"零旋转"，对 SOMA 来说就是它的绑定 A-pose（SMPL 是 T-pose）。
- `--soma-identity smpl|smplx` 会通过 SOMA 使用你们授权的 SMPL/SMPL-X 体型系数作为身份；这条路径依赖授权模型文件，
  仓库测试没有覆盖。
