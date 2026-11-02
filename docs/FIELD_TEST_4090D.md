# 4090D 现场测试 · 2026-11-02

- 计算与网页：HKU-4090D，`http://192.168.110.211:8792`。
- 代码：`/home/wangboran/projects/Vicon2Smpl`；独立 `.venv` 继承已有 mamma 环境，只在自身目录安装新增依赖。
- 输入：动捕电脑 `192.168.110.125:8802`，通过官方 DataStream C++ SDK 只读连接。
- 模型：`/data2/DEV/soma/support_files/smplx/neutral/model.npz`，既有 SMPL-X 几何模型；原文件未改动。
  直接加载其模板、形变基、关节回归器、蒙皮权重和骨骼树，不需要该文件缺少的手部 PCA 元数据。
- 运算使用 CPU，线程数限制为 1；不占用现有任务的 GPU。
- 输出：`/home/wangboran/projects/Vicon2Smpl/runtime`；每次采样独立编号，正式录制文件不变。

## 现场操作

1. 打开网页，确认人物名称与 Shogun 中实际被试一致、全身 marker 可见。
2. 点击“开始标定采样”。5 秒准备后，T-pose 保持 3 秒，再做活动范围动作；默认采样 45 秒。
3. 采样结束，点击“计算本次标定”。计算进度在本次目录的 `calibration.log`；完成后检查残差和朝向。
4. 点击“开始实时拟合”，对照人体、红色观测 marker、绿色拟合 marker；检查蹲起、转身、抬臂与手指。
5. 点击“停止拟合并保存”。该操作仅保存测试拟合，不控制正式录制。

换人、重新贴点后重新标定。手部点稀疏、点位有误或遮挡时，手指可能不可观；不得仅凭残差接受手指结果。
标定的每点偏移使用仓库既有统一半径先验，未引入按身体/手指区分的先验。

## 已验证与待验证

已验证：现场 `st` 标签集含 72 个 marker，8802 上收到 72 个有效点、300 Hz；801 同次探测返回这些点被遮挡。
浏览器实际显示当前点位，网页连接和居中控件通过人工界面检查。
新适配器/原生 NPZ 模型读取/采样保留与拟合参数保存测试 4 项通过；上游核心测试通过。
核心测试合成人体 oracle MPJPE 为 1.6 mm，求解中位数 0.98 ms；这不是现场精度。

待验证：真实被试 T-pose/ROM 标定、真实全身与手指拟合、实际连续运动效果。服务初始只做点位预览。

## 启动与日志

正式录制和测试采样结束后才维护测试服务。当前进程号记录在 `web.pid`；不要启动第二份实例。
启动参数如下（在已检查的代码目录中执行）：

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u run_web.py \
  --source 192.168.110.125:8802 --native-reader ./native/live_markers \
  --model-path /data2/DEV/soma/support_files/smplx/neutral/model.npz \
  --model-type smplx --bind 192.168.110.211 --port 8792 --output ./runtime
```

`web.log` 为服务日志，`runtime/status.json` 为当前状态。外网 Mac 已有单一 SSH master 的
`127.0.0.1:18792 → 192.168.110.211:8792` 转发；不需要再建 SSH 连接。
现场网页不对公网开放。代码与 SDK 分开；SDK 和模型均不提交 Git。
