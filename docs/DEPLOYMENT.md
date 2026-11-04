# Linux 共享部署与离线依赖包

本页对应 `run_web.py` 的原生 SMPL-X 身体、手腕和手指拟合。网页读取 Vicon DataStream，
不会调用 Shogun 开拍/停拍，也不修改正式录制。旧的 `aitviewer`、Python Vicon SDK 入口为可选路径，
不属于本页验证的 Linux 网页依赖闭包。

## 共享内容

- GitHub：全部项目源码、测试、配置模板、标定和运行入口、依赖导出/离线安装/打包/健康检查工具。
- 服务器内部包：Python、精确版本依赖与 wheelhouse、Vicon 原生桥接器和 SDK、授权 SMPL-X 模型、
  手形先验、核对过的 marker 区域配置、哈希清单。
- 个人标定、marker 采样、拟合结果和历史报告保留在服务器的独立输出目录，不进入公开 Git。
- Python 包和 SDK 原文件仍受各自许可证约束；模型按原授权范围使用，不能上传公开仓库。

同一网页是一个共用现场控制台：由一名指定操作者开始/停止采样和拟合，其余人观看。
控件没有按访客隔离；近景和居中操作也会影响其他访问者的视角。
开放 GitHub 代码读取不代表授予 GitHub 写入权限。

## 准备精确依赖

从当前验证过的 Python 环境运行，不根据未锁版本的 requirements 文件猜测版本：

```sh
<当前环境>/bin/python scripts/export_environment.py --output <新的环境包目录>
```

导出包含 Python/平台信息、精确 requirements、依赖 wheel 文件及 SHA256。
如先用 `--metadata-only` 检查环境，继续同一个未完成目录时加 `--resume`；已完整的包不覆盖。
如果 Torch 使用 `+cpu` 等本地版本，提供对应官方索引；导出失败或仅有元数据的目录不能当成完整离线包。
命令和报告不要包含带密码的索引地址。`environment.json` 和 `manifest.json` 保存版本及完成状态。

Python 需要完整、可搬移的独立安装，不要只复制一个链接到系统解释器的 venv。
可以使用已核对来源的 python-build-standalone/uv 管理的 Linux Python；记录来源和哈希，
版本、架构和 ABI 必须与 wheelhouse 匹配。Linux/glibc、ELF 加载器、系统 C/C++ 库仍是目标操作系统要求，
由健康检查的 `ldd` 结果核对，不声称能在任意操作系统直接运行。

## 构建内部包

下面都是占位路径。先确认 SDK、Python、模型、先验和区域配置的实际目录；输出必须是全新目录。
source 必须是已提交且干净的 Git checkout，避免漏掉现场代码。

```sh
python3 scripts/package_runtime.py \
  --source <Git-checkout> --environment <已完整导出的环境包> \
  --python-home <独立Python目录> \
  --sdk-root <已安装ViconSDK根目录> --sdk-library-dir <SDK动态库目录> \
  --native-reader <已编译live_markers> \
  --model <SMPL-X-neutral-model.npz> --hand-prior <pose_hand_prior.npz> \
  --body-regions <已核对52点区域.json> \
  --vicon-source <Shogun-IP>:8802 --bind <实验室IP> --port 8792 \
  --output <新的内部包目录>
```

包内结构：

```text
<bundle>/
  code/                  Git 指定提交的全部源码
  python/                独立 Python
  environment/           锁定依赖、wheelhouse、版本和哈希
  vendor/vicon/          原 SDK 文件和构建所需头文件/动态库
  native/live_markers    已编译的只读 DataStream 桥接器
  assets/                授权模型、先验、区域 JSON
  config/site.json       相对于本包的现场配置，不预装他人的个人标定
  runtime/               本实例热状态、锁和日志
  data/                  新标定采样和拟合输出，可与热状态分开配置
  numba_cache/           本实例可写缓存
  restore.sh             不联网重建 venv
  SOURCE_COMMIT          源码提交
  bundle-manifest.json   内部文件哈希；安装/实机验收另外记录
```

## 全新离线安装验证

在新的包目录执行 `./restore.sh`。它从 wheelhouse 重建新的 venv，禁止连接包索引，随后执行
`pip check` 与核心模块导入。已有环境能运行不能代替这一步；有文件清单也不能代替安装通过。
执行前还会核对整个内部包的 SHA256 与内部符号链接；安装回执另外生成，不改变原依赖清单。

```sh
./restore.sh
./venv/bin/python code/scripts/healthcheck.py --config config/site.json \
  --report runtime/healthcheck.json
./venv/bin/python -m pytest -q code/tests/test_native_source.py \
  code/tests/test_full_web_worker.py code/tests/test_head_calibration.py
```

健康检查不连接硬件、不启动第二个 DataStream 客户端。它核对导入、pip 依赖、SDK 动态链接、
模型/先验维数、标定身份及输出可写性。`--skip-model` 只适用于排查，结果为 `PASS_WITH_SKIPS`，不能当成完整部署验收。
报告使用新文件名，不覆盖旧检查记录。打包成功、全新安装成功和实际拟合检查是三个独立结果。

## 启动与访问

把配置中的 IP、输出目录和依赖路径改成已检查的现场值。路径可以相对配置文件。
服务默认只加载通用模型；新被试通过页面重新标定。只有同一被试、同一套贴点，才在本地配置添加
`calib` 和 `subject`，两项必须一起填写，校验不能绕过。

`output` 放热状态、锁和日志，`data_output` 放新采样和标定结果；省略后者时保持旧行为。
共享存储上小文件操作较慢时，热状态与 Numba 缓存应留在本地快目录，数据输出再指向共享存储。
已加载标定的拟合结果仍保留在其原父目录，不会因为新增参数改写历史关联。

```sh
./venv/bin/python code/scripts/service.py --config config/site.json preflight
./venv/bin/python code/scripts/service.py --config config/site.json start
./venv/bin/python code/scripts/service.py --config config/site.json status
```

启动工具对同一 output 持有 Linux 文件锁，并在建立硬件连接前检查端口。
重复启动不会再创建一份采样进程。它只管理自己启动并核对归属的进程，不会自动停止其他人的任务。
维护时所有人必须使用同一配置和 output；不同配置/端口不能用来绕过现场单实例约定。

实验室成员打开 `http://<4090D实验室IP>:8792/`。实验室外的成员使用自己已有的 SSH 权限转发，
复用每台主机的一个持久 SSH 连接；不公开无认证的控制端口，不复制别人的私钥。
访客共享控制状态，网页输入源的 300 Hz 不代表求解或浏览器渲染为 300 FPS。

## 保存与停止

先在网页点击“停止拟合并保存”，确认实际保存帧数和文件可读，再停止服务：

```sh
./venv/bin/python code/scripts/service.py --config config/site.json stop
```

启动工具默认拒绝停止正在采样、标定或拟合的服务。停止使用 SIGINT，保留正常收尾，不使用 kill -9。
现有程序把拟合结果累积在内存，停止拟合时写出 NPZ，没有断电恢复保证，应按短段停止并确认保存。
加载既有标定时，拟合输出写在该标定的父目录，`output` 接收运行状态；现场配置必须确认两处均可写。
不要覆盖原始 marker、旧标定或既有输出。磁盘余量、保存是否成功与拟合精度应分别检查。

## 更新和回退

更新前从正式录制服务查询状态：演员录制/准备/后台保存、台本 worker 均结束后才能切换部署。
先保存当前派生拟合，再停本项目服务；保留旧包、个人标定、历史输出和提交哈希。
新包完成离线安装、健康检查和源码哈希检查后，只启动一次。回退仅恢复代码/环境/启动配置，
不得用旧 runtime 覆盖更新期间新增的数据。

共享代码和依赖使用只读权限；服务状态、缓存和输出只允许指定操作者写入。不要对整个 home、SDK、
模型目录或历史数据执行递归 `chmod 777`。服务器最终路径、版本、依赖清单和验证结果应写入独立部署回执。

## 固定服务器的快速缓存恢复

系统盘空间不足、共享盘小文件访问较慢时，可以把已经离线安装验证的完整运行时归档到持久存储，
用 `scripts/launch_cached.py --config <启动配置.json>` 校验并恢复到指定用户的快速缓存目录。
该归档必须在同一个 `cache_root` 前缀构建，不能冒充可任意迁移的 venv；通用迁移仍使用上文的独立 Python 与 wheelhouse。
缓存只包含 Python、环境、代码、原生库和启动配置；个人采样、标定和拟合输出始终写持久数据目录。

启动配置包含 `archive`、`archive_sha256`、`cache_root`，以及可选 `service_config`（默认 `config/site.json`）。
首次启动核对归档哈希并安全解压，再生成 READY；后续复用匹配的完整缓存。缺少 READY、版本不符或损坏的
已有缓存会保留并拒绝覆盖，由操作者核对后另行处理。准备阶段使用独立文件锁，运行阶段继续使用服务单例锁。
`--prepare-only` 只恢复/检查，不连接硬件。

若缓存位于 `/run/user/<UID>`，它会在重启后消失；持久归档和数据不受影响。
用户服务可以在启动时重建缓存。部署回执应分别记录持久包、缓存恢复、离线安装、实际服务和输入源检查结果。
`config/vicon2smpl.service.example` 提供用户服务模板；先核对路径和用户 linger 状态。
它只启动预览控制台，不自动开始标定采样、拟合或演员录制。维护时仍先从网页停止并保存，再停止用户服务。
