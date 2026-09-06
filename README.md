# TrackLab

物理视频分析工具（Tracker 工作流的重做）。自动跟踪分为快速和精准：快速模式采用 Tracker Autotracker 风格的模板匹配，精准模式采用 SAM 2.1 Tiny 视频掩膜传播。

## 运行

```bash
cd ~/Projects/tracklab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# 精准模式（可选）：python -m pip install -r requirements-ai.txt
python -m app
```

打开窗口后，把 mp4 / mov 等文件拖进去，或点「打开」/ 点击中央区域。打开是秒开的，进度条一格一帧。

- 空格：播放 / 暂停（逐帧显示，不跳帧）
- T：按当前“快速 / 精准”模式自动跟踪；再按一次取消
- 新建轨迹：右侧列表可管理多条轨迹
- 点击画面：正点选目标；Shift+点击为负点；拖动为框选
- 跟踪完成后点击即手工修正当前帧，可再按 T 从该帧重跟踪
- 文件 → 打开/保存项目：多轨 JSON（含提示与修正）
- 文件 → 导出轨迹 JSON / CSV
- 底栏数据表：点击行跳到对应帧
- 拖动进度条：立即暂停，画面跟着指针走，松手后停在该帧
- 左右方向键：按底栏选择的步长前进 / 后退（1–5 帧）
- 底栏最左的下拉框：25% – 200% 播放速度
- 进度条上方两个三角：循环区间的起点和终点，拖动时画面同步预览；I / O 把起点、终点设到当前帧
- 底栏最右的循环按钮：在这个区间内反复播放
- 文件 → 关闭视频：关掉解码器并释放缓存

## macOS 安装包

不需要安装 Python。从 [GitHub Releases](https://github.com/haogesong2011-collab/tracklab/releases/latest) 下载对应芯片的 DMG：

- Apple 芯片（M 系列）：`TrackLab-arm64.dmg`
- Intel 芯片：`TrackLab-x86_64.dmg`

打开 DMG，把 `TrackLab` 拖到“应用程序”。当前版本未经 Apple 公证：请在 Finder 中 **右键 TrackLab → 打开**，并在提示中确认打开。之后可以像普通应用一样双击启动。

安装包包含快速跟踪、精准跟踪（SAM 2）和手工平面测量。AI 离面抽检未打包；需要时请从源码安装。精准模式首次使用已内置权重，无需再下载。

### 更新

应用启动后会异步检查 GitHub Releases（默认最多每 24 小时一次）。发现新版本时会显示版本号和说明，并提供“前往下载”。更新不会在运行中替换 `.app`。也可在“帮助 → 检查更新…”里手动检查，或关闭“启动时自动检查更新”。

## 取帧方式

不做导入期转码。打开文件时只 demux 一遍包头建立帧索引（60 秒 1080p 约 10 ms），播放和跳帧都靠 `engine/decoder.py` 现解：

- 顺序取下一帧 ≈ 2 ms，随机跳到任意一帧 ≈ 14 ms（关键帧 seek 后前滚）
- 解码跑在 `app/frame_pump.py` 的后台线程上，只服务最新的一次请求，拖动时不会堆积
- 帧数据从解码器直达屏幕，不经过任何再压缩
- 内存里只留最近约 128 MB 的已解码帧，长视频不会撑爆内存

`engine/video_index.py` 的索引按显示顺序（PTS 排序）建立，所以带 B 帧的视频也能按序号精确取帧。

## AI 评测

AI 模块与 UI 解耦：模型只输出 `ai/contracts.py` 里的结果类型，评测不依赖 Qt。

```bash
# 生成 / 刷新 10 段合成 CI 视频 + 60 槽 manifest
python -m tests.ai.generate_fixtures

# 单元测试（指标公式、拒识逻辑）
python -m unittest tests.ai.unit.test_metrics -v

# 集成测试（解码器 + 模型，无 UI）
python -m unittest tests.ai.integration.test_pipeline -v

# CI 评测（oracle 应全过；baseline 建立对照基线）
python -m tests.ai.evaluate --split ci --model oracle
python -m tests.ai.evaluate --split ci --model baseline --save-baseline

# 完整回归（60 槽，含 holdout；选型期间不要对着 holdout 调参）
python -m tests.ai.regression --split all --model baseline --include-holdout --save-baseline

# 桌面端验收（取消延迟 / 手工修正保存；需要 Qt，可用 offscreen）
QT_QPA_PLATFORM=offscreen python -m unittest tests.ai.integration.test_desktop_acceptance -v

# SAM 2.1 Tiny（正式跟踪器；需 pip install -r requirements-ai.txt，首次使用下载权重）
python -m tests.ai.evaluate --split ci --model sam2
```

提交前可跑小型 CI（目标 <2 分钟）：

```bash
python -m tests.ai.ci
```

数据约定见 [datasets/README.md](datasets/README.md)。Holdout 槽位（约 20%）选型期间禁止调参。

## 发布 macOS 安装包

版本号只写在 [`app/__init__.py`](app/__init__.py)，关于对话框和更新检查共用该值。发布步骤：

1. 把 `__version__` 改成新的 `主版本.次版本.修订号`，并在 [`CHANGELOG.md`](CHANGELOG.md) 增加对应章节。
2. 合并到默认分支后，打并推送同版本 tag，例如 `git tag v0.1.0 && git push origin v0.1.0`。
3. GitHub Actions 会先跑测试，再分别在 Apple 芯片与 Intel runner 上构建 `TrackLab-arm64.dmg` 与 `TrackLab-x86_64.dmg`，最后创建 GitHub Release 并上传两个 DMG 和 `SHA256SUMS.txt`。
4. 客户端随后即可通过 `releases/latest` 检测到新版本。

手动运行 **Release macOS** workflow 只上传 Actions artifact，不会创建正式 Release，避免把试构建当成最新版。

本地打包（当前机器架构）：

```bash
python -m pip install -r requirements.txt -r macos-packaging/requirements-build.txt
bash macos-packaging/build_macos.sh
QT_QPA_PLATFORM=offscreen dist/TrackLab.app/Contents/MacOS/TrackLab --smoke
```

