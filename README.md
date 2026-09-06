# TrackLab

物理视频分析工具（Tracker 工作流的重做）。当前版本是 Tracker 风格的 SAM 2 跟踪工作台：导入视频、多轨点选/框选目标、异步传播掩膜质心轨迹。快速模式用 Tiny 隔帧插值，精准模式用 Small 逐帧。另有平面测量、GitHub 更新检查，以及可选的 macOS DMG。

## 运行

```bash
cd ~/Projects/tracklab
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt
# 桌面 SAM 2 跟踪（可选）：.venv/bin/python3 -m pip install -r requirements-ai.txt
.venv/bin/python3 -m app
# 或：./run.sh
```

不要用系统的 `python3 -m app`。有的 Mac 上 `python3` 始终指向 `/Library/Frameworks/Python.framework/...`，虚拟环境里的包它看不见。

打开窗口后，把 mp4 / mov 等文件拖进去，或点「打开」/ 点击中央区域。打开是秒开的，进度条一格一帧。

- 空格：播放 / 暂停（逐帧显示，不跳帧）
- T：SAM 2 自动跟踪当前轨迹（独立线程）；再按一次取消
- 新建轨迹：右侧列表可管理多条轨迹
- 点击画面：正点选目标；Shift+点击为负点；拖动为框选
- 跟踪完成后点击即手工修正当前帧，可再按 T 从该帧重跟踪
- 文件 → 打开/保存项目：多轨 JSON（含提示与修正）
- 文件 → 导出轨迹 JSON / CSV
- 底栏数据表：点击行跳到对应帧
- 拖动进度条：立即暂停，画面跟着指针走，松手后停在该帧
- 左右方向键：按底栏选择的步长前进 / 后退（1–5 帧）
- 底栏最左的下拉框：25% – 200% 播放速度
- 进度条上方两个三角：循环区间的起点和终点，可直接拖动；I / O 把起点、终点设到当前帧
- 底栏最右的循环按钮：在这个区间内反复播放
- 文件 → 关闭视频：关掉解码器并释放缓存

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
TRACKLAB_SKIP_UPDATE_CHECK=1 python -m tests.ai.ci
```

查看版本：`.venv/bin/python3 -m app --version`

macOS 安装包由 `.github/workflows/release-macos.yml` 在打 `v*` tag 时构建。本地打包见 `macos-packaging/build_macos.sh`（捆绑 Tiny 权重；精准 Small 首次使用时下载）。

数据约定见 [datasets/README.md](datasets/README.md)。Holdout 槽位（约 20%）选型期间禁止调参。
