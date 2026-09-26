# TrackLab AI 测试

分层方式：

1. **单元测试** `tests/ai/unit/` — 指标公式、坐标变换、拒识逻辑，不读视频。
2. **集成测试** `tests/ai/integration/` — 通过 `engine.decoder` 取帧，检查帧号对齐、取消、标定拒识。不依赖界面事件循环（桌面验收除外）。
3. **CI 评测** `python -m tests.ai.evaluate --split ci` — 10 段冻结合成视频，目标 <2 分钟。`python -m tests.ai.ci` 一次跑完单元、集成、offscreen 桌面验收和 CI 评测（不下载 SAM 权重）。
4. **完整回归** `python -m tests.ai.regression --split all` — 60 槽 manifest（30 跟踪 / 15 标定 / 15 姿态）。holdout 约 20%，选型期间不要对着它调参。
5. **桌面验收** `QT_QPA_PLATFORM=offscreen python -m unittest tests.ai.integration.test_desktop_acceptance` — AI 在独立 `QThread` 中运行，不得占用 `FramePump`。
6. **SAM 2 模型验收**（不进快速 CI）`python -m tests.ai.evaluate --split ci --model sam2` — 需要 `requirements-ai.txt` 与本地权重。
7. **真实模型追踪评测** `python -m tests.ai.benchmark_tracking` — 桌面端同一套模型选择、预处理和结果生成。见下文。

桌面正式跟踪器是 SAM 2.1 Tiny（Apache 2.0）。色块跟踪只作为合成基线和无权重夹具，不能代表真实视频精度。

模型只输出 `ai/contracts.py` 中的类型。Oracle 适配器用来验证评测器本身，必须 100% 通过门槛。Baseline（色块跟踪 / 模板姿态 / 端点标定）用于冻结对照，允许低于门槛，但不得相对已提交基线回退。

```bash
python -m tests.ai.generate_fixtures
python -m unittest discover -s tests/ai -v
python -m tests.ai.evaluate --split ci --model oracle --save-baseline
python -m tests.ai.evaluate --split ci --model baseline --save-baseline
python -m tests.ai.regression --split all --model baseline --include-holdout --save-baseline
```

核心指标相对基线下降超过 2%、或耗时 ≥100 ms 的片段变慢超过 10%、或新增失败，评测以退出码 2 阻止合并。

## 真实模型追踪评测（benchmark_tracking）

```bash
# 物体中心（SAM 2.1 Small），三个变体
.venv/bin/python -m tests.ai.benchmark_tracking --target object_center --variant candidate --device cpu --output-dir bench/oc-candidate
.venv/bin/python -m tests.ai.benchmark_tracking --target object_center --variant raw       --device cpu --output-dir bench/oc-raw
.venv/bin/python -m tests.ai.benchmark_tracking --target object_center --variant current   --device cpu --output-dir bench/oc-current \
    --baseline-root ~/Downloads/tracklab-baseline-0925     # 解压修改前的工作区快照

# 指定表面点（BootsTAPIR）
.venv/bin/python -m tests.ai.benchmark_tracking --target surface_point --variant candidate --device cpu --output-dir bench/sp-candidate

# 汇总成表
.venv/bin/python -m tests.ai.benchmark_tracking compare bench/*/report.json
```

- 内置片段是冻结的合成用例：计划里的小目标回归（种子 77、12 帧、960×540、8px 纹理、16px 条纹、5×5 高斯 σ=0.9、每帧平移 (7,2)），以及 4/16/32px、遮挡后重现、急转、跨窗口、竖屏、1080p。`--split holdout` 用事先固定的种子 1077/2077/3077，调参期间不要看。
- `--manifest` 可以加真实片段；同一 SHA-256 的视频出现在不同划分会直接报错退出（`--allow-split-leak` 才放行）。
- 报告分开写「候选坐标准确率」（所有有位置的点）和「被接受测量的准确率」（`usable_for_measurement()`），没有高分样本时高分错误率写 `N/A`。
- 报告里记录模型、权重实际 SHA-256、SAM 2 / TAPNet 安装提交、实际设备、输入尺寸、提示帧与坐标。
- `--fake-model` 只检查管线（SAM 用真值掩膜、TAPIR 用 NCC 匹配），不代表模型精度。
- `--overlay` 输出叠加视频：绿色真值，蓝色可信，黄色待复核，红色丢失。
