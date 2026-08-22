# TrackLab AI 测试

分层方式：

1. **单元测试** `tests/ai/unit/` — 指标公式、坐标变换、拒识逻辑，不读视频。
2. **集成测试** `tests/ai/integration/` — 通过 `engine.decoder` 取帧，检查帧号对齐、取消、标定拒识。不依赖界面事件循环（桌面验收除外）。
3. **CI 评测** `python -m tests.ai.evaluate --split ci` — 10 段冻结合成视频，目标 <2 分钟。`python -m tests.ai.ci` 一次跑完单元、集成、offscreen 桌面验收和 CI 评测（不下载 SAM 权重）。
4. **完整回归** `python -m tests.ai.regression --split all` — 60 槽 manifest（30 跟踪 / 15 标定 / 15 姿态）。holdout 约 20%，选型期间不要对着它调参。
5. **桌面验收** `QT_QPA_PLATFORM=offscreen python -m unittest tests.ai.integration.test_desktop_acceptance` — AI 在独立 `QThread` 中运行，不得占用 `FramePump`。
6. **SAM 2 模型验收**（不进快速 CI）`python -m tests.ai.evaluate --split ci --model sam2` — 需要 `requirements-ai.txt` 与本地权重。

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
