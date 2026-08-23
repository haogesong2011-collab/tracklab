"""Render experiment reports locally. DeepSeek only supplies prose sections."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ai.contracts import (
    EXPERIMENT_LABELS,
    ExperimentAnalysis,
    ExperimentCandidate,
    ExperimentType,
    TeachingLevel,
)
from ai.experiment_catalog import formulas_for

SECTION_KEYS = ("purpose", "equipment", "principle", "errors", "conclusion")


def render_report_markdown(
    analysis: ExperimentAnalysis | None,
    *,
    confirmed_type: ExperimentType | None,
    sections: dict[str, str] | None = None,
    teaching_level: TeachingLevel = TeachingLevel.HIGH,
    generated_at: str | None = None,
    model_id: str = "",
    stale: bool = False,
) -> str:
    candidate = _pick(analysis, confirmed_type)
    label = EXPERIMENT_LABELS.get(confirmed_type or ExperimentType.UNKNOWN, "未确认")
    if candidate is not None:
        label = candidate.label
    stamp = generated_at or datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# {label}实验报告",
        "",
        f"- 生成时间：{stamp}",
        f"- 教学级别：{teaching_level.value}",
    ]
    if model_id:
        lines.append(f"- 讲解模型：{model_id}")
    if stale:
        lines.append("- 状态：**数据已变化，以下为旧结果**")
    lines.append("")
    prose = sections or {}
    lines.extend(_section("一、实验目的", prose.get("purpose", "（未生成文字说明）")))
    lines.extend(_section("二、器材与数据条件", _equipment_block(analysis, candidate, prose.get("equipment"))))
    lines.extend(_section("三、原理", _principle_block(confirmed_type, candidate, prose.get("principle"))))
    lines.extend(_section("四、公式与结果", _results_block(candidate, analysis)))
    lines.extend(_section("五、误差与注意", prose.get("errors", "以本地拟合残差与缺失条件为准。")))
    lines.extend(_section("六、结论", prose.get("conclusion", "（未生成文字说明）")))
    lines.append("")
    lines.append("数值全部来自 TrackLab 本地拟合；模型文字只作解释，不作为原始数据。")
    lines.append("")
    return "\n".join(lines)


def merge_report_sections(raw: dict[str, Any] | None) -> dict[str, str]:
    data = raw or {}
    return {key: str(data.get(key) or "").strip() for key in SECTION_KEYS}


def _pick(
    analysis: ExperimentAnalysis | None, confirmed_type: ExperimentType | None
) -> ExperimentCandidate | None:
    if analysis is None:
        return None
    if confirmed_type is not None:
        for item in analysis.candidates:
            if item.experiment_type is confirmed_type:
                return item
    return analysis.selected


def _section(title: str, body: str) -> list[str]:
    text = (body or "").strip() or "（无）"
    return [f"## {title}", "", text, ""]


def _equipment_block(
    analysis: ExperimentAnalysis | None,
    candidate: ExperimentCandidate | None,
    prose: str | None,
) -> str:
    bits = []
    if analysis is not None:
        bits.append(
            f"标定：{'已启用，单位 ' + analysis.position_unit if analysis.calibration_active else '未标定，单位 px'}"
        )
        bits.append(f"轨迹覆盖率：{analysis.coverage:.3f}")
        bits.append(f"平均跟踪置信度：{analysis.mean_track_confidence:.3f}")
        if analysis.missing:
            bits.append("缺失条件：" + "、".join(analysis.missing))
    if candidate is not None:
        bits.append(
            f"拟合区间：第 {candidate.fit.frame_start}–{candidate.fit.frame_end} 帧，"
            f"{candidate.fit.time_start_s:.3f}–{candidate.fit.time_end_s:.3f} s，"
            f"{candidate.fit.n_samples} 个样本"
        )
    local = "\n".join(f"- {item}" for item in bits) if bits else "- （无本地条件）"
    extra = (prose or "").strip()
    if extra:
        return local + "\n\n" + extra
    return local


def _principle_block(
    confirmed_type: ExperimentType | None,
    candidate: ExperimentCandidate | None,
    prose: str | None,
) -> str:
    kind = confirmed_type or (None if candidate is None else candidate.experiment_type)
    formulas = formulas_for(kind) if kind else []
    lines = []
    for entry in formulas:
        lines.append(f"- `{entry['expression']}`（{entry['name']}，{entry['id']}）")
        cond = "；".join(entry.get("conditions") or [])
        if cond:
            lines.append(f"  适用条件：{cond}")
    local = "\n".join(lines) if lines else "- （尚未确认实验类型，无标准公式）"
    extra = (prose or "").strip()
    if extra:
        return local + "\n\n" + extra
    return local


def _results_block(
    candidate: ExperimentCandidate | None, analysis: ExperimentAnalysis | None
) -> str:
    if candidate is None:
        return "无法计算：尚未确认实验类型或本地拟合不可用。"
    rows = [
        f"- 实验类型：{candidate.label}",
        f"- 本地置信度：{candidate.confidence:.3f}",
        f"- R²：{candidate.fit.r2:.4f}",
        f"- 归一化 RMSE：{candidate.fit.nrmse:.4f}",
    ]
    for name, value in candidate.fit.parameters.items():
        unit = candidate.fit.units.get(name, "")
        if value is None:
            rows.append(f"- {name}：无法计算")
        else:
            suffix = f" {unit}" if unit else ""
            rows.append(f"- {name} = {value:.6g}{suffix}")
    if analysis and analysis.warnings:
        rows.append("- 警告：" + "；".join(analysis.warnings))
    return "\n".join(rows)
