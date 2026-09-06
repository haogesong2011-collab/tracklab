"""Build the structured teaching JSON sent to DeepSeek. No raw video, no keys."""

from __future__ import annotations

import json
from typing import Any

from ai.contracts import (
    TEACHING_LEVEL_LABELS,
    ChatMessage,
    ExperimentAnalysis,
    ExperimentCandidate,
    ExperimentType,
    TeachingLevel,
)
from ai.experiment_catalog import catalog_payload, formulas_for
from ai.kinematics import KinematicSample

MAX_REPRESENTATIVE_POINTS = 24
MAX_HISTORY_MESSAGES = 12

SYSTEM_PROMPT = """你是 TrackLab 的物理实验助教。必须遵守：
1. 只能引用用户消息 JSON 上下文中的实测值、拟合参数和公式目录。
2. 不得发明公式、符号或数值；不得把本地候选识别说成已确认事实。
3. 缺失的量必须明确写「无法计算」，不要用 9.8 或其他常识值填空。
4. 区分「实验测量值」「理论值」和「模型假设」。
5. 讲解每个公式时说明符号、单位和适用条件。
6. 不要提及视频文件路径、API Key、操作系统或内部实现细节。
7. 用中文回答，语气对应教学级别。
"""

REPORT_JSON_INSTRUCTION = """请根据上下文写一份教学报告的文字段落，只输出 JSON 对象，例如：
{"purpose":"……","equipment":"……","principle":"……","errors":"……","conclusion":"……"}
不要改写 context.results 里的数字，不要输出 Markdown，不要包裹代码围栏。
"""


def representative_points(samples: list[KinematicSample], limit: int = MAX_REPRESENTATIVE_POINTS) -> list[dict[str, Any]]:
    visible = [
        item
        for item in samples
        if item.visible and item.x is not None and item.y is not None
    ]
    if not visible:
        return []
    if len(visible) <= limit:
        return [_point_payload(item) for item in visible]
    chosen: dict[int, KinematicSample] = {}
    for item in (visible[0], visible[-1]):
        chosen[item.frame] = item
    xs = [item.x or 0.0 for item in visible]
    ys = [item.y or 0.0 for item in visible]
    for idx in (
        int(min(range(len(visible)), key=lambda i: xs[i])),
        int(max(range(len(visible)), key=lambda i: xs[i])),
        int(min(range(len(visible)), key=lambda i: ys[i])),
        int(max(range(len(visible)), key=lambda i: ys[i])),
    ):
        chosen[visible[idx].frame] = visible[idx]
    if len(chosen) < limit:
        stride = max(1, len(visible) // (limit - len(chosen)))
        for item in visible[::stride]:
            chosen[item.frame] = item
            if len(chosen) >= limit:
                break
    ordered = sorted(chosen.values(), key=lambda item: item.frame)[:limit]
    return [_point_payload(item) for item in ordered]


def build_teaching_context(
    analysis: ExperimentAnalysis | None,
    *,
    confirmed_type: ExperimentType | None,
    samples: list[KinematicSample] | None = None,
    teaching_level: TeachingLevel = TeachingLevel.HIGH,
    pendulum_length_m: float | None = None,
    stale: bool = False,
    interpolated: bool = False,
) -> dict[str, Any]:
    candidate = _confirmed_candidate(analysis, confirmed_type)
    formulas = formulas_for(confirmed_type) if confirmed_type else catalog_payload()["formulas"]
    results = _result_payload(candidate, analysis)
    warnings = list(analysis.warnings) if analysis else []
    missing = list(analysis.missing) if analysis else ["analysis"]
    if stale:
        warnings.append("数据已变化，以下结果可能过期")
    if interpolated:
        warnings.append("轨迹含 Tiny 插值，测 g 请改精准（Small）后重跟")
    payload = {
        "role": "tracklab_teaching_context",
        "teaching_level": teaching_level.value,
        "teaching_level_label": TEACHING_LEVEL_LABELS[teaching_level],
        "confirmed": confirmed_type is not None and confirmed_type is not ExperimentType.UNKNOWN,
        "confirmed_type": None if confirmed_type is None else confirmed_type.value,
        "stale": stale,
        "catalog": {"formulas": formulas} if confirmed_type else catalog_payload(),
        "results": results,
        "missing": missing,
        "warnings": warnings,
        "representative_points": representative_points(samples or []),
        "pendulum_length_m": pendulum_length_m,
        "notes": [
            "representative_points 只含压缩后的拐点，不是逐帧轨迹",
            "数值由本地拟合给出，模型不得改写",
        ],
    }
    return payload


def chat_messages(
    context: dict[str, Any],
    history: list[ChatMessage],
    user_text: str,
    *,
    teaching_level: TeachingLevel = TeachingLevel.HIGH,
) -> list[dict[str, str]]:
    level = TEACHING_LEVEL_LABELS[teaching_level]
    user_blob = {
        "teaching_level": level,
        "question": user_text,
        "context": context,
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    for item in history[-MAX_HISTORY_MESSAGES:]:
        if item.role in {"user", "assistant"} and item.content.strip():
            messages.append({"role": item.role, "content": item.content})
    messages.append(
        {
            "role": "user",
            "content": json.dumps(user_blob, ensure_ascii=False, indent=2),
        }
    )
    return messages


def report_messages(
    context: dict[str, Any],
    *,
    teaching_level: TeachingLevel = TeachingLevel.HIGH,
) -> list[dict[str, str]]:
    blob = {
        "teaching_level": TEACHING_LEVEL_LABELS[teaching_level],
        "task": "experiment_report_sections",
        "json": True,
        "context": context,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT + "\n" + REPORT_JSON_INSTRUCTION},
        {"role": "user", "content": json.dumps(blob, ensure_ascii=False, indent=2)},
    ]


def preview_payload(context: dict[str, Any]) -> str:
    return json.dumps(context, ensure_ascii=False, indent=2)


def assert_private_context(context: dict[str, Any]) -> None:
    text = json.dumps(context, ensure_ascii=False).lower()
    forbidden = ("api_key", "authorization", "deepseek_api_key", "sk-")
    for token in forbidden:
        if token in text:
            raise ValueError(f"teaching context leaked secret field: {token}")
    if "video_path" in context or "path" in context:
        raise ValueError("teaching context must not include file paths")


def _point_payload(sample: KinematicSample) -> dict[str, Any]:
    return {
        "frame": sample.frame,
        "t": round(sample.time_s, 4),
        "x": None if sample.x is None else round(sample.x, 4),
        "y": None if sample.y is None else round(sample.y, 4),
        "unit": sample.position_unit,
    }


def _confirmed_candidate(
    analysis: ExperimentAnalysis | None,
    confirmed_type: ExperimentType | None,
) -> ExperimentCandidate | None:
    if analysis is None:
        return None
    if confirmed_type is not None:
        for item in analysis.candidates:
            if item.experiment_type is confirmed_type:
                return item
        if analysis.selected and analysis.selected.experiment_type is confirmed_type:
            return analysis.selected
    return analysis.selected


def _result_payload(
    candidate: ExperimentCandidate | None,
    analysis: ExperimentAnalysis | None,
) -> dict[str, Any]:
    if candidate is None:
        return {
            "available": False,
            "experiment_type": None,
            "parameters": {},
            "units": {},
            "r2": None,
            "nrmse": None,
            "coverage": None if analysis is None else analysis.coverage,
        }
    return {
        "available": True,
        "experiment_type": candidate.experiment_type.value,
        "label": candidate.label,
        "confidence": round(candidate.confidence, 4),
        "formula_id": candidate.fit.formula_id,
        "parameters": candidate.fit.parameters,
        "units": candidate.fit.units,
        "r2": round(candidate.fit.r2, 4),
        "nrmse": round(candidate.fit.nrmse, 4),
        "n_samples": candidate.fit.n_samples,
        "time_range_s": [candidate.fit.time_start_s, candidate.fit.time_end_s],
        "frame_range": [candidate.fit.frame_start, candidate.fit.frame_end],
        "evidence": list(candidate.evidence),
        "missing": list(candidate.missing),
        "coverage": None if analysis is None else round(analysis.coverage, 4),
        "position_unit": None if analysis is None else analysis.position_unit,
    }
