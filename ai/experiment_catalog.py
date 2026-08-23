"""Standard formulas and teaching facts for first-version experiments.

DeepSeek must quote these entries rather than inventing symbols or equations.
"""

from __future__ import annotations

from typing import Any

from ai.contracts import ExperimentType

CATALOG_VERSION = "1.0.0"

_ENTRIES: dict[str, dict[str, Any]] = {
    "uniform.s": {
        "id": "uniform.s",
        "experiment_type": ExperimentType.UNIFORM_LINEAR.value,
        "name": "匀速直线运动位移",
        "expression": "s = s0 + v t",
        "symbols": {
            "s": "沿主运动方向的位移",
            "s0": "计时起点的位置",
            "v": "恒定速度",
            "t": "时间",
        },
        "conditions": [
            "合外力为零或近似为零",
            "加速度相对速度变化可忽略",
            "在连续可见段内拟合，不跨越遮挡",
        ],
        "common_errors": [
            "把平均速度当成任意时刻的瞬时速度",
            "未标定就把像素速度写成 m/s",
        ],
    },
    "accel.s": {
        "id": "accel.s",
        "experiment_type": ExperimentType.UNIFORM_ACCEL.value,
        "name": "匀加速直线运动位移",
        "expression": "s = s0 + v0 t + (1/2) a t^2",
        "symbols": {
            "s": "沿主运动方向的位移",
            "s0": "计时起点的位置",
            "v0": "初速度",
            "a": "恒定加速度",
            "t": "时间",
        },
        "conditions": [
            "合外力近似恒定",
            "二次项显著优于纯线性模型",
            "在连续可见段内拟合，不跨越遮挡",
        ],
        "common_errors": [
            "把末速度公式和位移公式混用",
            "未检查二次模型是否真的优于匀速",
        ],
    },
    "freefall.y": {
        "id": "freefall.y",
        "experiment_type": ExperimentType.FREE_FALL.value,
        "name": "自由落体竖直位移",
        "expression": "y = y0 + v0y t + (1/2) g t^2",
        "symbols": {
            "y": "竖直位移（分析坐标默认向下为正）",
            "y0": "计时起点的纵坐标",
            "v0y": "竖直初速度",
            "g": "重力加速度",
            "t": "时间",
        },
        "conditions": [
            "水平漂移远小于竖直位移",
            "竖直方向二次模型显著优于线性模型",
            "空气阻力可忽略",
            "需要标尺才能把 g 写成 m/s^2",
        ],
        "common_errors": [
            "把屏幕坐标向上为正，却直接套用向下为正的拟合结果",
            "无标尺时仍报告 9.8 m/s^2",
        ],
    },
    "projectile.xy": {
        "id": "projectile.xy",
        "experiment_type": ExperimentType.PROJECTILE.value,
        "name": "抛体运动分解",
        "expression": "x = x0 + v0x t;  y = y0 + v0y t + (1/2) g t^2",
        "symbols": {
            "x": "水平位移",
            "y": "竖直位移（分析坐标默认向下为正）",
            "v0x": "水平初速度",
            "v0y": "竖直初速度",
            "g": "重力加速度",
            "t": "时间",
        },
        "conditions": [
            "水平方向近似匀速",
            "竖直方向近似匀加速",
            "忽略空气阻力",
            "需要标尺才能输出 SI 单位的速度和 g",
        ],
        "common_errors": [
            "把斜抛当成自由落体",
            "用像素二次项系数直接当作 g（m/s^2）",
        ],
    },
    "pendulum.period": {
        "id": "pendulum.period",
        "experiment_type": ExperimentType.PENDULUM.value,
        "name": "单摆周期",
        "expression": "T = 2π √(L / g)",
        "symbols": {
            "T": "小角度周期",
            "L": "摆长",
            "g": "重力加速度",
        },
        "conditions": [
            "摆角较小，近似简谐运动",
            "空气阻力和摆线质量可忽略",
            "周期由过零或自相关从轨迹测得",
            "没有摆长时只报告周期，不反推 g",
        ],
        "common_errors": [
            "大角度时仍用小角度公式反推 g",
            "把摆球到画面边缘的距离当成摆长",
        ],
    },
    "pendulum.g": {
        "id": "pendulum.g",
        "experiment_type": ExperimentType.PENDULUM.value,
        "name": "由周期反推重力加速度",
        "expression": "g = 4π^2 L / T^2",
        "symbols": {
            "g": "重力加速度",
            "L": "摆长",
            "T": "测得的周期",
        },
        "conditions": [
            "必须已知摆长 L",
            "T 来自轨迹，不是理论值",
        ],
        "common_errors": [
            "缺少摆长时仍输出 g",
        ],
    },
}

TYPE_TO_FORMULA_IDS = {
    ExperimentType.UNIFORM_LINEAR: ("uniform.s",),
    ExperimentType.UNIFORM_ACCEL: ("accel.s",),
    ExperimentType.FREE_FALL: ("freefall.y",),
    ExperimentType.PROJECTILE: ("projectile.xy",),
    ExperimentType.PENDULUM: ("pendulum.period", "pendulum.g"),
    ExperimentType.UNKNOWN: (),
}


def formula_entry(formula_id: str) -> dict[str, Any] | None:
    entry = _ENTRIES.get(formula_id)
    if entry is None:
        return None
    return dict(entry)


def formulas_for(experiment_type: ExperimentType | str) -> list[dict[str, Any]]:
    if isinstance(experiment_type, str):
        try:
            kind = ExperimentType(experiment_type)
        except ValueError:
            return []
    else:
        kind = experiment_type
    return [dict(_ENTRIES[fid]) for fid in TYPE_TO_FORMULA_IDS.get(kind, ()) if fid in _ENTRIES]


def catalog_payload(experiment_type: ExperimentType | str | None = None) -> dict[str, Any]:
    if experiment_type is None:
        entries = [dict(item) for item in _ENTRIES.values()]
    else:
        entries = formulas_for(experiment_type)
    return {"version": CATALOG_VERSION, "formulas": entries}
