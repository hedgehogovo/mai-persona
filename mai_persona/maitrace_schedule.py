"""Schedule provider API response normalization.

Supports the `maibot.schedule` protocol (v1) envelope used by Mai Life and
MaiTrace's legacy day-timeline shape.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


def normalize_snapshot(response: Any) -> dict[str, Any]:
    """Normalize get_current_activity's public snapshot without exposing private data."""

    if not isinstance(response, dict):
        raise ValueError("自主规划插件未返回有效的日程快照")
    if response.get("success") is False or response.get("error"):
        raise ValueError(str(response.get("message") or response.get("error") or "API 调用失败"))

    data = response.get("data", response)
    if not isinstance(data, dict):
        raise ValueError("自主规划插件日程快照格式无效")

    # `activity` 是 MaiTrace 的旧字段；`maibot.schedule` 协议用 `current`。
    current = data.get("activity", data.get("current"))
    if current is not None and not isinstance(current, dict):
        current = {"name": str(current)}
    if not data.get("has_activity", current is not None):
        current = None
    if current is not None and (
        current.get("private") is True
        or current.get("is_private") is True
        or str(current.get("visibility", "")).lower() in {"private", "personal", "hidden"}
    ):
        current = None

    next_activities = data.get("next_activities", [])
    if not isinstance(next_activities, list):
        next_activities = []
    public_next = [
        item
        for item in next_activities[:8]
        if isinstance(item, dict)
        and item.get("private") is not True
        and item.get("is_private") is not True
        and str(item.get("visibility", "")).lower() not in {"private", "personal", "hidden"}
    ]
    return {
        "current_activity": current,
        "next_activities": public_next,
    }


def normalize_schedule(response: Any) -> list[dict[str, Any]]:
    """Normalize a day-timeline response (MaiTrace legacy or maibot.schedule v1)."""

    if isinstance(response, dict):
        if response.get("success") is False or response.get("error"):
            raise ValueError(str(response.get("message") or response.get("error")))
        response = response.get("activities", response.get("schedule", response.get("data")))
        if isinstance(response, dict):
            # `maibot.schedule` 协议把活动列表包在 data.activities 里。
            response = response.get("activities", response.get("schedule"))
    if not isinstance(response, list):
        raise ValueError("日程提供方未返回有效的日程列表")
    return [
        item
        for item in response
        if isinstance(item, dict)
        and item.get("private") is not True
        and item.get("is_private") is not True
        and str(item.get("visibility", "")).lower() not in {"private", "personal", "hidden"}
    ][:24]


def _activity_name(activity: dict[str, Any] | None) -> str:
    if not activity:
        return ""
    for key in ("name", "activity", "title", "summary"):
        value = re.sub(r"\s+", " ", str(activity.get(key) or "")).strip()
        if value:
            return value[:120]
    return ""


def format_current_activity(activity: dict[str, Any] | None) -> str:
    name = _activity_name(activity)
    if not name:
        return ""
    window = re.sub(r"\s+", " ", str(activity.get("time_window") or "")).strip()
    if not window:
        # maibot.schedule 协议给的是 start_at/end_at，没有 time_window。
        start = _format_time(activity.get("start") or activity.get("start_at"))
        end = _format_time(activity.get("end") or activity.get("end_at"))
        window = f"{start}-{end}" if start and end else start or end
    description = re.sub(r"\s+", " ", str(activity.get("description") or "")).strip()
    suffix = f"（{window}）" if window else ""
    if description and description != name:
        return f"{name}{suffix}：{description[:160]}"
    return f"{name}{suffix}"


def _format_time(value: Any) -> str:
    text = str(value or "").strip()
    if "T" in text:
        text = text.split("T", 1)[1]
    if len(text) >= 5 and text[2] == ":":
        return text[:5]
    try:
        return datetime.fromtimestamp(float(value)).strftime("%H:%M")
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def format_schedule_summary(
    snapshot: dict[str, Any],
    schedule: list[dict[str, Any]] | None = None,
    limit: int = 8,
) -> str:
    entries: list[str] = []
    current = snapshot.get("current_activity")
    current_text = format_current_activity(current if isinstance(current, dict) else None)
    if current_text:
        entries.append(f"当前：{current_text}")

    if schedule is not None:
        for item in schedule[:limit]:
            name = _activity_name(item)
            if not name:
                continue
            start = _format_time(item.get("start") or item.get("start_at"))
            end = _format_time(item.get("end") or item.get("end_at"))
            time_window = f"{start}-{end}" if start and end else start or end
            description = re.sub(r"\s+", " ", str(item.get("description") or "")).strip()
            label = f"{time_window} {name}".strip()
            if description and description != name:
                label += f"：{description[:120]}"
            entries.append(label)
        return "；".join(dict.fromkeys(entries))[:800]

    next_activities = snapshot.get("next_activities", [])
    if isinstance(next_activities, list):
        for item in next_activities[:limit]:
            if not isinstance(item, dict):
                continue
            name = _activity_name(item)
            if not name:
                continue
            time_window = re.sub(r"\s+", " ", str(item.get("time") or item.get("time_window") or "")).strip()
            entries.append(f"{time_window} {name}".strip())
    return "；".join(entries)[:800]


def activity_name(snapshot: dict[str, Any]) -> str:
    current = snapshot.get("current_activity")
    return _activity_name(current if isinstance(current, dict) else None)
