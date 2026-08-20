"""Generic per-physics-step telemetry collection."""

from __future__ import annotations

from typing import Any, Protocol


class PhysicsRule(Protocol):
    """Task semantics consumed by the generic physics monitor."""

    def start(self, task: Any) -> None:
        ...

    def observe(self, task: Any, step_index: int) -> None:
        ...

    def finalize(self, task: Any) -> dict[str, Any]:
        ...


class PhysicsMonitor:
    """Forward simulation-step notifications to a task-specific rule."""

    def __init__(self, rule: PhysicsRule):
        self.rule = rule
        self.step_count = 0
        self.started = False

    def start(self, task: Any) -> None:
        if self.started:
            raise RuntimeError("Physics monitor has already started")
        self.rule.start(task)
        self.started = True

    def on_physics_step(self, task: Any) -> None:
        if not self.started:
            raise RuntimeError("Physics monitor must be started before simulation")
        self.rule.observe(task, self.step_count)
        self.step_count += 1

    def finalize(self, task: Any) -> dict[str, Any]:
        if not self.started:
            raise RuntimeError("Physics monitor must be started before finalization")
        report = self.rule.finalize(task)
        report["physics_step_count"] = self.step_count
        return report
