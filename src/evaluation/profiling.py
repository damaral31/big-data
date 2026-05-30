"""Lightweight execution-time profiler.

The assignment explicitly asks to *report total execution time and profile the
ML phases*.  This profiler is used throughout the notebook: wrap any phase in
``with profiler.track("phase name"):`` and call ``profiler.report()`` at the end.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

import pandas as pd

logger = logging.getLogger(__name__)


class Profiler:
    def __init__(self):
        self._times: dict[str, float] = {}

    @contextmanager
    def track(self, phase: str):
        start = time.perf_counter()
        logger.info("[start] %s", phase)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._times[phase] = self._times.get(phase, 0.0) + elapsed
            logger.info("[done ] %s (%.2fs)", phase, elapsed)

    def record(self, phase: str, seconds: float) -> None:
        self._times[phase] = self._times.get(phase, 0.0) + seconds

    @property
    def total(self) -> float:
        return sum(self._times.values())

    def report(self) -> pd.DataFrame:
        df = pd.DataFrame(
            [(k, v) for k, v in self._times.items()],
            columns=["phase", "seconds"],
        ).sort_values("seconds", ascending=False)
        total = df["seconds"].sum() or 1.0
        df["pct"] = (df["seconds"] / total * 100).round(1)
        return df.reset_index(drop=True)
