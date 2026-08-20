# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Opt-in profiling helpers shared by training and inference entry points."""

import contextlib
import dataclasses
import logging
import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import torch

import weathergen.common.config as config
from weathergen.common.config import Config
from weathergen.utils.distributed import get_rank, is_root

logger = logging.getLogger(__name__)

_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"
_MAX_MEMORY_EVENTS = 100_000


class ProfilerSection:
    """NVTX range for a named region. Active when nsys profiling is enabled."""

    def __init__(self, name: str, profile: bool | None = None):
        self.name = name
        if profile is None:
            profile = os.environ.get("WEATHERGEN_NSYS_PROFILING", "0") == "1"
        self.profile = profile

    def __enter__(self):
        if self.profile:
            torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, *args, **kwargs):
        if self.profile:
            torch.cuda.nvtx.range_pop()


@dataclasses.dataclass(frozen=True)
class ProfilingConfig:
    """The profiling options needed by the runtime entry points."""

    enabled: bool = False
    memory_snapshot: bool = False

    @classmethod
    def from_config(cls, cf: Config) -> "ProfilingConfig":
        """Read profiling options while remaining compatible with older run configs."""
        profiling_cfg = cf.get("profiling") or {}
        memory_snapshot_cfg = profiling_cfg.get("memory_snapshot") or {}
        defaults = cls()
        return cls(
            enabled=profiling_cfg.get("enabled", defaults.enabled),
            memory_snapshot=memory_snapshot_cfg.get("enabled", defaults.memory_snapshot),
        )

    @property
    def records_memory_snapshot(self) -> bool:
        """Whether CUDA allocator history should be recorded for this run."""
        return self.enabled and self.memory_snapshot


@contextlib.contextmanager
def memory_snapshot_session(cf: Config) -> Iterator[None]:
    """Record and dump root-rank PyTorch CUDA allocator history for the enclosed runtime."""
    profiling_cfg = ProfilingConfig.from_config(cf)
    if not profiling_cfg.records_memory_snapshot or not is_root():
        yield
        return

    if not torch.cuda.is_available():
        logger.info("CUDA unavailable; not recording memory history.")
        yield
        return

    traces_path = config.get_path_profiling_traces(cf)
    traces_path.mkdir(exist_ok=True, parents=True)
    logger.info("Starting CUDA allocator memory-history recording.")
    torch.cuda.memory._record_memory_history(max_entries=_MAX_MEMORY_EVENTS)
    try:
        yield
    finally:
        snapshot_path = _memory_snapshot_path(traces_path)
        try:
            logger.info("Writing CUDA allocator memory snapshot to %s.", snapshot_path)
            torch.cuda.memory._dump_snapshot(snapshot_path)
        except Exception:
            logger.exception("Failed to write CUDA allocator memory snapshot to %s.", snapshot_path)
        finally:
            logger.info("Stopping CUDA allocator memory-history recording.")
            torch.cuda.memory._record_memory_history(enabled=None)


def _memory_snapshot_path(traces_path: Path) -> Path:
    timestamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
    return traces_path / f"{timestamp}_rank_{get_rank()}_memory_snapshot.pickle"
