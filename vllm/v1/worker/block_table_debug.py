# SPDX-License-Identifier: Apache-2.0
"""Debug: detect append_row + add_row on the same request in one _update_states step.

Enable: VLLM_DEBUG_BLOCK_TABLE=1
Optional: VLLM_DEBUG_BLOCK_TABLE_DUMP_DIR, VLLM_DEBUG_BLOCK_TABLE_ABORT=1 (tests)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


@dataclass
class BlockTableViolation:
    step: int
    kind: str
    req_id: str | None = None
    row_idx: int | None = None


class BlockTableDebug:
    """Per-step tracker for block table dual-path invariant.

    Invariant (one _update_states step, one request):
      continue path → append_row only  (Loop 2, req already in batch)
      re-add path   → add_row only    (Loop 3, via reqs_to_add)

    Hooks: begin/end in gpu_model_runner; record_* in block_table.py.
    """

    def __init__(self) -> None:
        self.enabled = envs.VLLM_DEBUG_BLOCK_TABLE
        self.dump_dir = envs.VLLM_DEBUG_BLOCK_TABLE_DUMP_DIR
        self.abort = envs.VLLM_DEBUG_BLOCK_TABLE_ABORT
        self.step = 0  # _update_states call counter
        self.violations: list[BlockTableViolation] = []  # all steps, for tests
        # Per-step scratch (cleared in begin_update_states):
        self._continued: set[str] = set()  # Loop 2 continue
        self._added: set[str] = set()  # Loop 3 reqs_to_add
        self._append_rows: set[int] = set()  # BlockTable.append_row rows
        self._add_rows: set[int] = set()  # BlockTable.add_row rows
        if self.enabled:
            if self.dump_dir:
                os.makedirs(self.dump_dir, exist_ok=True)
            logger.info("Block table debug enabled (dump_dir=%s)", self.dump_dir)

    def begin_update_states(self, scheduler_output: SchedulerOutput) -> None:
        """Start a new step; reset per-step sets."""
        if not self.enabled:
            return
        self.step += 1
        self._continued.clear()
        self._added.clear()
        self._append_rows.clear()
        self._add_rows.clear()

        # Risk signal only (not a hard violation by itself).
        new_ids = {r.req_id for r in scheduler_output.scheduled_new_reqs}
        cached_ids = set(scheduler_output.scheduled_cached_reqs.req_ids)
        dual = sorted(new_ids & cached_ids)
        if dual:
            logger.warning(
                "[block_table_debug] step=%d scheduler dual-listed %s",
                self.step,
                dual,
            )

    def note_continued(self, req_id: str, row_idx: int) -> None:
        """Loop 2: req_index is not None → append path."""
        if self.enabled:
            self._continued.add(req_id)

    def note_added(self, req_id: str, row_idx: int) -> None:
        """Loop 3: add_request after reqs_to_add."""
        if self.enabled:
            self._added.add(req_id)

    def record_append(self, row_idx: int) -> None:
        """BlockTable.append_row (incremental)."""
        if self.enabled:
            self._append_rows.add(row_idx)

    def record_add(self, row_idx: int) -> None:
        """BlockTable.add_row (full rewrite)."""
        if self.enabled:
            self._add_rows.add(row_idx)

    def end_update_states(self, runner: GPUModelRunner) -> None:
        """Check invariant; log / dump / abort on violation."""
        if not self.enabled:
            return

        index_to_req = {
            idx: rid for rid, idx in runner.input_batch.req_id_to_index.items()
        }
        req_to_index = runner.input_batch.req_id_to_index
        found: list[BlockTableViolation] = []

        # Same req_index: both append_row and add_row in one step.
        for row in self._append_rows & self._add_rows:
            found.append(
                BlockTableViolation(
                    self.step, "append_and_add_same_row", index_to_req.get(row), row
                )
            )

        # Same req_id: both continue and re-add paths in _update_states.
        for req_id in self._continued & self._added:
            found.append(
                BlockTableViolation(
                    self.step,
                    "continued_and_added_same_req",
                    req_id,
                    req_to_index.get(req_id),
                )
            )

        if not found:
            return

        self.violations.extend(found)
        for v in found:
            logger.error(
                "[block_table_debug] step=%d %s req_id=%s row=%s",
                v.step,
                v.kind,
                v.req_id,
                v.row_idx,
            )

        if self.dump_dir:
            payload = {
                "step": self.step,
                "violations": [v.__dict__ for v in found],
                "continued": sorted(self._continued),
                "added": sorted(self._added),
                "append_rows": sorted(self._append_rows),
                "add_rows": sorted(self._add_rows),
            }
            with open(
                os.path.join(self.dump_dir, "latest_violation.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(payload, f, indent=2)

        if self.abort:
            raise RuntimeError(f"block table invariant violated: {found[0].kind}")


_debug: BlockTableDebug | None = None


def get_block_table_debug() -> BlockTableDebug:
    global _debug
    if _debug is None:
        _debug = BlockTableDebug()
    return _debug
