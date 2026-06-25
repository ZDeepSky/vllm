# SPDX-License-Identifier: Apache-2.0
"""Block table dual-path debug: streaming fault-injection repro."""

from unittest.mock import patch

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.worker.block_table_debug import get_block_table_debug

from .test_gpu_model_runner import _schedule_new_request, dist_init, model_runner

pytestmark = pytest.mark.cpu_test


def _dual_path_scheduler_output(req_id: str) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=[1, 2, 3, 10, 4, 5],
                mm_features=[],
                sampling_params=SamplingParams(),
                pooling_params=None,
                block_ids=([0, 1],),
                num_computed_tokens=4,
                lora_request=None,
            )
        ],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[req_id],
            resumed_req_ids=set(),
            new_token_ids=[[]],
            all_token_ids={},
            new_block_ids=[([2],)],
            num_computed_tokens=[4],
            num_output_tokens=[0],
        ),
        num_scheduled_tokens={req_id: 2},
        total_num_scheduled_tokens=2,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


@pytest.fixture(autouse=True)
def enable_block_table_debug(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_DEBUG_BLOCK_TABLE", "1")
    monkeypatch.setenv("VLLM_DEBUG_BLOCK_TABLE_DUMP_DIR", str(tmp_path))
    import vllm.envs as envs
    import vllm.v1.worker.block_table_debug as debug_mod

    envs.VLLM_DEBUG_BLOCK_TABLE = True
    envs.VLLM_DEBUG_BLOCK_TABLE_DUMP_DIR = str(tmp_path)
    debug_mod._debug = None
    yield
    debug_mod._debug = None


def test_streaming_dual_path_append_and_add(model_runner, dist_init):
    """Streaming re-add + cached continue in one step → append_row + add_row."""
    req_id = "stream_req"
    model_runner._update_states(_schedule_new_request(req_id))

    with patch.object(model_runner.input_batch, "remove_request", return_value=None):
        model_runner._update_states(_dual_path_scheduler_output(req_id))

    kinds = {v.kind for v in get_block_table_debug().violations}
    assert "append_and_add_same_row" in kinds or "continued_and_added_same_req" in kinds
