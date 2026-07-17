# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM TIPC frontend server.

Spawned by TipcProcessManager when VLLM_TIPC_SERVER_NUM > 0.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace
from typing import Any

import uvloop

from vllm.entrypoints.openai.api_server import build_async_engine_client
from vllm.entrypoints.openai.tipc_server import TipcSocketServer


def run_tipc_server_worker_proc(
    args: Namespace,
    client_config: dict[str, Any] | None = None,
) -> None:
    """Sync entrypoint for TipcProcessManager child processes."""
    client_config = client_config or {}
    uvloop.run(run_tipc_server_worker(args, client_config))


async def run_tipc_server_worker(
    args: Namespace,
    client_config: dict[str, Any],
) -> None:
    """Build EngineClient over pre-allocated ZMQ addresses, then serve TIPC."""
    client_config = dict(client_config)
    tipc_type = int(client_config.pop("tipc_type"))
    tipc_instance = int(client_config.pop("tipc_instance"))

    async with build_async_engine_client(
        args,
        client_config=client_config,
    ) as engine_client:
        try:
            server = TipcSocketServer(engine_client, tipc_type, tipc_instance)
            await server.serve_forever()
        except OSError:
            # No AF_TIPC on this host; keep the ZMQ frontend slot alive.
            await asyncio.Event().wait()
