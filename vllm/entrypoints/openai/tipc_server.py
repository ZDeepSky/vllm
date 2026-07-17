# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AF_TIPC JSON request server backed by EngineClient."""

from __future__ import annotations

import asyncio
import json
import signal
import socket
import struct
from typing import Any

from vllm.engine.protocol import EngineClient
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

# Linux tipc.h: TIPC_ADDR_NAMESEQ
TIPC_ADDR_NAMESEQ = 1
_HEADER = struct.Struct("!I")


class TipcSocketServer:
    """Minimal AF_TIPC JSON request server backed by EngineClient."""

    def __init__(
        self,
        engine_client: EngineClient,
        tipc_type: int,
        tipc_instance: int,
    ):
        self.engine_client = engine_client
        self.tipc_type = tipc_type
        self.tipc_instance = tipc_instance
        self._sock: socket.socket | None = None
        self._stop = asyncio.Event()

    def _bind(self) -> socket.socket:
        if not hasattr(socket, "AF_TIPC"):
            raise OSError("socket.AF_TIPC is not available on this platform")
        sock = socket.socket(socket.AF_TIPC, socket.SOCK_RDM)
        sock.bind(
            (TIPC_ADDR_NAMESEQ, self.tipc_type, self.tipc_instance, self.tipc_instance)
        )
        sock.setblocking(False)
        return sock

    async def serve_forever(self) -> None:
        self._sock = self._bind()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except (NotImplementedError, RuntimeError):
                pass

        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, addr = await loop.sock_recvfrom(self._sock, 65535)
            except (BlockingIOError, InterruptedError):
                await asyncio.sleep(0)
                continue
            except OSError:
                if self._stop.is_set():
                    break
                raise
            try:
                reply = await self.handle_one(data)
            except Exception as e:
                reply = _encode({"ok": False, "error": str(e)})
            try:
                await loop.sock_sendto(self._sock, reply, addr)
            except OSError:
                pass

        self._sock.close()
        self._sock = None

    async def handle_one(self, raw: bytes) -> bytes:
        req = _decode(raw)
        op = req.get("op", "health")
        if op == "health":
            return _encode({"ok": True, "op": "health"})
        if op == "probe":
            tasks = await self.engine_client.get_supported_tasks()
            return _encode(
                {"ok": True, "op": "probe", "supported_tasks": list(tasks)}
            )
        if op == "generate":
            prompt = req["prompt"]
            max_tokens = int(req.get("max_tokens", 16))
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                skip_clone=True,
            )
            request_id = random_uuid()
            final = None
            async for output in self.engine_client.generate(
                prompt, sampling_params, request_id
            ):
                final = output
            assert final is not None
            texts = [o.text for o in final.outputs]
            return _encode({"ok": True, "op": "generate", "text": texts})
        return _encode({"ok": False, "error": f"unknown op: {op}"})


def _encode(obj: dict[str, Any]) -> bytes:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return _HEADER.pack(len(body)) + body


def _decode(raw: bytes) -> dict[str, Any]:
    if len(raw) >= 4:
        (n,) = _HEADER.unpack_from(raw, 0)
        if n + 4 == len(raw):
            raw = raw[4:]
    return json.loads(raw.decode("utf-8"))
