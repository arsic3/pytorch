"""
Wire protocol for the torchmux coordinator.

Defines op names, error codes, framing, and header shapes shared between
``_coordinator.py`` and ``_coord_client.py``.

Wire format per message::

    [ 4 bytes big-endian u32 : header_len ]
    [ header_len bytes       : UTF-8 JSON header ]
    [ 8 bytes big-endian u64 : payload_len ]    # 0 if no payload
    [ payload_len bytes      : concatenated tensor bodies ]

Every response header includes ``{"ok": bool, "error": str | None, ...}``.
"""

import asyncio
import json
import struct
from typing import TypedDict


# ---- Op names ----

OP_REGISTER = "register"
OP_WAIT_FOR_TURN = "wait_for_turn"
OP_PREPARE = "prepare"
OP_RELEASE_GPU = "release_gpu"
OP_RELEASE_BATON = "release_baton"
OP_ACQUIRE_BATON = "acquire_baton"
OP_DONE = "done"

# ---- Error codes ----

ERR_NO_PEERS = "no_peers"
ERR_PEER_GONE = "peer_gone"
ERR_MISMATCH = "mismatch"

# ---- Header shapes ----


class TensorHeader(TypedDict):
    shape: list[int]
    dtype: str
    nbytes: int


class SendEntry(TypedDict):
    dsts: list[int]
    tensor: TensorHeader | None


class RecvEntry(TypedDict):
    src: int
    tensor: TensorHeader | None


# ---- Framing ----


async def read_message(
    reader: asyncio.StreamReader,
) -> tuple[dict, bytes]:
    (hdr_len,) = struct.unpack(">I", await reader.readexactly(4))
    header: dict = json.loads((await reader.readexactly(hdr_len)).decode("utf-8"))
    (payload_len,) = struct.unpack(">Q", await reader.readexactly(8))
    payload: bytes = await reader.readexactly(payload_len) if payload_len else b""
    return header, payload


async def write_message(
    writer: asyncio.StreamWriter,
    header: dict,
    payload: bytes = b"",
) -> None:
    hdr_bytes = json.dumps(header).encode("utf-8")
    writer.write(struct.pack(">I", len(hdr_bytes)))
    writer.write(hdr_bytes)
    writer.write(struct.pack(">Q", len(payload)))
    if payload:
        writer.write(payload)
    await writer.drain()
