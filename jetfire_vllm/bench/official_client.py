"""Run the official vLLM benchmark CLI with strict LF SSE event framing.

This subprocess-only adapter preserves raw bytes within complete events before
vLLM's unchanged request coroutine receives them. It does not modify installed
packages or the server. See docs/ARTIFACT_EVALUATION.md for protocol scope.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


async def aligned_sse_chunks(chunks):
    """Preserve bytes within each SSE event across arbitrary transport cuts.

    The official coroutine strips every input chunk. Supplying whole events
    prevents it from stripping a split `data: ` prefix or JSON string spaces.
    Validation is strict: malformed UTF-8 and an incomplete code point at EOF
    raise in the unchanged official request coroutine's existing error handler.
    """
    pending = bytearray()
    async for chunk in chunks:
        pending.extend(chunk)
        while True:
            end = pending.find(b'\n\n')
            if end < 0:
                break
            frame = bytes(pending[:end + 2])
            del pending[:end + 2]
            frame.decode('utf-8', errors='strict')
            yield frame
    if pending:
        trailing = bytes(pending)
        trailing.decode('utf-8', errors='strict')
        yield trailing


def install_incremental_decoder(aiohttp):
    reader = aiohttp.StreamReader
    if getattr(reader, '_jetfire_ae_incremental_utf8', False):
        raise RuntimeError('The benchmark decoder adapter was installed twice')
    original = reader.iter_any
    @functools.wraps(original)
    async def framed(self):
        async for chunk in aligned_sse_chunks(original(self)):
            yield chunk
    reader.iter_any = framed
    reader._jetfire_ae_incremental_utf8 = True


def client_identity() -> dict:
    distribution = importlib.metadata.distribution("vllm")
    upstream = Path(distribution.locate_file(
        "vllm/benchmarks/lib/endpoint_request_func.py")).resolve()
    transport = importlib.metadata.distribution("aiohttp")
    streams = Path(transport.locate_file("aiohttp/streams.py")).resolve()
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {
        "decoder": "incremental",
        "adapter_protocol": "strict-sse-frames-v1",
        "worker_path": str(Path(__file__).resolve()),
        "worker_sha256": digest(__file__),
        "vllm_version": distribution.version,
        "upstream_path": str(upstream),
        "upstream_sha256": digest(upstream),
        "aiohttp_version": transport.version,
        "upstream_streams_path": str(streams),
        "upstream_streams_sha256": digest(streams),
    }


def _record_client(argv: list[str], identity: dict) -> None:
    if "--result-dir" not in argv:
        return
    index = argv.index("--result-dir")
    if index + 1 == len(argv):
        return  # The official argument parser reports the missing value.
    directory = Path(argv[index + 1])
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "identity": identity,
        "official_command": [sys.executable, "-m", "vllm.entrypoints.cli.main", *argv],
        "wrapped_command": [sys.executable, "-m", "jetfire_vllm.bench.official_client", *argv],
    }
    (directory / "client_decoder.json").write_text(json.dumps(record, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    import aiohttp

    install_incremental_decoder(aiohttp)
    identity = client_identity()
    _record_client(arguments, identity)
    print("[jetfire-official-client] " + json.dumps(identity, sort_keys=True), flush=True)
    sys.argv = ["vllm", *arguments]
    from vllm.entrypoints.cli.main import main as official_main

    return official_main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
