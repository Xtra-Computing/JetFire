"""CPU regressions against the actual installed official request coroutine."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from jetfire_vllm.bench import official_client as adapter


def upstream():
    path = adapter.client_identity()['upstream_path']
    name = '_jetfire_ae_upstream_endpoint_test'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Session:
    def __init__(self, chunks, aligned=True):
        self.chunks = chunks
        self.aligned = aligned
        self.payload = None

    def post(self, **kwargs):
        self.payload = kwargs
        return self

    async def __aenter__(self):
        return SimpleNamespace(status=200, content=self)

    async def __aexit__(self, *args):
        return False

    async def iter_any(self):
        if self.aligned:
            async for chunk in adapter.aligned_sse_chunks(self.raw_chunks()):
                yield chunk
        else:
            async for chunk in self.raw_chunks():
                yield chunk

    async def raw_chunks(self):
        for chunk in self.chunks:
            await asyncio.sleep(0)
            yield chunk


def stream(text='你😀€'):
    objects = [{'choices': [{'text': text}]}, {'usage': {'completion_tokens': 128}}]
    return b''.join(('data: ' + json.dumps(v, ensure_ascii=False) + '\n\n').encode() for v in objects) + b'data: [DONE]\n\n'


async def request(module, chunks, aligned=True):
    session = Session(chunks, aligned)
    inputs = module.RequestFuncInput(prompt='test', api_url='http://localhost/v1/completions',
                                    prompt_len=32768, output_len=128, model='test-model', ignore_eos=True)
    result = await module.ASYNC_REQUEST_FUNCS['vllm'](inputs, session)
    return result, session


def test_exact_observed_128000_byte_boundary_fails_old_and_passes_adapter():
    module = upstream()
    prefix = 'data: {"choices":[{"text":"'
    text = 'a' * (127998 - len(prefix)) + '你'
    payload = (prefix + text + '"}]}\n\n').encode() + stream('tail')
    chunks = [payload[:128000], payload[128000:]]
    result, _ = asyncio.run(request(module, chunks, aligned=False))
    assert not result.success and 'UnicodeDecodeError' in result.error and '127998-127999' in result.error
    result, session = asyncio.run(request(module, chunks))
    assert result.success and result.output_tokens == 128 and result.generated_text == text + 'tail'
    assert session.payload['json']['max_tokens'] == 128
    assert session.payload['json']['stream'] is True
    assert session.payload['json']['stream_options'] == {'include_usage': True}
    assert session.payload['json']['ignore_eos'] is True


def test_all_two_chunk_byte_splits_preserve_actual_official_results():
    module = upstream()
    payload = stream()
    for cut in range(len(payload) + 1):
        result, _ = asyncio.run(request(module, [payload[:cut], payload[cut:]]))
        assert result.success, (cut, result.error)
        assert result.generated_text == '你😀€' and result.output_tokens == 128


def test_one_byte_chunks_and_coalesced_events_produce_same_messages():
    module = upstream()
    original = module.StreamedResponseHandler
    expected = original().add_chunk(stream())
    handler = original()
    async def frames():
        return [v async for v in adapter.aligned_sse_chunks(Session([bytes([b]) for b in stream()]).raw_chunks())]
    frames = asyncio.run(frames())
    assert b''.join(frames) == stream()
    actual = [message for frame in frames for message in handler.add_chunk(frame)]
    assert actual == expected


@pytest.mark.parametrize('bad', [b'\xff', b'\xe4\xff', b'\xf0\x9f\xff'])
def test_malformed_utf8_remains_a_failed_official_request(bad):
    module = upstream()
    result, _ = asyncio.run(request(module, [stream(), bad]))
    assert not result.success and 'UnicodeDecodeError' in result.error


@pytest.mark.parametrize('tail', [b'\xe4', b'\xe4\xbd', b'\xf0\x9f\x98'])
def test_truncated_utf8_at_eof_fails_even_after_valid_usage(tail):
    module = upstream()
    result, _ = asyncio.run(request(module, [stream(), tail]))
    assert not result.success
    assert 'UnicodeDecodeError' in result.error and 'unexpected end of data' in result.error


def test_concurrent_requests_do_not_share_decoder_or_eof_state():
    module = upstream()
    async def concurrent():
        payload = stream()
        good = [payload[i:i+1] for i in range(len(payload))]
        return await asyncio.gather(request(module, good), request(module, [payload, b'\xe4']), request(module, good))
    results = asyncio.run(concurrent())
    assert [r.success for r, _ in results] == [True, False, True]
    assert results[0][0].generated_text == results[2][0].generated_text == '你😀€'





def test_installation_is_guarded_and_wraps_reader_without_changing_bytes():
    class Reader:
        async def iter_any(self):
            for byte in stream():
                yield bytes([byte])
    module = SimpleNamespace(StreamReader=Reader)
    adapter.install_incremental_decoder(module)
    async def collect():
        return b''.join([chunk async for chunk in Reader().iter_any()])
    assert asyncio.run(collect()) == stream()
    with pytest.raises(RuntimeError, match='twice'):
        adapter.install_incremental_decoder(module)


def test_prefix_and_json_string_whitespace_survives_arbitrary_byte_cuts():
    module = upstream()
    payload = stream(' leading and  interior spaces\t trailing ')
    for cut in range(len(payload) + 1):
        result, _ = asyncio.run(request(module, [payload[:cut], payload[cut:]]))
        assert result.success, (cut, result.error)
        assert result.generated_text == ' leading and  interior spaces\t trailing '


def test_complete_final_event_without_separator_is_not_discarded():
    module = upstream()
    payload = stream().rstrip(b'\n')
    result, _ = asyncio.run(request(module, [payload]))
    assert result.success and result.output_tokens == 128 and result.generated_text == '你😀€'


@pytest.mark.parametrize("warmups", [0, 1])
@pytest.mark.parametrize("prefill_len,decode_len", [(32768, 128), (1009, 64), (4097, 257)])
def test_harness_preserves_official_cli_arguments(warmups, tmp_path, prefill_len, decode_len):
    from jetfire_vllm.bench.official_serve_benchmark import RunSpec, _bench_cmd
    args = SimpleNamespace(served_model_name="jetfire-test", model="/models/tokenizer",
                           ctx=prefill_len, output_len=decode_len, request_rate="inf",
                           server_timeout_s=1200, seed=41, warmups=warmups)
    command = _bench_cmd(args, "http://127.0.0.1:8000", RunSpec("jetfire", 4, 16, 2),
                         tmp_path, "official_result.json")
    assert command[:3] == [sys.executable, "-m", "jetfire_vllm.bench.official_client"]
    expected = ["bench", "serve", "--backend", "vllm", "--base-url", "http://127.0.0.1:8000",
                "--endpoint", "/v1/completions", "--model", "jetfire-test", "--tokenizer", "/models/tokenizer",
                "--dataset-name", "random", "--random-input-len", str(prefill_len), "--random-output-len", str(decode_len),
                "--random-range-ratio", "0", "--num-prompts", "16"]
    if warmups:
        expected += ["--num-warmups", "1"]
    expected += ["--request-rate", "inf", "--max-concurrency", "4", "--ready-check-timeout-sec", "60",
                 "--ignore-eos", "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
                 "--save-result", "--save-detailed", "--result-dir", str(tmp_path), "--result-filename",
                 "official_result.json", "--seed", "43"]
    assert command[3:] == expected


def test_client_record_preserves_cli_tail_and_source_identity(tmp_path):
    arguments = ["bench", "serve", "--seed", "7", "--result-dir", str(tmp_path),
                 "--result-filename", "official_result.json"]
    identity = adapter.client_identity()
    adapter._record_client(arguments, identity)
    record = json.loads((tmp_path / "client_decoder.json").read_text())
    assert record["identity"] == identity
    assert record["identity"]["adapter_protocol"] == "strict-sse-frames-v1"
    assert record["official_command"][:3] == [sys.executable, "-m", "vllm.entrypoints.cli.main"]
    assert record["wrapped_command"][:3] == [sys.executable, "-m", "jetfire_vllm.bench.official_client"]
    assert record["official_command"][3:] == record["wrapped_command"][3:] == arguments


def test_real_http_official_cli_preserves_unicode_spaces_and_request_controls(tmp_path):
    """Exercise the real official CLI with a tiny local tokenizer and HTTP server."""
    import os
    from aiohttp import web
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    repo = Path(__file__).resolve().parents[1]
    tokenizer_dir = tmp_path / 'tokenizer'
    tokenizer = Tokenizer(WordLevel({'[UNK]': 0, **{f'word{i}': i+1 for i in range(32)}}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', model_max_length=1024).save_pretrained(tokenizer_dir)
    calls = []

    async def completion(request):
        calls.append(await request.json())
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        first = ('data: ' + json.dumps({'choices': [{'text': ' 你😀€ '}]}, ensure_ascii=False) + '\n\n').encode()
        cuts = sorted({2, 5, 6, first.find('你'.encode())+1,
                       first.find('😀'.encode())+2, len(first)-1})
        previous = 0
        for end in cuts + [len(first)]:
            await response.write(first[previous:end])
            previous = end
            await asyncio.sleep(.005)
        for frame in [b'data: {"choices":[{"text":"tail"}]}\n\n',
                      b'data: {"usage":{"completion_tokens":2}}\n\n', b'data: [DONE]\n\n']:
            await response.write(frame)
            await asyncio.sleep(.005)
        await response.write_eof()
        return response

    async def models(request):
        return web.json_response({'data': [{'id': 'jetfire-test'}]})

    async def metrics(request):
        return web.Response(text='')

    async def check():
        application = web.Application()
        application.router.add_post('/v1/completions', completion)
        application.router.add_get('/v1/models', models)
        application.router.add_get('/metrics', metrics)
        runner = web.AppRunner(application)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        destination = tmp_path / 'results'
        command = [sys.executable, '-m', 'jetfire_vllm.bench.official_client', 'bench', 'serve',
                   '--backend', 'vllm', '--base-url', f'http://127.0.0.1:{port}',
                   '--endpoint', '/v1/completions', '--model', 'jetfire-test', '--tokenizer', str(tokenizer_dir),
                   '--dataset-name', 'random', '--random-input-len', '16', '--random-output-len', '2',
                   '--random-range-ratio', '0', '--num-prompts', '2', '--num-warmups', '1',
                   '--request-rate', 'inf', '--max-concurrency', '2', '--ready-check-timeout-sec', '0',
                   '--ignore-eos', '--disable-tqdm', '--save-result', '--save-detailed',
                   '--result-dir', str(destination), '--result-filename', 'official_result.json', '--seed', '0']
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONNOUSERSITE': '1', 'OMP_NUM_THREADS': '1',
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'VLLM_DO_NOT_TRACK': '1'}
        env.pop('PYTHONPATH', None)
        try:
            process = await asyncio.create_subprocess_exec(*command, cwd=repo, env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            output, _ = await asyncio.wait_for(process.communicate(), timeout=60)
            assert process.returncode == 0, output.decode()
            result = json.loads((destination/'official_result.json').read_text())
            assert result['completed'] == 2 and result['failed'] == 0
            assert result['output_lens'] == [2, 2]
            assert result['generated_texts'] == [' 你😀€ tail', ' 你😀€ tail']
            assert all(call['max_tokens'] == 2 and call['stream'] is True
                       and call['stream_options'] == {'include_usage': True}
                       and call['ignore_eos'] is True for call in calls)
            identity = json.loads((destination/'client_decoder.json').read_text())
            assert identity['identity']['worker_path'] == str(repo/'jetfire_vllm/bench/official_client.py')
            assert identity['wrapped_command'][3:] == command[3:]
        finally:
            await runner.cleanup()

    asyncio.run(check())
