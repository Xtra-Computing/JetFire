from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm import SamplingParams
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import PauseState, Scheduler
from vllm.v1.request import Request, RequestStatus

from jetfire_vllm.adapter import registry


_NATIVE_SCHEDULE = Scheduler.schedule


class _Blocks:
    def get_block_ids(self, allow_none=False):
        return ([1],)


class _KV:
    """Small allocator: active requests retain their own decode reservation."""
    def __init__(self, capacity=4):
        self.capacity = capacity
        self.resident = {'running'}
        self.steps = 0
        self.allocations = []
        self.raise_on = set()
        self.zero_calls = 0
        self.empty_kv_cache_blocks = _Blocks()

    def new_step_starts(self):
        self.steps += 1

    def can_fit_full_sequence(self, request):
        return request.request_id in self.resident or len(self.resident) < self.capacity

    def get_computed_blocks(self, request):
        return self.empty_kv_cache_blocks, 0

    def allocate_slots(self, request, num_new_tokens, **kwargs):
        self.allocations.append((request.request_id, num_new_tokens))
        if request.request_id in self.raise_on:
            raise RuntimeError('allocation failure')
        if not self.can_fit_full_sequence(request):
            return None
        self.resident.add(request.request_id)
        return _Blocks()

    def get_blocks(self, request_id):
        return _Blocks()

    def get_num_common_prefix_blocks(self, request_id):
        return [0]

    def take_new_block_ids(self):
        self.zero_calls += 1
        return [99] if self.zero_calls == 1 else []


class _Encoder:
    def __init__(self):
        self.calls = 0

    def get_freed_mm_hashes(self):
        self.calls += 1
        return ['finished-mm'] if self.calls == 1 else []


def _request(name, *, running=False, priority=0):
    request = Request(name, [1] * 16, SamplingParams(max_tokens=8), None, priority=priority)
    if running:
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = 16
        request.append_output_token_ids(2)
    return request


def _scheduler(*, capacity=4, limit=4, waiting=1, policy=SchedulingPolicy.FCFS):
    scheduler = Scheduler.__new__(Scheduler)
    active = _request('running', running=True)
    fields = dict(
        running=[active], waiting=create_request_queue(policy),
        skipped_waiting=create_request_queue(policy), policy=policy,
        max_num_scheduled_tokens=64, max_num_encoder_input_tokens=64,
        max_num_running_reqs=limit, max_model_len=64, _pause_state=PauseState.UNPAUSED,
        kv_cache_manager=_KV(capacity), scheduler_config=SimpleNamespace(
            long_prefill_token_threshold=0, enable_chunked_prefill=False,
            async_scheduling=False), lora_config=None, connector=None,
        ec_connector=None, need_mamba_block_aligned_split=False,
        scheduler_reserve_full_isl=False, is_encoder_decoder=False,
        num_lookahead_tokens=0, log_stats=False,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[1]), use_v2_model_runner=False,
        use_pp=False, prev_step_scheduled_req_ids={'running'},
        needs_kv_cache_zeroing=False, finished_req_ids=set(),
        finished_recving_kv_req_ids=set(), encoder_cache_manager=_Encoder(),
        requests={'running':active},
    )
    for key, value in fields.items():
        setattr(scheduler, key, value)
    for i in range(waiting):
        req = _request(f'waiting{i}', priority=i)
        scheduler.waiting.add_request(req)
        scheduler.requests[req.request_id] = req
    return scheduler


@pytest.fixture(autouse=True)
def _install(monkeypatch):
    monkeypatch.setenv('JETFIRE_PREFILL_FIRST_SCHEDULING', '1')
    monkeypatch.setattr(Scheduler, 'schedule', _NATIVE_SCHEDULE)
    monkeypatch.setattr(Scheduler, '_jetfire_original_schedule', _NATIVE_SCHEDULE, raising=False)
    monkeypatch.setattr(registry, '_PATCHED_PREFILL_FIRST_SCHEDULER', False)
    registry._maybe_patch_prefill_first_scheduler()


def test_no_capacity_keeps_running_decode_progressing():
    scheduler = _scheduler(capacity=1)
    for _ in range(5):
        before = scheduler.requests['running'].num_computed_tokens
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {'running':1}
        assert scheduler.requests['running'].num_computed_tokens == before + 1
        scheduler.requests['running'].append_output_token_ids(2)
    assert scheduler.kv_cache_manager.steps == 5
    assert [r.request_id for r in scheduler.running] == ['running']
    assert scheduler.waiting.peek_request().status == RequestStatus.WAITING


def test_successful_prefill_retains_priority_and_running_order():
    scheduler = _scheduler(capacity=4)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {'waiting0':16}
    assert scheduler.requests['running'].num_computed_tokens == 16
    assert scheduler.requests['waiting0'].status == RequestStatus.RUNNING
    assert [r.request_id for r in scheduler.running] == ['running','waiting0']
    assert scheduler.max_num_running_reqs == 4
    assert scheduler.kv_cache_manager.steps == 1


def test_full_running_limit_does_not_overadmit():
    scheduler = _scheduler(capacity=4,limit=1,waiting=2)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {'running':1}
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 2
    assert scheduler.max_num_running_reqs == 1


@pytest.mark.parametrize('policy',[SchedulingPolicy.FCFS,SchedulingPolicy.PRIORITY])
def test_partial_running_limit_admits_only_remaining_slot(policy):
    scheduler = _scheduler(capacity=4,limit=2,waiting=2,policy=policy)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {'waiting0':16}
    assert [r.request_id for r in scheduler.running] == ['running','waiting0']
    assert [r.request_id for r in scheduler.waiting] == ['waiting1']
    assert scheduler.max_num_running_reqs == 2


@pytest.mark.parametrize('status',[
    RequestStatus.WAITING_FOR_REMOTE_KVS,
    RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
    RequestStatus.WAITING_FOR_STREAMING_REQ,
])
def test_blocked_waiting_status_does_not_starve_decode(status):
    scheduler = _scheduler(capacity=4)
    waiting = scheduler.waiting.peek_request()
    waiting.status = status
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {'running':1}
    assert waiting.status == status
    assert [r.request_id for r in scheduler.skipped_waiting] == ['waiting0']
    assert [r.request_id for r in scheduler.running] == ['running']


@pytest.mark.parametrize('queue_name',['waiting','skipped_waiting'])
def test_preempted_request_keeps_resumption_status_and_queue_order(queue_name):
    scheduler = _scheduler(capacity=4,limit=2,waiting=2)
    first = scheduler.waiting.peek_request()
    first.status = RequestStatus.PREEMPTED
    if queue_name == 'skipped_waiting':
        scheduler.skipped_waiting.add_request(scheduler.waiting.pop_request())
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {'waiting0':16}
    assert output.scheduled_cached_reqs.resumed_req_ids == {'waiting0'}
    assert first.status == RequestStatus.RUNNING
    assert [r.request_id for r in scheduler.running] == ['running','waiting0']
    assert [r.request_id for r in scheduler.waiting] == ['waiting1']


def test_capacity_fallback_preserves_notifications_and_previous_step_metadata():
    scheduler = _scheduler(capacity=1)
    scheduler.finished_req_ids = {'already-finished'}
    scheduler.needs_kv_cache_zeroing = True
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {'running':1}
    assert output.finished_req_ids == {'already-finished'}
    assert output.free_encoder_mm_hashes == ['finished-mm']
    assert output.new_block_ids_to_zero == [99]
    assert output.scheduled_cached_reqs.all_token_ids == {}
    assert scheduler.prev_step_scheduled_req_ids == {'running'}
    assert scheduler.finished_req_ids == set()
    assert scheduler.encoder_cache_manager.calls == 1
    assert scheduler.kv_cache_manager.zero_calls == 1
    assert scheduler.kv_cache_manager.steps == 1


def test_exception_during_deferred_prefill_restores_running_and_limit():
    scheduler = _scheduler(capacity=4)
    scheduler.kv_cache_manager.raise_on.add('waiting0')
    with pytest.raises(RuntimeError,match='allocation failure'):
        scheduler.schedule()
    assert [r.request_id for r in scheduler.running] == ['running']
    assert scheduler.max_num_running_reqs == 4
    assert scheduler.waiting.peek_request().status == RequestStatus.WAITING


def test_exception_after_partial_admission_preserves_native_state_once():
    scheduler = _scheduler(capacity=4,waiting=2)
    scheduler.kv_cache_manager.raise_on.add('waiting1')
    with pytest.raises(RuntimeError,match='allocation failure'):
        scheduler.schedule()
    assert [r.request_id for r in scheduler.running] == ['running','waiting0']
    assert scheduler.max_num_running_reqs == 4
    assert scheduler.requests['waiting0'].status == RequestStatus.RUNNING
    assert [r.request_id for r in scheduler.waiting] == ['waiting1']


def test_exception_in_native_capacity_fallback_does_not_duplicate_running():
    scheduler = _scheduler(capacity=1)
    scheduler.kv_cache_manager.raise_on.add('running')
    with pytest.raises(RuntimeError,match='allocation failure'):
        scheduler.schedule()
    assert [r.request_id for r in scheduler.running] == ['running']
    assert scheduler.max_num_running_reqs == 4


def test_capacity_queued_workload_finishes_every_request():
    scheduler = _scheduler(capacity=1,limit=4,waiting=3)
    completed = set()
    for _ in range(12):
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens > 0
        assert len(scheduler.running) <= scheduler.max_num_running_reqs
        for request_id in output.num_scheduled_tokens:
            request = scheduler.requests[request_id]
            request.append_output_token_ids(2)
            if request.num_output_tokens >= 2:
                request.status = RequestStatus.FINISHED_LENGTH_CAPPED
                scheduler.running.remove(request)
                scheduler.kv_cache_manager.resident.remove(request_id)
                scheduler.finished_req_ids.add(request_id)
                completed.add(request_id)
        if len(completed) == 4:
            break
    assert completed == {'running','waiting0','waiting1','waiting2'}
    assert not scheduler.running
    assert not scheduler.waiting
