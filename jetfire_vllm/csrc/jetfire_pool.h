// Custom thread pool for high-frequency, short-task CPU work in JetFire's
// decode path.  Replaces libgomp's `#pragma omp parallel for` to avoid the
// ~50 µs wake-up penalty its mutex+cv path pays per call.
//
// Design:
//   * N persistent worker threads, lazy-init on first dispatch.
//   * Two-phase wait per worker:
//       1. Spin on per-worker atomic `generation` (~30 µs default).
//       2. If still no work, FUTEX_WAIT on `futex_word` until woken.
//   * Per-worker direct dispatch — no team-wide broadcast or mutex
//     serialization.
//   * Release/acquire memory ordering on all atomics so the worker's
//     output writes are visible to the dispatcher after wait.
//
// Calling pattern:
//   struct Ctx { ... };
//   void worker_fn(int worker_id, int n_workers, void* ctx_void) {
//       auto* ctx = (Ctx*)ctx_void;
//       int64_t start = ctx->total * worker_id / n_workers;
//       int64_t end   = ctx->total * (worker_id + 1) / n_workers;
//       for (int64_t i = start; i < end; ++i) { ... }
//   }
//   Ctx ctx{...};
//   jetfire_pool::global_pool().dispatch_and_wait(&worker_fn, &ctx);

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace jetfire_pool {

using WorkerFn = void(*)(int worker_id, int n_workers, void* ctx);


// Cacheline-aligned to avoid false sharing between workers.
struct alignas(64) Worker {
    // Bumped by dispatcher when new work is ready.  Worker compares against
    // its private `last_gen` to detect new tasks.
    std::atomic<uint32_t> generation{0};
    // Bumped by worker after task completes.  Dispatcher waits for
    // done_gen == generation across all workers.
    std::atomic<uint32_t> done_gen{0};
    // Used for FUTEX_WAIT in deep idle.  0 = sleeping/about-to-sleep,
    // 1 = work pending.
    std::atomic<int>      futex_word{0};

    int  worker_id = 0;
    int  core_id   = -1;       // -1 = no pinning
    bool running   = true;
    std::thread thread;

    // Padding to cacheline size.
    char _pad[24];
};


class Pool {
public:
    Pool() = default;
    ~Pool() { if (n_ > 0) shutdown(); }
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    void init(int n_workers, bool pin_to_cores = false);
    void shutdown();
    void dispatch_and_wait(WorkerFn fn, void* ctx);
    // Split form: dispatch_async returns immediately after waking workers.
    // wait() spins-then-futexes on done_count_ for completion.  Caller
    // must invoke wait() before the next dispatch_async() / dispatch_and_wait()
    // (no concurrent in-flight jobs).  This is the entry point used by the
    // CUDA-graph submission path, which queues dispatch via cudaLaunchHostFunc
    // and normally lets a side-stream host callback perform wait() while the
    // main stream continues GPU work.
    void dispatch_async(WorkerFn fn, void* ctx);
    void wait();
    int  n() const { return n_; }
    bool is_initialized() const { return n_ > 0; }
    int  done_count() const { return done_count_.load(std::memory_order_acquire); }

private:
    void worker_loop(Worker* w);

    static long futex_wait_op(int* addr, int expected) {
        return syscall(SYS_futex, addr, FUTEX_WAIT_PRIVATE, expected,
                       nullptr, nullptr, 0);
    }
    static long futex_wait_op_timeout_us(int* addr, int expected,
                                         int timeout_us) {
        timespec ts;
        ts.tv_sec = timeout_us / 1000000;
        ts.tv_nsec = (timeout_us % 1000000) * 1000;
        return syscall(SYS_futex, addr, FUTEX_WAIT_PRIVATE, expected,
                       &ts, nullptr, 0);
    }
    static long futex_wake_op(int* addr, int n = 1) {
        return syscall(SYS_futex, addr, FUTEX_WAKE_PRIVATE, n,
                       nullptr, nullptr, 0);
    }

    int n_ = 0;
    std::vector<std::unique_ptr<Worker>> workers_;
    // Shared task state — dispatcher writes once, all workers read.
    std::atomic<WorkerFn> fn_{nullptr};
    std::atomic<void*>    ctx_{nullptr};
    // CUDA host callbacks may be issued from different callback threads.
    // The pool is intentionally single-flight, so publishing a new job must
    // be serialized with any other publisher that might reset done_count_ and
    // overwrite fn_/ctx_.
    std::mutex dispatch_mu_;
    // Completion barrier.  Reset to 0 by dispatcher at each dispatch; each
    // worker fetch_adds 1 after its task; the last worker (bringing the
    // count to n_) wakes the dispatcher via futex_wake.  Replaces the
    // per-worker busy-spin on `done_gen` that previously deadlocked at
    // long ctx + larger B when CPU was oversubscribed.
    alignas(64) std::atomic<int> done_count_{0};
};


inline void Pool::init(int n_workers, bool pin_to_cores) {
    n_ = n_workers;
    // No job is in flight immediately after init.  This matters for
    // dispatch_async(), which defensively waits for any previous job before it
    // publishes new task pointers.
    done_count_.store(n_, std::memory_order_release);
    workers_.reserve(n_);
    for (int i = 0; i < n_; ++i) {
        auto w = std::make_unique<Worker>();
        w->worker_id = i;
        w->core_id   = pin_to_cores ? i : -1;
        Worker* wp = w.get();
        w->thread = std::thread([this, wp]() {
            if (wp->core_id >= 0) {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(wp->core_id, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
            }
            this->worker_loop(wp);
        });
        workers_.push_back(std::move(w));
    }
}


inline void Pool::shutdown() {
    for (auto& w : workers_) {
        w->running = false;
        // Bump generation so spinning worker exits the wait loop, and wake
        // any worker that's currently in FUTEX_WAIT.
        w->generation.fetch_add(1, std::memory_order_acq_rel);
        w->futex_word.store(1, std::memory_order_release);
        int* fw = reinterpret_cast<int*>(&w->futex_word);
        futex_wake_op(fw);
    }
    for (auto& w : workers_) w->thread.join();
    workers_.clear();
    n_ = 0;
}


inline void Pool::worker_loop(Worker* w) {
    // Spin count: env JETFIRE_POOL_SPINCOUNT, default 100000 iterations of
    // _mm_pause (~30 µs at 3 GHz).  Covers the typical inter-call interval
    // without falling to futex.  futex_word semantics (eventcount):
    //   1 = awake / running / spinning (do NOT wake via syscall)
    //   0 = about to sleep / sleeping (DO wake via syscall)
    static const int SPIN_COUNT = []() {
        const char* s = std::getenv("JETFIRE_POOL_SPINCOUNT");
        return s ? std::atoi(s) : 100000;
    }();
    static const int FUTEX_TIMEOUT_US = []() {
        const char* s = std::getenv("JETFIRE_POOL_FUTEX_TIMEOUT_US");
        if (!s) return 1000;
        int v = std::atoi(s);
        return v < 0 ? 0 : v;
    }();

    uint32_t last_gen = 0;
    // Worker starts in "awake" state (futex_word=0 from init); set to 1 so
    // dispatcher's first exchange sees us as awake.
    w->futex_word.store(1, std::memory_order_release);

    while (true) {
        // ── Phase 1: spin (covers normal inter-call interval) ─────────
        bool got_work = false;
        for (int i = 0; i < SPIN_COUNT; ++i) {
            uint32_t gen = w->generation.load(std::memory_order_acquire);
            if (gen != last_gen) { got_work = true; break; }
            if (!w->running) return;
            __builtin_ia32_pause();
        }

        // ── Phase 2: futex sleep (deep idle) ──────────────────────────
        if (!got_work) {
            while (!got_work) {
                if (!w->running) return;
                // Atomically mark "sleeping" (futex_word=0).  Dispatcher's
                // exchange-to-1 will return 0 and trigger a futex_wake.
                w->futex_word.store(0, std::memory_order_release);
                // Race window: dispatcher may have bumped gen between
                // spin-loop exit and this store.  Re-check to avoid lost
                // wakeup.
                uint32_t gen = w->generation.load(std::memory_order_acquire);
                if (gen != last_gen) {
                    // Got work just before we marked sleeping — restore
                    // awake state and run.
                    w->futex_word.store(1, std::memory_order_release);
                    got_work = true;
                    break;
                }
                int* fw = reinterpret_cast<int*>(&w->futex_word);
                if (FUTEX_TIMEOUT_US > 0) {
                    // The eventcount protocol avoids wake syscalls on the
                    // hot path, but a stale generation read can otherwise
                    // turn a missed wake into a permanent one-worker stall.
                    // A short timeout preserves the fast wake path and bounds
                    // that rare race without switching to synchronous CPU
                    // correction.
                    futex_wait_op_timeout_us(fw, 0, FUTEX_TIMEOUT_US);
                } else {
                    futex_wait_op(fw, 0);  // sleeps until futex_word != 0
                }
                // Woken — re-check generation.
                gen = w->generation.load(std::memory_order_acquire);
                if (gen != last_gen) { got_work = true; break; }
                // Spurious wake — loop and re-sleep.
            }
        }

        if (!w->running) return;

        // ── Run the task ───────────────────────────────────────────────
        last_gen = w->generation.load(std::memory_order_acquire);
        WorkerFn fn = fn_.load(std::memory_order_acquire);
        void*    ctx = ctx_.load(std::memory_order_acquire);
        if (fn) fn(w->worker_id, n_, ctx);

        // Release-store so the dispatcher's acquire-load sees the worker's
        // output writes (e.g., to o_new buffer).
        w->done_gen.store(last_gen, std::memory_order_release);

        // Bump the completion barrier.  If we're the last worker for this
        // dispatch, wake the dispatcher if it's blocked in futex_wait.
        // (If it's still spinning the wake is a no-op.)
        int prev = done_count_.fetch_add(1, std::memory_order_acq_rel);
        if (prev + 1 >= n_) {
            int* dc = reinterpret_cast<int*>(&done_count_);
            futex_wake_op(dc);
        }
        // futex_word was set to 1 either at startup or by the dispatcher's
        // exchange.  We leave it at 1 here so the next dispatch_and_wait
        // sees us as awake and skips the syscall (since we'll be back in
        // the spin loop in <1 µs).
    }
}


inline void Pool::dispatch_async(WorkerFn fn, void* ctx) {
    std::lock_guard<std::mutex> lock(dispatch_mu_);

    // The async API is intentionally single-flight.  CUDA graph replay should
    // reach the next host callback only after the previous layer's stream wait,
    // but prefix-cache / continuous-batching replays can stress this invariant.
    // If a callback arrives early, serialize here instead of resetting
    // done_count_ and overwriting fn_/ctx_ while workers still use them.
    if (done_count_.load(std::memory_order_acquire) < n_) {
        wait();
    }

    // Reset completion barrier before publishing work.  Safe because the
    // previous dispatch's wait() (or whatever signals completion) ensured
    // no in-flight increments from prior work remain.
    done_count_.store(0, std::memory_order_release);

    // Publish task before bumping generation: a worker observing the new
    // generation must also see the matching fn/ctx (release-acquire pair).
    fn_.store(fn,  std::memory_order_release);
    ctx_.store(ctx, std::memory_order_release);

    // Bump generation + selectively wake.
    for (auto& w : workers_) {
        w->generation.fetch_add(1, std::memory_order_acq_rel);
        // Eventcount: exchange returns the previous futex_word.
        //   prev == 1 → worker is awake / spinning → no syscall (just
        //               atomic store, ~5 ns)
        //   prev == 0 → worker is asleep on futex_wait → wake it
        int prev = w->futex_word.exchange(1, std::memory_order_acq_rel);
        if (prev == 0) {
            int* fw = reinterpret_cast<int*>(&w->futex_word);
            futex_wake_op(fw);
        }
    }
}


inline void Pool::wait() {
    // Wait phase.  Spin first (covers normal 150-1000 µs work latency
    // without falling to a syscall), then futex_wait.
    static const int WAIT_SPIN = []() {
        const char* s = std::getenv("JETFIRE_POOL_DISPATCH_SPIN");
        return s ? std::atoi(s) : 200000;  // ~60 µs of `pause` at 3 GHz
    }();
    int* dc = reinterpret_cast<int*>(&done_count_);
    for (int i = 0; i < WAIT_SPIN; ++i) {
        if (done_count_.load(std::memory_order_acquire) >= n_) return;
        __builtin_ia32_pause();
    }
    // Slow path: block until the last worker stores n_ and wakes us.
    // Re-read each loop to handle EAGAIN (value changed between load and
    // syscall), spurious wakes from non-last workers, and EINTR.
    while (true) {
        int cur = done_count_.load(std::memory_order_acquire);
        if (cur >= n_) return;
        futex_wait_op(dc, cur);
    }
}


inline void Pool::dispatch_and_wait(WorkerFn fn, void* ctx) {
    dispatch_async(fn, ctx);
    wait();
}


// Singleton pool, lazy-initialized on first call.  Lifetime tied to
// process; threads exit at process teardown.
Pool& global_pool();

}  // namespace jetfire_pool
