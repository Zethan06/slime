"""Single-process metric-report invariance tests.

Pins train-side / rollout-side report formulas implemented in
``slime.observability.train_metric_utils``: the reported number for a given set
of samples must be the same regardless of

  - how samples are distributed across micro-batches / DP ranks
  - whether context parallelism is on or off
  - whether the path is per-rollout-mean or per-token-loss

Single-process variants use a mock dp-with-cp group + a no-op
``dist.all_reduce`` to keep things lightweight; the multi-process
end-to-end variants (real torch.distributed) live in
``test_metric_report_dist.py``.
"""

from __future__ import annotations

# Import the helpers BEFORE the slime imports so the megatron stub lands
# in sys.modules first. pytest's prepend importmode puts this file's
# directory (``tests/``) on sys.path, which is what makes the bare-name
# import work without an ``__init__.py``.
import _cp_dist_helpers  # noqa: F401
import pytest
import torch

from slime.backends.megatron_utils.cp_utils import (  # noqa: E402
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
)
from slime.observability.train_metric_utils import (  # noqa: E402
    reduce_train_step_metrics,
    rollout_log_metric_contribution,
)

NUM_GPUS = 0


@pytest.fixture
def mock_dp_with_cp_group(monkeypatch):
    """A sentinel "process group" object plus a no-op ``dist.all_reduce``.

    Lets the train-step report tests exercise the production call shape
    (``dist.all_reduce(values, group=dp_with_cp_group)``) without standing
    up a real torch.distributed runtime. The test itself simulates cross-
    rank summation in pure Python and feeds the already-summed tensor in;
    the no-op all_reduce leaves it untouched.
    """
    import torch.distributed as dist

    monkeypatch.setattr(dist, "all_reduce", lambda tensor, group=None, op=None: None)
    return object()  # opaque sentinel — only used as the ``group`` argument


# ---------------------------------------------------------------------------
# Mirrors the actual train_one_step reporting math:
#
#   per-rollout-mean path:
#       reported = sum_of_reducer_per_mb / step_global_batch_size
#   per-token-loss path:
#       reported = sum_of_reducer_per_mb / sum_of_per_mb_num_tokens
#
# The reducer is the same callable used at train time (and inside
# log_rollout_data on the rollout side).
# ---------------------------------------------------------------------------


# 4 samples: rollout R0 owns indices 0,1,2 (mask sums 3+3+3=9); rollout R1
# owns index 3 (mask sum 3). Pre-computed per-sample denom = group sum.
# Per-rollout-mean: R0 = 5, R1 = 11, sum = 16, divided by 2 rollouts → 8.
# Per-token-loss:   sum of all x = 78, total clamped mask = 12, → 6.5.
_FIXED_RESPONSE_LENGTHS = [3, 3, 3, 3]
_FIXED_TOTAL_LENGTHS = [r + 4 for r in _FIXED_RESPONSE_LENGTHS]
_FIXED_LOSS_MASKS = [torch.ones(r, dtype=torch.float32) for r in _FIXED_RESPONSE_LENGTHS]
_FIXED_ROLLOUT_DENOMS = [9.0, 9.0, 9.0, 3.0]
_FIXED_X_PER_SAMPLE = [
    torch.tensor([1.0, 2.0, 3.0]),
    torch.tensor([4.0, 5.0, 6.0]),
    torch.tensor([7.0, 8.0, 9.0]),
    torch.tensor([10.0, 11.0, 12.0]),
]
_FIXED_STEP_GBS = 2  # 2 distinct rollouts in the step
_EXPECTED_PER_ROLLOUT_MEAN_REPORT = 8.0
_EXPECTED_PER_TOKEN_LOSS_REPORT = 78.0 / 12.0


# Each entry: list of "rank"s, each rank is a list of mbs, each mb is the
# sample-index list packed into that mb. Covers: single mb, evenly split by
# rollout, split inside a rollout (R0 across mbs), uneven distribution, and
# fully singleton mbs per rank.
_PARTITION_CONFIGS = [
    [[[0, 1, 2, 3]]],  # 1 rank, 1 mb
    [[[0, 1, 2], [3]]],  # 1 rank, 2 mbs split at rollout boundary
    [[[0, 1], [2, 3]]],  # 1 rank, 2 mbs splitting R0 across them — the tricky case
    [[[0, 1]], [[2, 3]]],  # 2 ranks, 1 mb each
    [[[0, 1, 3]], [[2]]],  # 2 ranks, R0 split across BOTH ranks (worst case for split-across-mb bug)
    [[[0]], [[1]], [[2]], [[3]]],  # 4 ranks, 1 sample per rank
]


def _simulate_report(partition, *, per_token_loss: bool) -> float:
    """Reproduce train_one_step's reporting math for one partition config."""
    metric_sum = 0.0
    num_tokens_sum = 0
    for rank_mbs in partition:
        for mb_indices in rank_mbs:
            mb_total = [_FIXED_TOTAL_LENGTHS[i] for i in mb_indices]
            mb_resp = [_FIXED_RESPONSE_LENGTHS[i] for i in mb_indices]
            mb_masks = [_FIXED_LOSS_MASKS[i] for i in mb_indices]
            mb_x = torch.cat([_FIXED_X_PER_SAMPLE[i] for i in mb_indices])
            if per_token_loss:
                # Per-token-loss: caller uses ``calculate_per_token_loss=True``
                # to get ``sum_of_token`` (no per-sample denom).
                reducer = get_sum_of_sample_mean(mb_total, mb_resp, mb_masks, calculate_per_token_loss=True)
                num_tokens_sum += sum(max(int(m.sum().item()), 1) for m in mb_masks)
            else:
                mb_denoms = torch.tensor([_FIXED_ROLLOUT_DENOMS[i] for i in mb_indices], dtype=torch.float32)
                reducer = get_sum_of_sample_mean(mb_total, mb_resp, mb_masks, mb_denoms)
            metric_sum += reducer(mb_x).item()
    if per_token_loss:
        return metric_sum / num_tokens_sum
    return metric_sum / _FIXED_STEP_GBS


@pytest.mark.unit
@pytest.mark.parametrize("partition", _PARTITION_CONFIGS)
def test_per_rollout_mean_report_invariant_to_mb_distribution(partition):
    """Same samples should yield the same per-rollout-mean report regardless of
    how they're spread across DP ranks / micro-batches — this is what lets us
    change parallelism without changing wandb numbers."""
    assert _simulate_report(partition, per_token_loss=False) == pytest.approx(_EXPECTED_PER_ROLLOUT_MEAN_REPORT)


@pytest.mark.unit
@pytest.mark.parametrize("partition", _PARTITION_CONFIGS)
def test_per_token_loss_report_invariant_to_mb_distribution(partition):
    """Same invariant for the per-token-loss reporting path."""
    assert _simulate_report(partition, per_token_loss=True) == pytest.approx(_EXPECTED_PER_TOKEN_LOSS_REPORT)


def _simulate_rollout_report(samples_per_rank):
    """Reproduce log_rollout_data + gather_log_data's averaging math for the
    per-token metric branch.

    Each "rank" applies the reducer once over its full sample subset, then
    ``rollout_log_metric_contribution`` (the same helper the reporter uses) emits
    the ``(per_rank_sum, count)`` tuple. We aggregate via
    ``Σsum / Σcount`` — the same shape ``gather_log_data`` uses.
    """
    dp_size = len(samples_per_rank)
    pairs: list[tuple[float, float]] = []
    for indices in samples_per_rank:
        if not indices:
            pairs.append(
                rollout_log_metric_contribution(
                    0.0, cp_size=1, num_rollouts_in_rollout=_FIXED_STEP_GBS, dp_size=dp_size
                )
            )
            continue
        tl = [_FIXED_TOTAL_LENGTHS[i] for i in indices]
        rl = [_FIXED_RESPONSE_LENGTHS[i] for i in indices]
        masks = [_FIXED_LOSS_MASKS[i] for i in indices]
        denoms = torch.tensor([_FIXED_ROLLOUT_DENOMS[i] for i in indices], dtype=torch.float32)
        x = torch.cat([_FIXED_X_PER_SAMPLE[i] for i in indices])
        reducer = get_sum_of_sample_mean(tl, rl, masks, denoms)
        pairs.append(
            rollout_log_metric_contribution(
                reducer(x).item(),
                cp_size=1,
                num_rollouts_in_rollout=_FIXED_STEP_GBS,
                dp_size=dp_size,
            )
        )
    total_sum = sum(p[0] for p in pairs)
    total_count = sum(p[1] for p in pairs)
    return total_sum / total_count


_DP_PARTITIONS = [
    [[0, 1, 2, 3]],  # 1 rank holds everything
    [[0, 1, 2], [3]],  # 2 ranks, balanced by rollout
    [[0, 1], [2, 3]],  # 2 ranks splitting R0 across mb-and-rank
    [[0, 1, 3], [2]],  # 2 ranks with R0 spread across BOTH (one of R0's samples is on rank 1)
    [[0], [1], [2], [3]],  # 4 ranks, one sample each (R0's samples spread across 3 ranks)
]


@pytest.mark.unit
@pytest.mark.parametrize("dp_partition", _DP_PARTITIONS)
def test_rollout_report_matches_train_report_in_single_step(dp_partition):
    """In a 1-step rollout, the rollout-side report (log_rollout_data → gather)
    must equal the train-side report (train_one_step ``value / step_global_batch_size``)
    for the same samples — otherwise wandb numbers between phases drift.

    Both go through the same reducer with the same precomputed denominators;
    the contract this test pins is that the gather count plumbing on the
    rollout side sums to the same denominator the train side uses
    (``step_global_batch_size``), independent of how the rollout's samples
    are spread across DP ranks.
    """
    rollout_report = _simulate_rollout_report(dp_partition)
    assert rollout_report == pytest.approx(_EXPECTED_PER_ROLLOUT_MEAN_REPORT)


@pytest.mark.unit
def test_train_one_step_per_rollout_mean_report_invariant_to_cp(monkeypatch, mock_dp_with_cp_group):
    """End-to-end check of train_one_step's report formula across CP sizes.

    Mirrors the actual reduction order:
      1. Each (DP, CP) rank computes per-mb reducer output.
      2. Per-rank values are summed across mbs locally.
      3. All-reduce sums across DP*CP ranks.
      4. ``reduce_train_step_metrics`` applied (the same helper
         ``train_one_step`` calls, so this test stays honest if the
         implementation changes).

    cp_size = 1 vs cp_size = 2 must give the same reported number —
    otherwise wandb metrics would drift the moment a user enables CP.
    """
    from megatron.core import mpu as _mpu

    total_lengths = [12, 12]
    response_lengths = [8, 8]
    loss_masks = [torch.ones(r, dtype=torch.float32) for r in response_lengths]
    sample_denoms = torch.tensor([16.0, 16.0], dtype=torch.float32)
    x_full = [
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
    ]
    step_global_batch_size = 1  # one rollout in the step

    def simulate(cp_size: int) -> float:
        monkeypatch.setattr(_mpu, "get_context_parallel_world_size", lambda: cp_size)
        monkeypatch.setattr(_mpu, "get_context_parallel_group", lambda: None, raising=False)
        # values[0] is the per-token-loss path's num_tokens slot; for
        # per-rollout-mean it's a zero placeholder (loss_function sets 0).
        value_after_allreduce = 0.0
        for cp_rank in range(cp_size):
            monkeypatch.setattr(_mpu, "get_context_parallel_rank", lambda r=cp_rank: r)
            if cp_size == 1:
                x_for_rank = torch.cat(x_full)
            else:
                x_chunks_per_sample = []
                for tl, rl, x in zip(total_lengths, response_lengths, x_full, strict=True):
                    prompt_length = tl - rl
                    _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(tl, rl)
                    c0 = x[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
                    c1 = x[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
                    x_chunks_per_sample.append(torch.cat([c0, c1]))
                x_for_rank = torch.cat(x_chunks_per_sample)
            reducer = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, sample_denoms)
            value_after_allreduce += reducer(x_for_rank).item()
        reduced = reduce_train_step_metrics(
            [{"keys": ["metric"], "values": torch.tensor([0.0, value_after_allreduce])}],
            calculate_per_token_loss=False,
            step_global_batch_size=step_global_batch_size,
            cp_size=cp_size,
            dp_with_cp_group=mock_dp_with_cp_group,
        )
        return reduced["metric"]

    assert simulate(1) == pytest.approx(simulate(2))


@pytest.mark.unit
def test_train_one_step_per_token_loss_report_invariant_to_cp(monkeypatch, mock_dp_with_cp_group):
    """Same end-to-end check for the per-token-loss path: divisor is
    ``values[0] = num_tokens`` (computed in loss.py from FULL loss masks),
    which each CP rank duplicates and all-reduce sums by ``cp_size``. The
    ``cp_factor = cp_size`` multiplier inside ``reduce_train_step_metrics``
    cancels that inflation, so the report stays CP-invariant.
    """
    from megatron.core import mpu as _mpu

    total_lengths = [12, 12]
    response_lengths = [8, 8]
    loss_masks = [torch.ones(r, dtype=torch.float32) for r in response_lengths]
    num_tokens_per_mb = sum(int(m.sum().item()) for m in loss_masks)  # = 16
    x_full = [
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
    ]

    def simulate(cp_size: int) -> float:
        monkeypatch.setattr(_mpu, "get_context_parallel_world_size", lambda: cp_size)
        monkeypatch.setattr(_mpu, "get_context_parallel_group", lambda: None, raising=False)
        value_after_allreduce = 0.0
        num_tokens_after_allreduce = 0  # each CP rank reports the same num_tokens
        for cp_rank in range(cp_size):
            monkeypatch.setattr(_mpu, "get_context_parallel_rank", lambda r=cp_rank: r)
            if cp_size == 1:
                x_for_rank = torch.cat(x_full)
            else:
                x_chunks_per_sample = []
                for tl, rl, x in zip(total_lengths, response_lengths, x_full, strict=True):
                    prompt_length = tl - rl
                    _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(tl, rl)
                    c0 = x[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
                    c1 = x[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
                    x_chunks_per_sample.append(torch.cat([c0, c1]))
                x_for_rank = torch.cat(x_chunks_per_sample)
            reducer = get_sum_of_sample_mean(
                total_lengths, response_lengths, loss_masks, calculate_per_token_loss=True
            )
            value_after_allreduce += reducer(x_for_rank).item()
            num_tokens_after_allreduce += num_tokens_per_mb
        reduced = reduce_train_step_metrics(
            [
                {
                    "keys": ["metric"],
                    "values": torch.tensor([num_tokens_after_allreduce, value_after_allreduce], dtype=torch.float32),
                }
            ],
            calculate_per_token_loss=True,
            step_global_batch_size=999,  # unused in per-token-loss path
            cp_size=cp_size,
            dp_with_cp_group=mock_dp_with_cp_group,
        )
        return reduced["metric"]

    assert simulate(1) == pytest.approx(simulate(2))


@pytest.mark.unit
def test_train_one_step_per_token_loss_report_under_dynamic_cp(mock_dp_with_cp_group):
    """Per-token-loss report under dynamic CP: ``scale_per_token_loss`` counts only
    the loss tokens in each rank's CP chunk, so ``values[0]`` already sums to the
    exact token count and ``train_one_step`` passes ``cp_size=1``. A layout mixing a
    2-rank CP block with single-rank micro-batches must report the CP=1 token mean.
    """
    from argparse import Namespace

    from slime.backends.megatron_utils.cp_utils import CPContext, slice_log_prob_with_cp
    from slime.backends.megatron_utils.loss import scale_per_token_loss

    # (total_length, response_length, x, loss_mask). Sample 1's masked-out head
    # leaves its rank-0 chunk of the 2-rank block without any loss token.
    samples = [
        (9, 6, torch.arange(1.0, 7.0), torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.int)),
        (13, 10, torch.arange(10.0, 110.0, 10.0), torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 0, 1], dtype=torch.int)),
        (7, 4, torch.arange(100.0, 500.0, 100.0), torch.tensor([1, 0, 1, 1], dtype=torch.int)),
    ]
    rank_micro_batches = [
        [[(0, CPContext(1, 0, None))], [(1, CPContext(2, 0, None))]],
        [[(2, CPContext(1, 0, None))], [(1, CPContext(2, 1, None))]],
    ]

    values_after_allreduce = torch.zeros(2)
    chunk_num_tokens = []
    for micro_batches in rank_micro_batches:
        for micro_batch in micro_batches:
            cp = micro_batch[0][1]
            total_lengths = [samples[i][0] for i, _ in micro_batch]
            response_lengths = [samples[i][1] for i, _ in micro_batch]
            loss_masks = [samples[i][3] for i, _ in micro_batch]
            x = torch.cat(
                [slice_log_prob_with_cp(samples[i][2], samples[i][0], samples[i][1], cp=cp) for i, _ in micro_batch]
            )
            loss = get_sum_of_sample_mean(
                total_lengths, response_lengths, loss_masks, calculate_per_token_loss=True, cps=[cp] * len(micro_batch)
            )(x)
            batch = {
                "total_lengths": total_lengths,
                "response_lengths": response_lengths,
                "loss_masks": loss_masks,
                "cp": cp,
            }
            scaled_loss, num_tokens = scale_per_token_loss(Namespace(dynamic_cp=True), batch, loss)
            assert scaled_loss.item() == loss.item()  # no CP factor under dynamic CP
            chunk_num_tokens.append(int(num_tokens))
            values_after_allreduce += torch.tensor([num_tokens, loss])
    assert 0 in chunk_num_tokens

    reduced = reduce_train_step_metrics(
        [{"keys": ["metric"], "values": values_after_allreduce}],
        calculate_per_token_loss=True,
        step_global_batch_size=999,  # unused in per-token-loss path
        cp_size=1,
        dp_with_cp_group=mock_dp_with_cp_group,
    )
    token_sum = sum((x * mask).sum().item() for _, _, x, mask in samples)
    token_count = sum(mask.sum().item() for *_, mask in samples)
    assert reduced["metric"] == pytest.approx(token_sum / token_count)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


# ---------------------------------------------------------------------------
# log_rollout_data: token-level and per-sample scalar metrics must report the
# same number under CP=1, static CP and dynamic CP (where a sample is replicated
# over a CP block whose size differs from sample to sample).
# ---------------------------------------------------------------------------


def _rollout_report(monkeypatch, rank_layouts, samples):
    """Run log_rollout_data once per simulated rank and reduce like gather_log_data.

    ``rank_layouts[r]`` = list of ``(sample_id, CPContext)`` the rank holds.
    """
    from argparse import Namespace

    from megatron.core import mpu

    from slime.backends.megatron_utils import cp_utils
    from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp
    from slime.observability import train_metric_utils

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda **_kwargs: True, raising=False)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_data_parallel_world_size", lambda with_context_parallel=False: 1, raising=False)
    captured = []
    monkeypatch.setattr(train_metric_utils, "gather_log_data", lambda *args: captured.append(args[3]))
    args = Namespace(ci_test=False, log_multi_turn=False, log_passrate=False, log_correct_samples=False)

    for layout in rank_layouts:
        cps = [cp for _, cp in layout]
        monkeypatch.setattr(cp_utils, "get_sample_cp_contexts", lambda _rollout_data, cps=cps: cps)
        rollout_data = {"response_lengths": [], "total_lengths": [], "loss_masks": [], "entropy": [], "rewards": []}
        for sample_id, cp in layout:
            total, response, entropy, reward = samples[sample_id]
            rollout_data["response_lengths"].append(response)
            rollout_data["total_lengths"].append(total)
            rollout_data["loss_masks"].append(torch.ones(response))
            rollout_data["entropy"].append(
                entropy if cp.size == 1 else slice_log_prob_with_cp(entropy, total, response, cp=cp)
            )
            rollout_data["rewards"].append(reward)
        rollout_data["global_batch_sizes"] = [len(samples)]
        train_metric_utils.log_rollout_data(0, args, rollout_data)

    return {
        key: sum(log_dict[key][0] for log_dict in captured) / sum(log_dict[key][1] for log_dict in captured)
        for key in ("entropy", "rewards")
    }


@pytest.mark.unit
def test_rollout_report_token_and_scalar_metrics_invariant_to_cp_layout(monkeypatch):
    from slime.backends.megatron_utils.cp_utils import CPContext

    generator = torch.Generator().manual_seed(0)
    samples = [
        (total, response, torch.rand(response, generator=generator), reward)
        for total, response, reward in [(9, 6, 1.0), (13, 10, 0.0), (7, 4, 0.5)]
    ]
    expected = {
        "entropy": torch.cat([entropy for _, _, entropy, _ in samples]).mean().item(),
        "rewards": sum(reward for *_, reward in samples) / len(samples),
    }

    def single(rank=0):
        return CPContext(1, rank, None)

    layouts = {
        "cp1": [[(0, single()), (1, single()), (2, single())]],
        "static_cp2": [[(i, CPContext(2, r, None)) for i in range(3)] for r in range(2)],
        # Dynamic CP over 2 ranks: sample 1 runs on a 2-rank block, samples 0 and 2 on
        # one rank each.
        "dynamic_cp": [
            [(0, single()), (1, CPContext(2, 0, None))],
            [(2, single()), (1, CPContext(2, 1, None))],
        ],
    }
    for name, rank_layouts in layouts.items():
        reported = _rollout_report(monkeypatch, rank_layouts, samples)
        for key, value in expected.items():
            assert reported[key] == pytest.approx(value), f"{name}: {key}"
