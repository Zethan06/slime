"""CPU unit tests for slime.utils.dp_schedule.build_dp_schedule and build_dynamic_cp_schedule.

The tests assert the invariants documented at the top of dp_schedule.py against
a range of static / dynamic / VPP / oversize / balance / uneven scenarios.
"""

import random
from types import SimpleNamespace

import pytest

from slime.utils.dp_schedule import build_dp_schedule, build_dynamic_cp_schedule


NUM_GPUS = 0


def make_args(
    *,
    micro_batch_size=1,
    use_dynamic_batch_size=False,
    max_tokens_per_gpu=None,
    balance_data=False,
    balance_by_flops=False,
):
    return SimpleNamespace(
        micro_batch_size=micro_batch_size,
        use_dynamic_batch_size=use_dynamic_batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
        balance_data=balance_data,
        balance_by_flops=balance_by_flops,
        hidden_size=16,
        num_attention_heads=2,
        num_query_groups=2,
        vocab_size=32,
        ffn_hidden_size=64,
        num_experts=None,
        num_layers=2,
        kv_channels=8,
    )


def make_tp(dp_size=1, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1):
    return {
        "dp_size": dp_size,
        "cp_size": cp_size,
        "vpp_size": vpp_size,
        "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage,
    }


def assert_invariants(
    partitions,
    micro_batch_indices,
    num_microbatches,
    *,
    dp_size,
    expected_global_sample_indices,
    total_lengths,
    max_per_bin=None,
):
    """Check the invariants documented at the top of dp_schedule.py.

    ``expected_global_sample_indices`` is the set of global sample indices
    that should end up covered (after trim). Trailing rollouts that don't
    fit are excluded.
    """
    seen_global: set[int] = set()
    for r in range(dp_size):
        partition = partitions[r]
        mbi = micro_batch_indices[r]

        # Same num_mbs per rank (PP sync).
        assert len(mbi) == sum(num_microbatches), f"rank {r}: mbs count mismatch"

        # Flattened micro_batch_indices == range(len(partition)).
        flat = [i for mbs in mbi for i in mbs]
        assert flat == list(range(len(partition))), f"rank {r}: micro_batch_indices don't tile [0, n)"

        # Disjoint partitions whose union covers every kept sample.
        assert seen_global.isdisjoint(partition), f"rank {r}: overlap with other ranks"
        seen_global.update(partition)
    assert seen_global == set(expected_global_sample_indices), "covered sample set mismatch"

    if max_per_bin is None:
        return

    # Every mbs <= max_per_bin tokens, EXCEPT a singleton bin holding an oversized sample.
    for r in range(dp_size):
        partition = partitions[r]
        for mbs in micro_batch_indices[r]:
            bin_total = sum(total_lengths[partition[i]] for i in mbs)
            if bin_total > max_per_bin:
                assert len(mbs) == 1, f"rank {r}: mbs sum {bin_total} > {max_per_bin} but contains {len(mbs)} samples"


@pytest.mark.unit
def test_static_stride_single_step():
    """Static + strided DP split, single step (1 rollout = 1 sample)."""
    total_lengths = [10] * 16
    rollout_indices = list(range(16))
    args = make_args(micro_batch_size=2)
    tp = make_tp(dp_size=4)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=16, rollout_indices=rollout_indices
    )

    assert nmb == [2]
    assert gbs_per_step == [16]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=4,
        expected_global_sample_indices=range(16),
        total_lengths=total_lengths,
    )


@pytest.mark.unit
def test_static_balance_multi_step():
    """Static + balance_data + 2 training steps."""
    total_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1]
    rollout_indices = list(range(16))
    args = make_args(micro_batch_size=2, balance_data=True)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=8, rollout_indices=rollout_indices
    )

    assert nmb == [2, 2]
    assert gbs_per_step == [8, 8]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(16),
        total_lengths=total_lengths,
    )


@pytest.mark.unit
def test_balance_data_distributes_by_flops():
    """balance_data uses FLOPs weights for rank assignment, not raw token sums."""
    total_lengths = [1, 2, 3, 4, 5, 7, 9, 10]
    rollout_indices = list(range(8))
    args = make_args(micro_batch_size=1, balance_data=True)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=8, rollout_indices=rollout_indices
    )

    assert partitions == [[1, 2, 5, 6], [0, 3, 4, 7]]
    assert nmb == [4]
    assert gbs_per_step == [8]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(8),
        total_lengths=total_lengths,
    )


@pytest.mark.unit
def test_dynamic_uniform():
    """Dynamic mbs on uniform-length samples."""
    total_lengths = [5] * 8
    rollout_indices = list(range(8))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=8, rollout_indices=rollout_indices
    )

    assert gbs_per_step == [8]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(8),
        total_lengths=total_lengths,
        max_per_bin=10,
    )


@pytest.mark.unit
def test_dynamic_oversized_sample_lands_alone():
    """A sample larger than max_per_bin must end up alone in its mbs."""
    total_lengths = [15, 3, 3, 3, 3, 3, 3, 3]
    rollout_indices = list(range(8))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=8, rollout_indices=rollout_indices
    )

    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(8),
        total_lengths=total_lengths,
        max_per_bin=10,
    )
    oversize_idx = total_lengths.index(15)
    found = False
    for r in range(2):
        if oversize_idx not in partitions[r]:
            continue
        local = partitions[r].index(oversize_idx)
        for mbs in mbi[r]:
            if local in mbs:
                assert mbs == [local], f"oversized sample shares an mbs: {mbs}"
                found = True
    assert found


@pytest.mark.unit
def test_dynamic_with_vpp_rounds_to_mb_group():
    """num_microbatches per rank should be a multiple of mb_group when vpp_size > 1."""
    total_lengths = [4] * 32
    rollout_indices = list(range(32))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=8)
    tp = make_tp(dp_size=2, vpp_size=2, microbatch_group_size_per_vp_stage=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=16, rollout_indices=rollout_indices
    )

    for n in nmb:
        assert n % 2 == 0, f"num_microbatches {n} is not a multiple of mb_group=2"
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(32),
        total_lengths=total_lengths,
        max_per_bin=8,
    )


@pytest.mark.unit
def test_rollout_grouping_keeps_samples_together():
    """compact / subagent simulation: rollout 0 emits 3 samples, rollout 1 emits 2,
    rollout 2 emits 4. Splitter keeps every rollout's samples in a single step."""
    rollout_indices = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    total_lengths = [3] * 9
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=12)
    tp = make_tp(dp_size=1)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=1, rollout_indices=rollout_indices
    )

    # 3 rollouts / 1 per step → 3 steps, gbs constant.
    assert gbs_per_step == [1, 1, 1]
    # For each step, collect the samples (global indices) that landed in that step's mbs
    # on rank 0, then verify they exactly equal the rollout's sample positions.
    expected_per_step = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
    rank0_partition = partitions[0]
    mbs_cursor = 0
    for step_i, n_mbs in enumerate(nmb):
        step_locals = sorted(j for mbs in mbi[0][mbs_cursor : mbs_cursor + n_mbs] for j in mbs)
        step_globals = [rank0_partition[j] for j in step_locals]
        assert (
            sorted(step_globals) == expected_per_step[step_i]
        ), f"step {step_i} samples = {step_globals}, expected {expected_per_step[step_i]}"
        mbs_cursor += n_mbs
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=1,
        expected_global_sample_indices=range(9),
        total_lengths=total_lengths,
        max_per_bin=12,
    )


@pytest.mark.unit
def test_trims_trailing_rollouts_that_dont_fill_a_step():
    """5 rollouts, gbs=2 → 2 steps × 2 rollouts; trailing rollout 4 (sample positions 6, 7)
    is dropped."""
    rollout_indices = [0, 0, 1, 2, 2, 3, 4, 4]
    total_lengths = [3] * 8
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=12)
    tp = make_tp(dp_size=1)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=2, rollout_indices=rollout_indices
    )

    assert gbs_per_step == [2, 2]
    # Sample positions 6 and 7 belong to the trimmed rollout 4 and must be absent.
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=1,
        expected_global_sample_indices=range(6),
        total_lengths=total_lengths,
        max_per_bin=12,
    )


@pytest.mark.unit
def test_rejects_when_fewer_rollouts_than_gbs():
    """gbs=4 with only 3 distinct rollouts → cannot form one step."""
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=12)
    tp = make_tp(dp_size=1)
    with pytest.raises(AssertionError, match="num_rollouts"):
        build_dp_schedule(args, tp, [3] * 6, global_batch_size=4, rollout_indices=[0, 0, 1, 1, 2, 2])



def assert_dcp_invariants(
    partitions,
    micro_batch_indices,
    micro_batch_cp_sizes,
    num_microbatches,
    *,
    world_size,
    expected_global_sample_indices,
    total_lengths,
    max_tokens_per_gpu,
):
    """Check the DCP round invariants documented on build_dynamic_cp_schedule."""
    num_rounds = sum(num_microbatches)
    for r in range(world_size):
        # Same round count per rank.
        assert len(micro_batch_indices[r]) == num_rounds, f"rank {r}: round count mismatch"
        assert len(micro_batch_cp_sizes[r]) == num_rounds, f"rank {r}: cp size count mismatch"
        flat = [i for mbs in micro_batch_indices[r] for i in mbs]
        assert flat == list(range(len(partitions[r]))), f"rank {r}: micro_batch_indices don't tile [0, n)"

    scheduled: list[int] = []
    for j in range(num_rounds):
        rank_samples = [[partitions[r][i] for i in micro_batch_indices[r][j]] for r in range(world_size)]
        rank = 0
        while rank < world_size:
            cp_size = micro_batch_cp_sizes[rank][j]
            # Aligned, contiguous power-of-2 block covering [rank, rank + cp_size).
            assert cp_size >= 1 and cp_size & (cp_size - 1) == 0, f"round {j}: cp_size {cp_size} not a power of 2"
            assert rank % cp_size == 0, f"round {j}: block at {rank} not aligned to {cp_size}"
            assert rank + cp_size <= world_size, f"round {j}: block at {rank} overflows world_size"
            block = range(rank, rank + cp_size)
            assert all(micro_batch_cp_sizes[b][j] == cp_size for b in block), f"round {j}: cp size differs in block"
            assert all(rank_samples[b] == rank_samples[rank] for b in block), f"round {j}: samples differ in block"
            assert rank_samples[rank], f"round {j}: block at {rank} is idle"
            unit_tokens = sum(total_lengths[i] for i in rank_samples[rank])
            assert unit_tokens <= cp_size * max_tokens_per_gpu, f"round {j}: {unit_tokens} tokens > cp {cp_size} cap"
            scheduled.extend(rank_samples[rank])
            rank += cp_size

    # Every kept sample scheduled exactly once (each block counted once).
    assert sorted(scheduled) == sorted(expected_global_sample_indices), "scheduled sample set mismatch"


def run_dcp(total_lengths, *, dp_size, cp_size, max_tokens_per_gpu, global_batch_size=None):
    rollout_indices = list(range(len(total_lengths)))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=max_tokens_per_gpu)
    tp = make_tp(dp_size=dp_size, cp_size=cp_size)
    partitions, mbi, cp_sizes, nmb, gbs_per_step = build_dynamic_cp_schedule(
        args,
        tp,
        total_lengths,
        global_batch_size=global_batch_size or len(total_lengths),
        rollout_indices=rollout_indices,
    )
    assert_dcp_invariants(
        partitions,
        mbi,
        cp_sizes,
        nmb,
        world_size=dp_size * cp_size,
        expected_global_sample_indices=range(len(total_lengths)),
        total_lengths=total_lengths,
        max_tokens_per_gpu=max_tokens_per_gpu,
    )
    return partitions, mbi, cp_sizes, nmb, gbs_per_step


@pytest.mark.unit
@pytest.mark.parametrize("dp_size,cp_size", [(1, 2), (2, 2), (4, 2)])
def test_dcp_all_short(dp_size, cp_size):
    """Short samples run on single ranks; rounds stay fully covered."""
    world_size = dp_size * cp_size
    _, _, cp_sizes, nmb, gbs_per_step = run_dcp(
        [3] * (4 * world_size), dp_size=dp_size, cp_size=cp_size, max_tokens_per_gpu=10
    )
    assert gbs_per_step == [4 * world_size]
    # 3 samples fit per rank: the first round is all single-rank blocks, the leftover
    # units fill a second round.
    assert nmb == [2]
    assert all(cp_sizes[r][0] == 1 for r in range(world_size))


@pytest.mark.unit
@pytest.mark.parametrize("dp_size,cp_size", [(1, 2), (2, 2), (4, 2)])
def test_dcp_all_long(dp_size, cp_size):
    """Samples needing 2 ranks each run on aligned pairs."""
    world_size = dp_size * cp_size
    _, _, cp_sizes, nmb, _ = run_dcp([15] * world_size, dp_size=dp_size, cp_size=cp_size, max_tokens_per_gpu=10)
    assert nmb == [2]
    assert all(k == 2 for r in range(world_size) for k in cp_sizes[r])


@pytest.mark.unit
@pytest.mark.parametrize("dp_size,cp_size", [(1, 2), (2, 2), (4, 2)])
def test_dcp_mixed_lengths(dp_size, cp_size):
    total_lengths = [1, 35, 8, 12, 4, 25, 9, 2, 17, 3, 6, 40, 5, 11, 7, 1]
    world_size = dp_size * cp_size
    total_lengths = [min(length, world_size * 10) for length in total_lengths]
    run_dcp(total_lengths, dp_size=dp_size, cp_size=cp_size, max_tokens_per_gpu=10)


@pytest.mark.unit
@pytest.mark.parametrize("dp_size,cp_size", [(1, 2), (2, 2), (4, 2)])
def test_dcp_single_sample_needs_full_world(dp_size, cp_size):
    world_size = dp_size * cp_size
    _, _, cp_sizes, nmb, _ = run_dcp([world_size * 10], dp_size=dp_size, cp_size=cp_size, max_tokens_per_gpu=10)
    assert nmb == [1]
    assert all(cp_sizes[r] == [world_size] for r in range(world_size))


@pytest.mark.unit
def test_dcp_single_short_sample_grows_to_full_world():
    """A lone short sample is grown to cover every rank rather than leaving ranks idle."""
    partitions, _, cp_sizes, nmb, _ = run_dcp([3], dp_size=4, cp_size=1, max_tokens_per_gpu=10)
    assert nmb == [1]
    assert partitions == [[0]] * 4
    assert all(cp_sizes[r] == [4] for r in range(4))


@pytest.mark.unit
def test_dcp_fills_idle_block_by_merging_buddy():
    """W=4, one k=2 sample at ranks [0, 2) and one k=1 sample at rank 2: rank 3 is free,
    so the rank-2 unit grows to k=2 at [2, 4)."""
    _, _, cp_sizes, nmb, _ = run_dcp([15, 3], dp_size=2, cp_size=2, max_tokens_per_gpu=10)
    assert nmb == [1]
    assert [cp_sizes[r][0] for r in range(4)] == [2, 2, 2, 2]


@pytest.mark.unit
@pytest.mark.parametrize("dp_size,cp_size", [(1, 2), (2, 2), (4, 2)])
def test_dcp_multi_step(dp_size, cp_size):
    """Rounds are per step; each step independently covers all ranks."""
    world_size = dp_size * cp_size
    total_lengths = [3, 15, 7, 2, 9, 4, 1, 12, 6, 8, 5, 11]
    partitions, mbi, cp_sizes, nmb, gbs_per_step = run_dcp(
        total_lengths, dp_size=dp_size, cp_size=cp_size, max_tokens_per_gpu=10, global_batch_size=4
    )
    assert gbs_per_step == [4, 4, 4]
    assert len(nmb) == 3
    # Each step's rounds only hold that step's samples.
    cursor = 0
    for step_i, n in enumerate(nmb):
        step_samples = {
            partitions[r][i] for r in range(world_size) for mbs in mbi[r][cursor : cursor + n] for i in mbs
        }
        assert step_samples == set(range(step_i * 4, (step_i + 1) * 4))
        cursor += n


@pytest.mark.unit
@pytest.mark.parametrize("dp_size,cp_size", [(1, 2), (2, 2), (4, 2), (8, 4)])
def test_dcp_random_lengths(dp_size, cp_size):
    world_size = dp_size * cp_size
    rng = random.Random(world_size)
    for _ in range(20):
        total_lengths = [rng.randint(1, world_size * 10) for _ in range(rng.randint(1, 40))]
        run_dcp(total_lengths, dp_size=dp_size, cp_size=cp_size, max_tokens_per_gpu=10)


@pytest.mark.unit
def test_dcp_rejects_too_long_sample():
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2, cp_size=2)
    with pytest.raises(ValueError, match="exceeding world_size"):
        build_dynamic_cp_schedule(args, tp, [3, 41], global_batch_size=2, rollout_indices=[0, 1])


@pytest.mark.unit
def test_dcp_rejects_non_power_of_two_world():
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=3, cp_size=1)
    with pytest.raises(AssertionError, match="power of 2"):
        build_dynamic_cp_schedule(args, tp, [3, 4, 5], global_batch_size=3, rollout_indices=[0, 1, 2])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
