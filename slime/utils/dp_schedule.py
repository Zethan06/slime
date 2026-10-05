"""Per-rollout DP/microbatch scheduling.

Pure-Python logic that decides, for one rollout's worth of sample lengths,
how to group samples into micro-batches and which DP rank owns each mbs.
Lives outside the ray/sglang-importing modules so it can be unit-tested
under CPU-only CI.

The scheduling philosophy is **pack first, distribute second**:

  1. Group samples by rollout id (``rollout_indices[i]`` =
     ``samples[i].index``) and split rollouts into steps of
     ``global_batch_size`` rollouts each. In the common case one rollout
     emits one training sample so this is the same as a contiguous chunk;
     under compact / subagent one rollout may emit multiple training
     samples, in which case all of those samples stay in the same step.
  2. For each step, pack its samples into ``K`` micro-batches with a
     single first-fit pass (dynamic batch) or fixed-size chunking
     (static batch).
  3. Adjust ``K`` to a multiple of ``dp_size * (mb_group if vpp>1 else 1)``
     by splitting the largest multi-sample bins (dynamic only).
  4. Distribute the ``K`` mbs across ``dp_size`` ranks, ``K / dp_size``
     each, with either a strided round-robin or a Karmarkar-Karp pass on
     estimated mbs FLOPs.

Invariants guaranteed by :func:`build_dp_schedule` (asserted by the tests):
  - every DP rank runs the **same** ``num_microbatches`` per training step
    (required for PP sync);
  - every mbs (dynamic path without ``balance_by_flops``) holds
    ``<= max_tokens_per_gpu * cp_size`` tokens, with one exception — an
    individual sample larger than that cap lands alone in its own mbs (and
    that mbs is the only one allowed to exceed the cap);
  - the union of per-rank sample indices equals the set of samples kept
    after trimming trailing rollouts (every kept sample placed exactly
    once);
  - flattening ``micro_batch_indices`` for a rank yields
    ``range(num_samples_rank)`` (each rank's samples are tiled exactly
    once by its mbs schedule).

:func:`build_dynamic_cp_schedule` is the dynamic context parallel variant:
all ``dp_size * cp_size`` ranks are independent workers, each step is split
into rounds, and in every round the ranks are partitioned into aligned
power-of-2 blocks, each block running one mbs with CP size = block size.
Rounds are formed by Megatron-LM's packing-aware dynamic CP scheduler
(``next_hdp_group_packing_aware``, vendored below).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from math import ceil, log2
from typing import Any

from slime.utils.flops_utils import calculate_fwd_flops
from slime.utils.seqlen_balancing import expand_bins_by_splitting, first_fit_pack, get_seqlen_balanced_partitions

logger = logging.getLogger(__name__)


def _calculate_workloads(step_lengths, args):
    return [calculate_fwd_flops([sl], args) for sl in step_lengths]


def _pack_step_into_mbs(
    step_lengths: list[int],
    *,
    args: Any,
    use_dynamic_batch_size: bool,
    max_per_bin: int | None,
    micro_batch_size: int | None,
    balance_by_flops: bool = False,
) -> list[list[int]]:
    """Group a step's samples into mbs. Returns ``mbs[k]`` = local indices into ``step_lengths``."""
    if use_dynamic_batch_size:
        assert max_per_bin is not None
        if balance_by_flops:
            total_tokens = sum(step_lengths)
            num_mbs = max(1, (total_tokens + max_per_bin - 1) // max_per_bin)
            if num_mbs >= len(step_lengths):
                return [[i] for i in range(len(step_lengths))]
            workloads = _calculate_workloads(step_lengths, args)
            # FLOPs balancing does not enforce the token cap per mbs. A
            # partition can exceed max_per_bin and may OOM if the cap is tight.
            return get_seqlen_balanced_partitions(workloads, num_mbs, equal_size=False)
        return first_fit_pack(step_lengths, max_per_bin)
    assert micro_batch_size is not None
    n = len(step_lengths)
    return [list(range(i, min(i + micro_batch_size, n))) for i in range(0, n, micro_batch_size)]


def _split_samples_into_steps(rollout_indices: list[int], global_batch_size: int) -> list[list[int]]:
    """Return ``steps[s]`` = global sample indices of step s, ``global_batch_size`` rollouts per step."""
    # Group samples by rollout id (preserve first-occurrence order). All
    # samples from one rollout stay in a single step so the per-rollout loss
    # reducer is well-defined.
    rollout_id_to_samples: dict[int, list[int]] = {}
    for sample_pos, rid in enumerate(rollout_indices):
        rollout_id_to_samples.setdefault(rid, []).append(sample_pos)
    rollout_ids = list(rollout_id_to_samples.keys())

    num_steps = len(rollout_ids) // global_batch_size
    assert num_steps >= 1, (
        f"num_rollouts ({len(rollout_ids)}) < global_batch_size ({global_batch_size}); "
        f"need at least one rollout per step."
    )
    return [
        [
            pos
            for rid in rollout_ids[step_i * global_batch_size : (step_i + 1) * global_batch_size]
            for pos in rollout_id_to_samples[rid]
        ]
        for step_i in range(num_steps)
    ]


def build_dp_schedule(
    args: Any,
    train_parallel_config: dict,
    total_lengths: list[int],
    *,
    global_batch_size: int,
    rollout_indices: list[int],
) -> tuple[list[list[int]], list[list[list[int]]], list[int], list[int]]:
    """Compute the per-rank DP partition and micro-batch schedule.

    See module docstring for the pack-first-distribute-second strategy.

    Args:
        args: Namespace with ``micro_batch_size``, ``use_dynamic_batch_size``,
            ``max_tokens_per_gpu``, ``balance_data``.
        train_parallel_config: ``{"dp_size", "cp_size", "vpp_size",
            "microbatch_group_size_per_vp_stage"}``.
        total_lengths: token count per sample, indexed globally.
        global_batch_size: number of rollouts (NOT training samples) per
            training step. Number of training steps =
            ``num_rollouts // global_batch_size``; trailing rollouts whose
            samples don't fit are dropped.
        rollout_indices: rollout id for each sample (``samples[i].index``).
            Samples sharing the same id are kept together in one step.

    Returns:
        ``(partitions, micro_batch_indices, num_microbatches, global_batch_sizes)``.
        ``global_batch_sizes[s]`` = rollout count for step s (constant
        ``global_batch_size`` for every step).
    """
    dp_size = train_parallel_config["dp_size"]
    cp_size = train_parallel_config["cp_size"]
    vpp_size = train_parallel_config["vpp_size"]
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]

    max_per_bin = None
    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None
        max_per_bin = args.max_tokens_per_gpu * cp_size

    # mbs count per step must be divisible by (dp_size * mb_group_for_vpp) so
    # every rank ends up with the same num_mbs and (for VPP) the per-rank mbs
    # count is a multiple of mb_group.
    align_to = dp_size * (mb_group if vpp_size > 1 else 1)

    partitions: list[list[int]] = [[] for _ in range(dp_size)]
    micro_batch_indices: list[list[list[int]]] = [[] for _ in range(dp_size)]
    num_microbatches: list[int] = []
    global_batch_sizes: list[int] = []

    for step_i, sample_indices in enumerate(_split_samples_into_steps(rollout_indices, global_batch_size)):
        step_lengths = [total_lengths[i] for i in sample_indices]
        global_batch_sizes.append(global_batch_size)
        assert len(sample_indices) >= dp_size, (
            f"step {step_i}: {len(sample_indices)} samples < dp_size {dp_size}; "
            f"each step needs at least one sample per rank."
        )

        # 1. Pack samples in this step into mbs with one global pass.
        # ``step_mbs`` indices are LOCAL into ``sample_indices``.
        step_mbs = _pack_step_into_mbs(
            step_lengths,
            args=args,
            use_dynamic_batch_size=args.use_dynamic_batch_size,
            max_per_bin=max_per_bin,
            micro_batch_size=getattr(args, "micro_batch_size", None),
            balance_by_flops=args.balance_by_flops,
        )

        # 2. Align mbs count to a multiple of ``align_to``.
        target_K = max(((len(step_mbs) + align_to - 1) // align_to) * align_to, align_to)
        if target_K != len(step_mbs):
            if args.use_dynamic_batch_size:
                expand_bins_by_splitting(step_mbs, target_K, step_lengths)
                assert len(step_mbs) == target_K, (
                    f"dynamic path: could only produce {len(step_mbs)} mbs after maximal splitting; "
                    f"need {target_K}. step {step_i} has {len(sample_indices)} samples, below the "
                    f"alignment threshold ({align_to})."
                )
            else:
                raise AssertionError(
                    f"static path: num_mbs ({len(step_mbs)}) is not a multiple of "
                    f"dp_size * mb_group ({align_to}); got "
                    f"step_size={len(sample_indices)}, micro_batch_size={args.micro_batch_size}, "
                    f"dp_size={dp_size}, mb_group={mb_group if vpp_size > 1 else 1}. "
                    f"Splitting static mbs would break the fixed-size invariant; adjust the config "
                    f"so step_size % (dp_size * micro_batch_size * mb_group) == 0."
                )

        K = len(step_mbs)
        num_mbs_per_rank = K // dp_size
        num_microbatches.append(num_mbs_per_rank)

        # 3. Distribute mbs across ranks: KK on estimated FLOPs when rank
        # workload balancing is enabled, otherwise a strided round-robin.
        if args.balance_data:
            step_workloads = _calculate_workloads(step_lengths, args)
            mbs_weights = [sum(step_workloads[i] for i in bin_) for bin_ in step_mbs]
            rank_mbs_idx = get_seqlen_balanced_partitions(mbs_weights, dp_size, equal_size=True)
        else:
            rank_mbs_idx = [list(range(r, K, dp_size)) for r in range(dp_size)]

        # 4. Build per-rank partitions (global sample indices) and micro_batch_indices
        # (local indices into partitions[r]).
        for r in range(dp_size):
            for mbs_idx in rank_mbs_idx[r]:
                mbs_locals = step_mbs[mbs_idx]  # local indices into sample_indices
                local_start = len(partitions[r])
                partitions[r].extend(sample_indices[i] for i in mbs_locals)
                micro_batch_indices[r].append(list(range(local_start, local_start + len(mbs_locals))))

    return partitions, micro_batch_indices, num_microbatches, global_batch_sizes


# ---------------------------------------------------------------------------
# Vendored from NVIDIA/Megatron-LM commit d2e7ec5b (PR #5154, "Improve default
# dynamic CP packing scheduler"), megatron/core/datasets/data_schedule_utils.py.
# Copyright (c) 2025 NVIDIA CORPORATION. Licensed under the Apache License 2.0.
# The logic is copied verbatim; only type annotations use builtin generics and
# lines are rewrapped (ruff UP rules, black at 119 cols). Replace with an import from
# ``megatron.core.datasets.data_schedule_utils`` once slime's pinned Megatron
# includes d2e7ec5b.
# ---------------------------------------------------------------------------

_DYNAMIC_CP_WORKLOAD_CAP_DELTA = 0.05


@lru_cache(maxsize=128)
def dcp_gpus_needed(seq_len: int, max_seq_len_per_rank: int, min_cp_size: int = 1) -> int:
    """Number of GPUs needed, rounded up to the next power of 2, lower-bounded by min_cp_size."""
    raw = max(1, 2 ** ceil(log2(seq_len / max_seq_len_per_rank)))
    return max(min_cp_size, raw)


def next_hdp_group_packing_aware(
    sample_seqlens: list[tuple[int, int]],
    total_gpus: int,
    max_seq_len_per_rank: int,
    min_cp_size: int = 1,
) -> tuple[list[list[int]], list[tuple[int, int]], list[float], list[list[int]]]:
    """Form one DCP microbatch with packing-aware CP group selection.

    This differs from the legacy DCP scheduler in two ways:
    1. Short sequences may use a larger CP group than their minimum required
       CP size when that lowers the critical-path rank workload.
    2. Candidate placements are bounded by ``tall * max_seq_len_per_rank``,
       the per-rank workload upper bound for packing sequences no longer than
       the local tallest sequence in the microbatch.

    The scheduler keeps the legacy invariant that each returned microbatch has
    no empty DPxCP rank after the fill step. For non-power-of-two DPxCP layouts,
    it falls back to the full DPxCP group if power-of-two expansion cannot fill
    every rank.
    """
    if not sample_seqlens:
        return (
            [[] for _ in range(total_gpus)],
            [],
            [0.0 for _ in range(total_gpus)],
            [[] for _ in range(total_gpus)],
        )

    def cp_min_fn(seq_len: int) -> int:
        return dcp_gpus_needed(seq_len, max_seq_len_per_rank, min_cp_size)

    def workload(seq_len: int, cp_size: int) -> float:
        return (seq_len * seq_len) / cp_size

    sample_seqlens = sorted(sample_seqlens, key=lambda x: x[1], reverse=True)
    local_tall = sample_seqlens[0][1]
    cap = float(local_tall) * float(max_seq_len_per_rank) * (1.0 + _DYNAMIC_CP_WORKLOAD_CAP_DELTA)

    micro_batches: list[list[int]] = [[] for _ in range(total_gpus)]
    exec_times: list[float] = [0.0 for _ in range(total_gpus)]
    sample_ids_per_gpu: list[list[int]] = [[] for _ in range(total_gpus)]
    packing_sequence_len: dict[int, float] = {}

    gpu_group_id: list[int | None] = [None] * total_gpus
    group_members: dict[int, list[int]] = {}
    group_size: dict[int, int] = {}
    next_gid = 0

    sample_id, seq_len = sample_seqlens[0]
    cp_size = cp_min_fn(seq_len)
    assert cp_size <= total_gpus, (
        f"Sequence length {seq_len} requires CP size {cp_size}, " f"but only {total_gpus} DPxCP ranks are available."
    )
    group_id = next_gid
    next_gid += 1
    members = list(range(cp_size))
    group_members[group_id] = members
    group_size[group_id] = cp_size
    packing_sequence_len[group_id] = seq_len / cp_size
    per_gpu_cost = workload(seq_len, cp_size)
    for rank in members:
        gpu_group_id[rank] = group_id
        micro_batches[rank].append(seq_len)
        exec_times[rank] += per_gpu_cost
        sample_ids_per_gpu[rank].append(sample_id)

    leftovers: list[tuple[int, int]] = []
    for sample_id, seq_len in sample_seqlens[1:]:
        min_needed = cp_min_fn(seq_len)
        best = None

        cp_size = min_needed
        while cp_size <= total_gpus:
            per_gpu_cost = workload(seq_len, cp_size)

            for group_id, size in list(group_size.items()):
                if size != cp_size:
                    continue
                if packing_sequence_len.get(group_id, 0) + seq_len / cp_size > max_seq_len_per_rank:
                    continue
                members = group_members[group_id]
                member_set = set(members)
                projected_max = max(
                    time + per_gpu_cost if rank in member_set else time for rank, time in enumerate(exec_times)
                )
                if projected_max <= cap and (best is None or projected_max < best[0]):
                    best = (projected_max, cp_size, "add", group_id, None)

            free_ranks = [rank for rank, assigned_group_id in enumerate(gpu_group_id) if assigned_group_id is None]
            if len(free_ranks) >= cp_size:
                chosen_members = sorted(free_ranks, key=lambda rank: exec_times[rank])[:cp_size]
                chosen_set = set(chosen_members)
                projected_max = max(
                    time + per_gpu_cost if rank in chosen_set else time for rank, time in enumerate(exec_times)
                )
                if projected_max <= cap and (best is None or projected_max < best[0]):
                    best = (projected_max, cp_size, "new", None, chosen_members)

            cp_size *= 2

        if best is None:
            leftovers.append((sample_id, seq_len))
            continue

        _, selected_cp_size, action, group_id, chosen_members = best
        per_gpu_cost = workload(seq_len, selected_cp_size)
        if action == "add":
            members = group_members[group_id]
            packing_sequence_len[group_id] += seq_len / selected_cp_size
            for rank in members:
                micro_batches[rank].append(seq_len)
                exec_times[rank] += per_gpu_cost
                sample_ids_per_gpu[rank].append(sample_id)
        else:
            group_id = next_gid
            next_gid += 1
            group_members[group_id] = chosen_members
            group_size[group_id] = selected_cp_size
            packing_sequence_len[group_id] = seq_len / selected_cp_size
            for rank in chosen_members:
                gpu_group_id[rank] = group_id
                micro_batches[rank].append(seq_len)
                exec_times[rank] += per_gpu_cost
                sample_ids_per_gpu[rank].append(sample_id)

    def fill_empty_gpus_once() -> bool:
        nonlocal micro_batches, exec_times, sample_ids_per_gpu

        empty_ranks = [rank for rank, micro_batch in enumerate(micro_batches) if not micro_batch]
        if not empty_ranks:
            return False
        assert all(
            not micro_batches[rank] for rank in range(empty_ranks[0], total_gpus)
        ), "fill_empty_gpus_once assumes empty ranks are contiguous at the tail"

        existing_group_sizes = set(group_size.values())
        if not existing_group_sizes:
            return False
        min_group_size = min(existing_group_sizes)
        next_power = min(min_group_size * 2, total_gpus)

        for group_id, size in list(group_size.items()):
            if size != min_group_size:
                continue

            members = group_members[group_id]
            needed_count = next_power - min_group_size
            group_start_rank = members[0]
            group_end_rank = members[-1]
            empty_rank = empty_ranks[0]
            if group_end_rank + 1 > empty_rank or group_end_rank + needed_count >= total_gpus:
                continue

            work_to_push = micro_batches[group_end_rank + 1 : empty_rank]
            exec_times_to_push = exec_times[group_end_rank + 1 : empty_rank]
            sample_ids_to_push = sample_ids_per_gpu[group_end_rank + 1 : empty_rank]

            new_micro_batches: list[list[int]] = [[] for _ in range(total_gpus)]
            new_exec_times: list[float] = [0.0 for _ in range(total_gpus)]
            new_sample_ids_per_gpu: list[list[int]] = [[] for _ in range(total_gpus)]

            for rank in range(group_start_rank):
                new_micro_batches[rank] = micro_batches[rank]
                new_exec_times[rank] = exec_times[rank]
                new_sample_ids_per_gpu[rank] = sample_ids_per_gpu[rank]

            for rank in range(group_start_rank, group_end_rank + needed_count + 1):
                new_micro_batches[rank] = list(micro_batches[group_end_rank])
                new_exec_times[rank] = sum(workload(length, next_power) for length in micro_batches[group_end_rank])
                new_sample_ids_per_gpu[rank] = list(sample_ids_per_gpu[group_end_rank])

            for idx, work in enumerate(work_to_push):
                target_rank = group_end_rank + needed_count + 1 + idx
                new_micro_batches[target_rank] = work
                new_exec_times[target_rank] = exec_times_to_push[idx]
                new_sample_ids_per_gpu[target_rank] = sample_ids_to_push[idx]

            group_size[group_id] = next_power
            group_members[group_id] = list(range(group_start_rank, group_end_rank + needed_count + 1))
            for other_group_id in list(group_size.keys()):
                if other_group_id == group_id:
                    continue
                if min(group_members[other_group_id]) > group_end_rank:
                    group_members[other_group_id] = [rank + needed_count for rank in group_members[other_group_id]]

            micro_batches = new_micro_batches
            exec_times = new_exec_times
            sample_ids_per_gpu = new_sample_ids_per_gpu
            return True

        return False

    def fill_with_full_dpxcp_group() -> None:
        nonlocal micro_batches, exec_times, sample_ids_per_gpu, leftovers

        selected: list[tuple[int, int]] = []
        next_leftovers: list[tuple[int, int]] = []
        packed_sequence_len = 0.0

        for sample_id, seq_len in sample_seqlens:
            per_rank_len = seq_len / total_gpus
            if packed_sequence_len + per_rank_len <= max_seq_len_per_rank:
                selected.append((sample_id, seq_len))
                packed_sequence_len += per_rank_len
            else:
                next_leftovers.append((sample_id, seq_len))

        assert selected, (
            "At least one sequence should fit in the full DPxCP group; " "try to increase 'max-seqlen-per-dp-cp-rank'."
        )

        selected_ids = [sample_id for sample_id, _ in selected]
        selected_lens = [seq_len for _, seq_len in selected]
        per_rank_work = sum(workload(seq_len, total_gpus) for _, seq_len in selected)

        micro_batches = [list(selected_lens) for _ in range(total_gpus)]
        exec_times = [per_rank_work for _ in range(total_gpus)]
        sample_ids_per_gpu = [list(selected_ids) for _ in range(total_gpus)]
        leftovers = next_leftovers

    while any(not micro_batch for micro_batch in micro_batches):
        if not fill_empty_gpus_once():
            fill_with_full_dpxcp_group()
            break

    return micro_batches, leftovers, exec_times, sample_ids_per_gpu


# ---------------------------------------------------------------------------
# End of vendored Megatron-LM code.
# ---------------------------------------------------------------------------


def _schedule_step_rounds(
    step_lengths: list[int], *, world_size: int, max_tokens_per_gpu: int
) -> list[list[list[int]]]:
    """Schedule one step's samples into rounds of aligned power-of-2 rank blocks.

    Same loop as Megatron's ``DefaultDynamicCPScheduler.get_groups_and_subsamples``
    (without VPP alignment): each round is one ``next_hdp_group_packing_aware``
    call, which packs short samples into the spare per-rank capacity of larger
    CP groups and balances the per-rank ``seq_len**2 / cp_size`` workload.

    Returns ``rounds[j][r]`` = local sample ids rank ``r`` runs in round ``j``;
    ranks of one block share the same list.
    """
    longest = max(step_lengths)
    if longest > world_size * max_tokens_per_gpu:
        raise ValueError(
            f"a sample has {longest} tokens, exceeding world_size * max_tokens_per_gpu "
            f"({world_size} * {max_tokens_per_gpu} = {world_size * max_tokens_per_gpu})."
        )

    sample_id_groups = []
    sample_id_seqlens = sorted(enumerate(step_lengths), key=lambda x: x[1], reverse=True)
    while sample_id_seqlens:
        _, sample_id_seqlens, _, sample_ids = next_hdp_group_packing_aware(
            sample_id_seqlens, world_size, max_seq_len_per_rank=max_tokens_per_gpu
        )
        sample_id_groups.append(sample_ids)
    return sample_id_groups


def build_dynamic_cp_schedule(
    args: Any,
    train_parallel_config: dict,
    total_lengths: list[int],
    *,
    global_batch_size: int,
    rollout_indices: list[int],
) -> tuple[list[list[int]], list[list[list[int]]], list[list[int]], list[int], list[int]]:
    """Compute the per-rank dynamic context parallel (DCP) schedule.

    All ``W = dp_size * cp_size`` ranks act as independent workers. Each step
    is split into rounds; in every round the W ranks are partitioned into
    aligned, contiguous power-of-2 blocks and each block runs one mbs, with
    all ranks of a block receiving the same sample list. The block size of an
    mbs is the CP size it runs with. Every rank runs the same number of rounds
    per step, and every mbs holds ``<= cp_size * max_tokens_per_gpu`` tokens.

    Args and step splitting are the same as :func:`build_dp_schedule`; only
    ``args.max_tokens_per_gpu`` is read.

    Returns:
        ``(partitions, micro_batch_indices, micro_batch_cp_sizes, num_microbatches,
        global_batch_sizes)``. ``partitions[r]`` lists the global sample indices
        rank r receives (a sample appears on every rank of its block),
        ``micro_batch_indices[r]`` holds per round the local indices into
        ``partitions[r]``, ``micro_batch_cp_sizes[r]`` holds per round the block
        size, and ``num_microbatches[s]`` is the round count of step s.
    """
    world_size = train_parallel_config["dp_size"] * train_parallel_config["cp_size"]
    assert (
        world_size >= 2 and world_size & (world_size - 1) == 0
    ), f"dynamic CP needs dp_size * cp_size to be a power of 2 >= 2, got {world_size}."
    assert args.max_tokens_per_gpu is not None

    partitions: list[list[int]] = [[] for _ in range(world_size)]
    micro_batch_indices: list[list[list[int]]] = [[] for _ in range(world_size)]
    micro_batch_cp_sizes: list[list[int]] = [[] for _ in range(world_size)]
    num_microbatches: list[int] = []
    global_batch_sizes: list[int] = []

    for sample_indices in _split_samples_into_steps(rollout_indices, global_batch_size):
        rounds = _schedule_step_rounds(
            [total_lengths[i] for i in sample_indices],
            world_size=world_size,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
        )
        num_microbatches.append(len(rounds))
        global_batch_sizes.append(global_batch_size)

        for rank_samples in rounds:
            # A block is a run of ranks sharing one sample list (lists of different
            # blocks are disjoint and non-empty). The train side looks up the CP group
            # by size alone, so every block must be an aligned power-of-2 slice.
            start = 0
            while start < world_size:
                samples = rank_samples[start]
                cp_size = 1
                while start + cp_size < world_size and rank_samples[start + cp_size] == samples:
                    cp_size += 1
                assert (
                    cp_size & (cp_size - 1) == 0 and start % cp_size == 0
                ), f"dynamic CP block at ranks [{start}, {start + cp_size}) is not an aligned power-of-2 slice"
                for r in range(start, start + cp_size):
                    local_start = len(partitions[r])
                    partitions[r].extend(sample_indices[i] for i in samples)
                    micro_batch_indices[r].append(list(range(local_start, local_start + len(samples))))
                    micro_batch_cp_sizes[r].append(cp_size)
                start += cp_size

    return partitions, micro_batch_indices, micro_batch_cp_sizes, num_microbatches, global_batch_sizes
