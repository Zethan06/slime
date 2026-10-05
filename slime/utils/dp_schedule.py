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
"""

from __future__ import annotations

import logging
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


def _schedule_step_rounds(
    step_lengths: list[int], *, world_size: int, max_tokens_per_gpu: int
) -> list[list[list[int]]]:
    """Schedule one step's samples into rounds of aligned power-of-2 rank blocks.

    Returns ``rounds[j][r]`` = local sample ids rank ``r`` runs in round ``j``;
    ranks of one block share the same list. Same layout as Megatron's
    ``DefaultDynamicCPScheduler.get_groups_and_subsamples`` sample id groups.
    """
    # 1. Smallest power-of-2 block size whose token budget fits each sample.
    sample_cp_sizes = []
    for i, length in enumerate(step_lengths):
        cp_size = 1
        while cp_size * max_tokens_per_gpu < length:
            cp_size *= 2
        if cp_size > world_size:
            raise ValueError(
                f"sample {i} has {length} tokens, exceeding world_size * max_tokens_per_gpu "
                f"({world_size} * {max_tokens_per_gpu} = {world_size * max_tokens_per_gpu})."
            )
        sample_cp_sizes.append(cp_size)

    # 2. Pack each block-size class into units of at most cp_size * max_tokens_per_gpu tokens.
    units = []
    for cp_size in sorted(set(sample_cp_sizes), reverse=True):
        class_samples = [i for i, k in enumerate(sample_cp_sizes) if k == cp_size]
        for bin_ in first_fit_pack([step_lengths[i] for i in class_samples], cp_size * max_tokens_per_gpu):
            units.append((cp_size, [class_samples[j] for j in bin_]))
    units.sort(key=lambda unit: (-unit[0], -sum(step_lengths[i] for i in unit[1])))

    # 3. Place each unit into the first round with room for it. Power-of-2 sizes summing
    # to <= world_size laid out in descending order are always aligned, so a round only
    # tracks its units.
    rounds: list[list[list]] = []
    for cp_size, samples in units:
        for round_units in rounds:
            if sum(unit[0] for unit in round_units) + cp_size <= world_size:
                round_units.append([cp_size, samples])
                break
        else:
            rounds.append([[cp_size, samples]])

    # 4. Leave no rank idle: double the smallest unit (most tokens first, then placement
    # order) until the round is full. Every size is a multiple of the smallest, so the
    # round never overflows.
    sample_id_groups = []
    for round_units in rounds:
        while sum(unit[0] for unit in round_units) < world_size:
            unit = min(round_units, key=lambda u: (u[0], -sum(step_lengths[i] for i in u[1])))
            unit[0] *= 2
        round_units.sort(key=lambda unit: -unit[0])
        sample_id_groups.append([samples for cp_size, samples in round_units for _ in range(cp_size)])
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
    assert world_size >= 2 and world_size & (world_size - 1) == 0, (
        f"dynamic CP needs dp_size * cp_size to be a power of 2 >= 2, got {world_size}."
    )
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
            for r, samples in enumerate(rank_samples):
                local_start = len(partitions[r])
                partitions[r].extend(sample_indices[i] for i in samples)
                micro_batch_indices[r].append(list(range(local_start, local_start + len(samples))))
                micro_batch_cp_sizes[r].append(rank_samples.count(samples))

    return partitions, micro_batch_indices, micro_batch_cp_sizes, num_microbatches, global_batch_sizes
