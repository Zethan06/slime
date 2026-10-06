"""Multi-process (gloo) checks of dynamic CP on a real mixed-k schedule.

Each worker builds Megatron-shaped groups (static CP groups, aligned power-of-2
hybrid groups, the DP x CP group), registers them with the real
``init_dynamic_cp_groups``, takes its share of a ``build_dynamic_cp_schedule``
schedule that mixes CP sizes 1, 2 and 4, and runs the per-sample CP collectives
used outside the forward pass. Results must match a CP=1 computation, and the
collectives must not deadlock even though block members also hold other samples.
"""

import json
from types import SimpleNamespace

import _cp_dist_helpers  # noqa: F401
import pytest
import torch
from _cp_dist_helpers import free_port, init_worker_process_group

NUM_GPUS = 0
WORLD_SIZE = 4
MAX_TOKENS_PER_GPU = 128
TOTAL_LENGTHS = [500, 400, 200, 120, 100, 40, 20, 10, 5, 400, 120, 20]


def _full(sample: int, length: int, salt: int) -> torch.Tensor:
    return torch.randn(length, generator=torch.Generator().manual_seed(1000 * sample + salt))


def _dynamic_cp_worker(rank, world_size, master_port, static_cp, result_path):
    import torch.distributed as dist

    dp_cp_group = init_worker_process_group(rank, world_size, master_port)
    from megatron.core import mpu

    from slime.backends.megatron_utils import cp_utils
    from slime.utils.dp_schedule import build_dynamic_cp_schedule
    from slime.utils.ppo_utils import get_advantages_and_returns_batch, get_reinforce_plus_plus_returns

    # Megatron-shaped groups: aligned blocks of consecutive DP x CP ranks.
    groups = {}
    for size in [static_cp] + [2**i for i in range(1, world_size.bit_length() - 1)]:
        for start in range(0, world_size, size):
            group = dist.new_group(list(range(start, start + size)))
            if start <= rank < start + size:
                groups.setdefault(size, group)
    mpu.get_data_parallel_world_size = lambda with_context_parallel=False: (
        world_size if with_context_parallel else world_size // static_cp
    )
    mpu.get_context_parallel_world_size = lambda: static_cp
    mpu.get_context_parallel_rank = lambda: rank % static_cp
    mpu.get_context_parallel_group = lambda: groups.get(static_cp)
    mpu.get_hybrid_data_context_parallel_groups = lambda group_size: (
        dp_cp_group if group_size == world_size else groups[group_size]
    )
    cp_utils.init_dynamic_cp_groups()
    try:
        cp_utils.get_cp_context()
        missing_cp_raises = False
    except RuntimeError:
        missing_cp_raises = True

    partitions, micro_batch_indices, micro_batch_cp_sizes, _, _ = build_dynamic_cp_schedule(
        SimpleNamespace(max_tokens_per_gpu=MAX_TOKENS_PER_GPU),
        {"dp_size": world_size // static_cp, "cp_size": static_cp, "dynamic_cp": True},
        TOTAL_LENGTHS,
        global_batch_size=len(TOTAL_LENGTHS) // 2,
        rollout_indices=list(range(len(TOTAL_LENGTHS))),
    )
    samples = partitions[rank]
    cp_sizes = [None] * len(samples)
    for indices, cp_size in zip(micro_batch_indices[rank], micro_batch_cp_sizes[rank], strict=True):
        for i in indices:
            cp_sizes[i] = cp_size
    cps = [cp_utils.get_cp_context(cp_size) for cp_size in cp_sizes]
    no_cp = [cp_utils.CPContext(1, 0, None)] * len(samples)

    total_lengths = [TOTAL_LENGTHS[sample] for sample in samples]
    response_lengths = [max(1, total // 3) for total in total_lengths]
    full_values = [_full(s, r, 1) for s, r in zip(samples, response_lengths, strict=True)]
    full_rewards = [_full(s, r, 2) for s, r in zip(samples, response_lengths, strict=True)]

    def local(tensors):
        return [
            cp_utils.slice_log_prob_with_cp(t, total, response, cp=cp)
            for t, total, response, cp in zip(tensors, total_lengths, response_lengths, cps, strict=True)
        ]

    def gathered(tensors):
        return [
            cp_utils.all_gather_with_cp(t, total, response, cp=cp)
            for t, total, response, cp in zip(tensors, total_lengths, response_lengths, cps, strict=True)
        ]

    advantages, _ = get_advantages_and_returns_batch(
        total_lengths, response_lengths, local(full_values), local(full_rewards), 0.99, 0.95, cps=cps
    )
    ref_advantages, _ = get_advantages_and_returns_batch(
        total_lengths, response_lengths, full_values, full_rewards, 0.99, 0.95, cps=no_cp
    )
    masks = [torch.ones(r) for r in response_lengths]
    rewards = torch.tensor([float(s) for s in samples])
    returns = get_reinforce_plus_plus_returns(
        rewards, local(full_rewards), masks, response_lengths, total_lengths, 0.1, 0.99, cps=cps
    )
    ref_returns = get_reinforce_plus_plus_returns(
        rewards, full_rewards, masks, response_lengths, total_lengths, 0.1, 0.99, cps=no_cp
    )

    def max_err(actual, expected):
        return max((a - e).abs().max().item() for a, e in zip(gathered(actual), expected, strict=True))

    sum_of_sample_mean = cp_utils.get_sum_of_sample_mean(total_lengths, response_lengths, masks, cps=cps)
    reduced = torch.tensor([sum_of_sample_mean(torch.cat(local(full_values))).item()])
    dist.all_reduce(reduced)

    with open(f"{result_path}.{rank}", "w") as f:
        json.dump(
            {
                "missing_cp_raises": missing_cp_raises,
                "cp_sizes": sorted(set(cp_sizes)),
                "advantage_err": max_err(advantages, ref_advantages),
                "returns_err": max_err(returns, ref_returns),
                "sum_of_sample_mean": reduced.item(),
            },
            f,
        )
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.unit
@pytest.mark.parametrize("static_cp", [1, 2])
def test_dynamic_cp_per_sample_collectives_match_cp1(tmp_path, static_cp):
    import torch.multiprocessing as mp

    result_path = str(tmp_path / "result")
    mp.spawn(
        _dynamic_cp_worker,
        args=(WORLD_SIZE, free_port(), static_cp, result_path),
        nprocs=WORLD_SIZE,
        join=True,
    )
    results = [json.load(open(f"{result_path}.{rank}")) for rank in range(WORLD_SIZE)]

    assert sorted({k for result in results for k in result["cp_sizes"]}) == [1, 2, 4]
    for result in results:
        assert result["missing_cp_raises"]
        assert result["advantage_err"] < 1e-5
        assert result["returns_err"] < 1e-5
    expected = sum(_full(s, max(1, total // 3), 1).mean().item() for s, total in enumerate(TOTAL_LENGTHS))
    assert results[0]["sum_of_sample_mean"] == pytest.approx(expected, rel=1e-5)
