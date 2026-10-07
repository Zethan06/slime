"""Multi-process (gloo) checks of MoE routing under dynamic CP with the real Megatron router.

A fixed two-round schedule over 4 ranks mixes CP sizes 2, 1, 1 (round 0) and 4 (round 1).
Each rank builds its micro-batch with the real ``get_batch`` (zigzag slices, per-sample and
tail padding, router padding mask) and routes the matching slice of per-token router logits
through a real ``TopKRouter``, pointed at the micro-batch's TP x CP group by
``set_routers_for_dynamic_cp``; ``with_moe_aux_loss_scale`` sets the aux-loss gradient scale after
Megatron has set its own. The aux loss and its logit gradients must match routing every
micro-batch whole, without padding, on a single rank (CP = 1).

The workers import the real Megatron (``_cp_dist_helpers``' stub would shadow it), so this
module does not import that helper; it is skipped where Megatron is not installed.
"""

import json
import os
import socket
from functools import partial
from types import SimpleNamespace

import pytest
import torch

NUM_GPUS = 0
WORLD_SIZE = 4
NUM_EXPERTS = 6
TOPK = 2
AUX_LOSS_COEFF = 0.1
PAD_MULTIPLIER = 4
# Per round: the ranks of each micro-batch and its sample lengths (aligned power-of-2 blocks).
ROUNDS = [
    [((0, 1), [13, 6]), ((2,), [7, 5]), ((3,), [9])],
    [((0, 1, 2, 3), [23])],
]
NUM_GLOBAL_MICROBATCHES = sum(len(round_) for round_ in ROUNDS)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _sample_logits(round_id: int, mb_id: int, sample_id: int, length: int) -> torch.Tensor:
    seed = 100 * round_id + 10 * mb_id + sample_id
    return torch.randn(length, NUM_EXPERTS, generator=torch.Generator().manual_seed(seed))


def _router(lb_type: str, per_token: bool, pg_collection):
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        moe_aux_loss_coeff=AUX_LOSS_COEFF,
        moe_router_load_balancing_type=lb_type,
        calculate_per_token_loss=per_token,
        add_bias_linear=False,
    )
    router = TopKRouter(config, pg_collection)
    router.set_layer_number(1)
    return router.train()


def _route(router, logits, padding_mask, scale, wrap):
    """Route like MoELayer, backprop a zero main loss like Megatron's schedule; return logit grads."""
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    logits = logits.clone().unsqueeze(1).requires_grad_()  # [s, b=1, E]
    probs, _ = router.routing(logits, padding_mask=padding_mask)
    loss, *_ = wrap(lambda output: (output, None, None))((probs * 0).sum())
    MoEAuxLossAutoScaler.set_loss_scale(scale)
    loss.backward()
    return logits.grad.squeeze(1)


def _tracker_value(name: str) -> float:
    from megatron.core.transformer.moe.moe_utils import get_moe_layer_wise_logging_tracker

    tracker = get_moe_layer_wise_logging_tracker()
    value = tracker[name]["values"][0].item()
    tracker[name]["values"].zero_()
    return value


def _worker(rank, world_size, port, per_token, lb_type, result_path):
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from megatron.core import mpu
        from megatron.core.transformer.moe.router import TopKRouter  # noqa: F401
    except ImportError as exc:
        with open(f"{result_path}.{rank}", "w") as f:
            json.dump({"skip": str(exc)}, f)
        return

    from slime.backends.megatron_utils import cp_utils, data, moe_utils
    from slime.backends.megatron_utils.cp_utils import CPContext, slice_with_cp

    world = dist.new_group(list(range(world_size)))
    groups = {}
    for size in (1, 2):
        for start in range(0, world_size, size):
            group = dist.new_group(list(range(start, start + size)))
            if start <= rank < start + size:
                groups[size] = group
    groups[world_size] = world
    contexts = {size: CPContext(size, dist.get_rank(g), g, tp_cp_group=g) for size, g in groups.items()}
    cp_utils._dynamic_cp_contexts.clear()
    cp_utils._dynamic_cp_contexts.update(contexts)
    mpu.get_tensor_model_parallel_world_size = lambda: 1
    mpu.get_data_parallel_group = lambda with_context_parallel=False: world
    mpu.get_data_parallel_world_size = lambda with_context_parallel=False: world_size
    data.accelerator = SimpleNamespace(device=lambda: "cpu", current_device=lambda: "cpu")

    args = SimpleNamespace(calculate_per_token_loss=per_token, sequence_parallel=False)
    # The routers start on a static TP x CP group (the whole world) that mixes micro-batches.
    router = _router(lb_type, per_token, SimpleNamespace(tp=groups[1], cp=world, tp_cp=world, tp_dp_cp=world))
    my_mbs = [
        next((r_id, mb_id, ranks, lengths) for mb_id, (ranks, lengths) in enumerate(round_) if rank in ranks)
        for r_id, round_ in enumerate(ROUNDS)
    ]
    step_scale = moe_utils.get_moe_aux_loss_step_scale(args, torch.ones(1), [len(ranks) for _, _, ranks, _ in my_mbs])

    rows = []
    padding_ok = True
    for round_id, mb_id, ranks, lengths in my_mbs:
        cp = contexts[len(ranks)]
        rollout_data = {
            "tokens": [torch.arange(n) + 1 for n in lengths],
            "loss_masks": [torch.ones(n - 1, dtype=torch.int) for n in lengths],
            "total_lengths": lengths,
            "response_lengths": [n - 1 for n in lengths],
        }
        iterator = data.DataIterator(rollout_data, [list(range(len(lengths)))], micro_batch_cp_sizes=[cp.size])
        batch = data.get_batch(
            iterator, ["tokens", "loss_masks", "total_lengths", "response_lengths"], pad_multiplier=PAD_MULTIPLIER
        )
        # get_batch pads with token id 0; real tokens are >= 1.
        padding_ok &= torch.equal(batch["padding_mask"], batch["tokens"] == 0)

        # Padding rows get logits 0 and position -1, laid out like get_batch's tokens.
        logits = torch.cat(
            [slice_with_cp(_sample_logits(round_id, mb_id, i, n), 0.0, cp=cp) for i, n in enumerate(lengths)]
        )
        offsets = torch.tensor([0] + lengths[:-1]).cumsum(0).tolist()
        positions = torch.cat(
            [slice_with_cp(torch.arange(n) + offset, -1, cp=cp) for n, offset in zip(lengths, offsets, strict=True)]
        )
        tail = batch["tokens"].shape[1] - logits.shape[0]
        logits = torch.nn.functional.pad(logits, (0, 0, 0, tail))
        positions = torch.nn.functional.pad(positions, (0, tail), value=-1)

        moe_utils.set_routers_for_dynamic_cp(router, cp)
        grads = _route(
            router,
            logits,
            batch["padding_mask"].transpose(0, 1),
            torch.tensor([123.0]),  # Megatron's static-CP scale, which the wrapped loss replaces
            partial(moe_utils.with_moe_aux_loss_scale, args=args, batch=batch, step_scale=step_scale),
        )
        padding_ok &= bool((grads[positions < 0] == 0).all())
        rows.append({"round": round_id, "mb": mb_id, "positions": positions.tolist(), "grads": grads.tolist()})

    name = {"aux_loss": "load_balancing_loss", "seq_aux_loss": "seq_load_balancing_loss"}[lb_type]
    aux_sum = torch.tensor([_tracker_value(name)], dtype=torch.float64)
    dist.all_reduce(aux_sum, group=world)
    with open(f"{result_path}.{rank}", "w") as f:
        json.dump({"padding_ok": padding_ok, "aux_mean": aux_sum.item() / NUM_GLOBAL_MICROBATCHES, "rows": rows}, f)
    dist.barrier()
    dist.destroy_process_group()


def _reference(per_token, lb_type):
    """Route every micro-batch whole on one rank without padding (CP = 1); grads as DDP sees them."""
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        single = dist.new_group([0])
        router = _router(lb_type, per_token, SimpleNamespace(tp=single, cp=single, tp_cp=single, tp_dp_cp=single))
        name = {"aux_loss": "load_balancing_loss", "seq_aux_loss": "seq_load_balancing_loss"}[lb_type]
        grads, aux = {}, []
        for round_id, round_ in enumerate(ROUNDS):
            for mb_id, (_, lengths) in enumerate(round_):
                logits = torch.cat([_sample_logits(round_id, mb_id, i, n) for i, n in enumerate(lengths)])
                # Per-sample: Megatron averages the aux loss over the step's micro-batches.
                # Per-token: the router weights it by the micro-batch's token count.
                scale = torch.tensor([1.0 if per_token else 1.0 / NUM_GLOBAL_MICROBATCHES])
                grads[(round_id, mb_id)] = _route(router, logits, None, scale, lambda loss_func: loss_func)
                aux.append(_tracker_value(name))
        return grads, sum(aux) / len(aux)
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
@pytest.mark.parametrize("lb_type", ["aux_loss", "seq_aux_loss"])
@pytest.mark.parametrize("per_token", [False, True])
def test_dynamic_cp_router_aux_loss_matches_cp1(tmp_path, per_token, lb_type):
    import torch.multiprocessing as mp

    result_path = str(tmp_path / "result")
    mp.spawn(_worker, args=(WORLD_SIZE, _free_port(), per_token, lb_type, result_path), nprocs=WORLD_SIZE, join=True)
    results = [json.load(open(f"{result_path}.{rank}")) for rank in range(WORLD_SIZE)]
    if "skip" in results[0]:
        pytest.skip(f"needs Megatron-LM: {results[0]['skip']}")

    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        ref_grads, ref_aux = pool.apply(_reference, (per_token, lb_type))

    for result in results:
        assert result["padding_ok"]
        assert result["aux_mean"] == pytest.approx(ref_aux, rel=1e-5)
    for rank, result in enumerate(results):
        for row in result["rows"]:
            positions = torch.tensor(row["positions"])
            grads = torch.tensor(row["grads"])
            valid = positions >= 0
            # Per-sample: DDP averages the gradients over the 4 ranks; per-token: Megatron divides
            # them by the same all-reduced token count under both layouts.
            actual = grads[valid] / (1 if per_token else WORLD_SIZE)
            expected = ref_grads[(row["round"], row["mb"])][positions[valid]]
            torch.testing.assert_close(
                actual,
                expected,
                rtol=1e-4,
                atol=1e-7,
                msg=f"rank {rank} round {row['round']}",
            )


def _tp_cp_groups_worker(rank, world_size, port, result_path):
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from megatron.core import mpu
    except ImportError as exc:
        with open(f"{result_path}.{rank}", "w") as f:
            json.dump({"skip": str(exc)}, f)
        return
    from slime.backends.megatron_utils import cp_utils

    # TP = 2 innermost (Megatron's tp-cp-ep-dp-pp order): rank = 2 * dp_cp_rank + tp_rank.
    tp_size, dp_cp_size = 2, world_size // 2
    tp_groups, hybrid = {}, {}
    for dp_cp_rank in range(dp_cp_size):
        group = dist.new_group([2 * dp_cp_rank, 2 * dp_cp_rank + 1])
        if rank // tp_size == dp_cp_rank:
            tp_groups["mine"] = group
    for size in (2, dp_cp_size):
        for tp_rank in range(tp_size):
            for start in range(0, dp_cp_size, size):
                members = [tp_size * r + tp_rank for r in range(start, start + size)]
                group = dist.new_group(members)
                if rank in members:
                    hybrid[size] = group
    mpu.get_tensor_model_parallel_world_size = lambda: tp_size
    mpu.get_tensor_model_parallel_group = lambda: tp_groups["mine"]
    mpu.get_data_parallel_world_size = lambda with_context_parallel=False: dp_cp_size
    mpu.get_context_parallel_world_size = lambda: 1
    mpu.get_hybrid_data_context_parallel_groups = lambda group_size: hybrid[group_size]

    cp_utils.init_dynamic_cp_groups(with_tp_cp_groups=True)
    result = {}
    for size, ctx in sorted(cp_utils._dynamic_cp_contexts.items()):
        value = torch.tensor([float(rank)])
        dist.all_reduce(value, group=ctx.tp_cp_group)
        result[size] = {
            "cp_ranks": dist.get_process_group_ranks(ctx.group),
            "tp_cp_ranks": dist.get_process_group_ranks(ctx.tp_cp_group),
            "sum": value.item(),
        }
    with open(f"{result_path}.{rank}", "w") as f:
        json.dump(result, f)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.unit
def test_dynamic_cp_tp_cp_groups_span_tp_ranks_of_cp_block(tmp_path):
    import torch.multiprocessing as mp

    world_size = 8
    result_path = str(tmp_path / "result")
    mp.spawn(_tp_cp_groups_worker, args=(world_size, _free_port(), result_path), nprocs=world_size, join=True)
    results = [json.load(open(f"{result_path}.{rank}")) for rank in range(world_size)]
    if "skip" in results[0]:
        pytest.skip(f"needs Megatron-LM: {results[0]['skip']}")

    for rank, result in enumerate(results):
        assert sorted(result) == ["1", "2", "4"]
        for size, entry in result.items():
            block = rank // 2 // int(size) * int(size)
            expected = [2 * r + t for r in range(block, block + int(size)) for t in range(2)]
            assert entry["tp_cp_ranks"] == expected
            assert set(entry["cp_ranks"]) <= set(expected) and rank in entry["cp_ranks"]
            assert entry["sum"] == sum(expected)
