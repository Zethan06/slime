"""CPU unit tests for the dynamic context parallel (DCP) wiring.

Covers the pieces between the DCP scheduler (``build_dynamic_cp_schedule``) and
the model: the per-rank payloads ``BatchBuilder.split_by_dp`` ships, the
per-sample ``cp_sizes`` ``process_rollout_data`` derives, the CP group lookup
in ``cp_utils``, the per-mbs ``cp_size`` of ``DataIterator`` and the
``PackedSeqParams`` ``get_batch`` hands to the model.

``_cp_dist_helpers`` stubs ``megatron.core.mpu``; the process groups below are
plain objects, so no ``torch.distributed`` setup is needed.
"""

from __future__ import annotations

import importlib
import sys
import types
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

import _cp_dist_helpers  # noqa: F401
import pytest
import ray
import torch

from slime.backends.megatron_utils import cp_utils
from slime.backends.megatron_utils.cp_utils import CPContext, get_cp_context, slice_with_cp
from slime.utils.data import process_rollout_data
from slime.utils.dp_schedule import build_dynamic_cp_schedule

NUM_GPUS = 0


@dataclass
class _PackedSeqParams:
    """The ``megatron.core.packed_seq_params.PackedSeqParams`` fields slime sets."""

    qkv_format: str = None
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_kv: torch.Tensor = None
    max_seqlen_q: int = None
    max_seqlen_kv: int = None
    local_cp_size: int = None
    cp_group: object = None


class _Group:
    def __init__(self, ranks, name):
        self.ranks = list(ranks)
        self.name = name

    def __repr__(self):
        return f"_Group({self.name}, {self.ranks})"


@pytest.fixture
def data_module(monkeypatch):
    packed = types.ModuleType("megatron.core.packed_seq_params")
    packed.PackedSeqParams = _PackedSeqParams
    monkeypatch.setitem(sys.modules, "megatron.core.packed_seq_params", packed)
    # Re-import so `data` binds this PackedSeqParams even if another test imported it first.
    sys.modules.pop("slime.backends.megatron_utils.data", None)
    data = importlib.import_module("slime.backends.megatron_utils.data")

    monkeypatch.setattr(data, "accelerator", SimpleNamespace(device=lambda: "cpu", current_device=lambda: "cpu"))
    monkeypatch.setattr(cp_utils.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(cp_utils.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None, raising=False)
    yield data
    sys.modules.pop("slime.backends.megatron_utils.data", None)


@pytest.fixture
def static_cp(monkeypatch):
    def set_static_cp(size, rank):
        group = _Group(range(size), "static") if size > 1 else None
        monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: size)
        monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_rank", lambda: rank)
        monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_group", lambda: group, raising=False)
        return group

    monkeypatch.setattr(cp_utils, "_dynamic_cp_contexts", {})
    return set_static_cp


def _fake_dist_and_mpu(monkeypatch, *, world_rank, dp_cp_size, static_cp_size):
    """Stub the groups ``init_dynamic_cp_groups`` reads for one rank (TP = PP = 1)."""
    world_size = dp_cp_size
    created = []

    def new_group(ranks):
        group = _Group(ranks, "singleton")
        created.append(group)
        return group

    def block(size):
        start = world_rank // size * size
        return range(start, start + size)

    static_group = _Group(block(static_cp_size), "static")
    hybrid = {size: _Group(block(size), f"hybrid{size}") for size in (2, 4, 8, 16) if size < dp_cp_size}
    dp_cp_group = _Group(range(dp_cp_size), "dp_cp")

    def get_hybrid_groups(group_size):
        # Megatron returns the DP-with-CP group for the full DP x CP size.
        return dp_cp_group if group_size == dp_cp_size else hybrid[group_size]

    monkeypatch.setattr(
        cp_utils,
        "dist",
        SimpleNamespace(
            new_group=new_group,
            get_rank=lambda group=None: world_rank if group is None else group.ranks.index(world_rank),
            get_world_size=lambda: world_size,
            get_process_group_ranks=lambda group: group.ranks,
        ),
    )
    monkeypatch.setattr(
        cp_utils.mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel=False: dp_cp_size,
        raising=False,
    )
    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: static_cp_size)
    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_rank", lambda: world_rank % static_cp_size)
    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_group", lambda: static_group, raising=False)
    monkeypatch.setattr(cp_utils.mpu, "get_hybrid_data_context_parallel_groups", get_hybrid_groups, raising=False)
    return created, static_group, hybrid, dp_cp_group


@pytest.mark.unit
@pytest.mark.parametrize("world_rank", range(8))
def test_dynamic_cp_context_lookup(monkeypatch, static_cp, world_rank):
    """Each size maps to the aligned block holding this rank: the singleton group,
    the static CP group, a hybrid group, or the DP-with-CP group."""
    created, static_group, hybrid, dp_cp_group = _fake_dist_and_mpu(
        monkeypatch, world_rank=world_rank, dp_cp_size=8, static_cp_size=2
    )
    cp_utils.init_dynamic_cp_groups()

    # One singleton group per rank, created collectively.
    assert [group.ranks for group in created] == [[rank] for rank in range(8)]
    expected_groups = {1: created[world_rank], 2: static_group, 4: hybrid[4], 8: dp_cp_group}
    for size, group in expected_groups.items():
        cp = get_cp_context(size)
        assert cp == CPContext(size=size, rank=world_rank % size, group=group)
        assert get_cp_context(size) is cp
    # Under dynamic CP the static CP group is never an implicit default.
    with pytest.raises(RuntimeError, match="--dynamic-cp"):
        get_cp_context()
    with pytest.raises(KeyError):
        get_cp_context(16)


@pytest.mark.unit
def test_dynamic_cp_with_static_cp_one_uses_singleton_group(monkeypatch, static_cp):
    """With static CP size 1, size 1 still gets a real (singleton) group: a None group
    would make Megatron RoPE fall back to the static CP group."""
    created, _, hybrid, dp_cp_group = _fake_dist_and_mpu(monkeypatch, world_rank=3, dp_cp_size=4, static_cp_size=1)
    cp_utils.init_dynamic_cp_groups()

    assert get_cp_context(1) == CPContext(size=1, rank=0, group=created[3])
    assert get_cp_context(2) == CPContext(size=2, rank=1, group=hybrid[2])
    assert get_cp_context(4) == CPContext(size=4, rank=3, group=dp_cp_group)


@pytest.mark.unit
def test_static_cp_rejects_other_sizes(static_cp):
    static_cp(2, 1)
    assert get_cp_context(2) == get_cp_context()
    with pytest.raises(NotImplementedError):
        get_cp_context(4)


@pytest.mark.unit
def test_data_iterator_carries_cp_size_per_microbatch(data_module):
    rollout_data = {"tokens": ["a", "b", "c"]}

    iterator = data_module.DataIterator(rollout_data, [[0, 1], [2]], micro_batch_cp_sizes=[1, 4])
    assert iterator.get_next(["tokens"]) == {"tokens": ["a", "b"], "cp_size": 1}
    assert iterator.get_next(["tokens"]) == {"tokens": ["c"], "cp_size": 4}

    static = data_module.DataIterator(rollout_data, [[0, 1], [2]])
    assert static.get_next(["tokens"]) == {"tokens": ["a", "b"]}

    iterators = data_module.get_data_iterator(
        {**rollout_data, "micro_batch_indices": [[2], [0, 1]], "micro_batch_cp_sizes": [2, 1]}
    )
    assert [iterators[0].get_next(["tokens"])["cp_size"] for _ in range(2)] == [2, 1]


def _get_batch_inputs(total_lengths, response_lengths):
    return {
        "tokens": [torch.arange(n) + 1 for n in total_lengths],
        "loss_masks": [torch.ones(r, dtype=torch.int) for r in response_lengths],
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
    }


@pytest.mark.unit
def test_get_batch_static_cp_keeps_model_cp_groups(data_module, static_cp):
    """Static CP must not set local_cp_size / cp_group: that would override the
    model's own (possibly hierarchical a2a+p2p) CP groups."""
    static_cp(2, 1)
    rollout_data = _get_batch_inputs([10], [4])
    iterator = data_module.DataIterator(rollout_data, [[0]])
    keys = ["tokens", "loss_masks", "total_lengths", "response_lengths"]

    batch = data_module.get_batch(iterator, keys, pad_multiplier=1)
    params = batch["packed_seq_params"]
    assert params.local_cp_size is None and params.cp_group is None
    assert batch["cp"] == get_cp_context()
    assert params.cu_seqlens_q.tolist() == [0, 12]


@pytest.mark.unit
@pytest.mark.parametrize("static_size", [1, 2])
def test_get_batch_dynamic_cp_passes_microbatch_group(data_module, static_cp, static_size):
    static_cp(static_size, 0)
    groups = {1: _Group([1], "singleton"), 2: _Group([0, 1], "two"), 4: _Group([0, 1, 2, 3], "four")}
    cp_utils._dynamic_cp_contexts.update(
        {size: CPContext(size=size, rank=1, group=group) for size, group in groups.items()}
    )
    total_lengths, response_lengths = [10, 7], [4, 3]
    rollout_data = _get_batch_inputs(total_lengths, response_lengths)
    iterator = data_module.DataIterator(rollout_data, [[0, 1], [0], [1]], micro_batch_cp_sizes=[4, 2, 1])
    keys = ["tokens", "loss_masks", "total_lengths", "response_lengths"]

    for indices, size in [([0, 1], 4), ([0], 2), ([1], 1)]:
        batch = data_module.get_batch(iterator, keys, pad_multiplier=1)
        cp = CPContext(size=size, rank=1, group=groups[size])
        params = batch["packed_seq_params"]
        assert batch["cp"] == cp
        assert params.local_cp_size == size and params.cp_group is groups[size]
        expected_tokens = torch.cat([slice_with_cp(rollout_data["tokens"][i], 0, cp=cp) for i in indices])
        assert torch.equal(batch["tokens"][0], expected_tokens)
        # THD cu_seqlens are in unsliced (whole-sequence) units; CP pads to 2 * size chunks.
        padded_lengths = [
            -(-total_lengths[i] // (2 * size)) * 2 * size if size > 1 else total_lengths[i] for i in indices
        ]
        assert params.cu_seqlens_q.tolist() == [0] + torch.tensor(padded_lengths).cumsum(0).tolist()
        assert batch["full_loss_masks"].shape == batch["tokens"].shape


def _builder(monkeypatch, *, dp_size, cp_size, transport="object-store"):
    from slime.data import batch_builder

    monkeypatch.setattr(batch_builder.ray, "put", lambda value, **kwargs: value)
    builder = batch_builder.BatchBuilder.__new__(batch_builder.BatchBuilder)
    builder.args = SimpleNamespace(global_batch_size=8, max_tokens_per_gpu=64, rollout_data_transport=transport)
    builder.rollout_id = 0
    builder.train_parallel_config = {
        "dp_size": dp_size,
        "cp_size": cp_size,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
        "dynamic_cp": True,
    }
    return builder


def _rollout_batch():
    # Lengths spread over 1..4 ranks' worth of tokens so the scheduler mixes CP sizes.
    total_lengths = [200, 150, 90, 40, 30, 20, 60, 10]
    return {
        "tokens": [list(range(n)) for n in total_lengths],
        "response_lengths": [n // 2 for n in total_lengths],
        "loss_masks": [[1] * (n // 2) for n in total_lengths],
        "rewards": [float(i) for i in range(len(total_lengths))],
        "raw_reward": [float(i) for i in range(len(total_lengths))],
        "rollout_ids": list(range(len(total_lengths))),
    }


@pytest.mark.unit
def test_split_by_dp_dynamic_cp_ships_one_payload_per_dp_cp_rank(monkeypatch):
    builder = _builder(monkeypatch, dp_size=2, cp_size=2)
    data = _rollout_batch()
    total_lengths = [len(t) for t in data["tokens"]]
    partitions, micro_batch_indices, micro_batch_cp_sizes, num_microbatches, _ = build_dynamic_cp_schedule(
        builder.args, builder.train_parallel_config, total_lengths, global_batch_size=8, rollout_indices=list(range(8))
    )
    assert len({k for sizes in micro_batch_cp_sizes for k in sizes}) > 1, "fixture should mix CP sizes"

    payloads = [box.inner for box in builder.split_by_dp(data)]

    assert len(payloads) == 4
    for rank, payload in enumerate(payloads):
        assert payload["partition"] == partitions[rank]
        assert payload["micro_batch_indices"] == micro_batch_indices[rank]
        assert payload["micro_batch_cp_sizes"] == micro_batch_cp_sizes[rank]
        assert payload["num_microbatches"] == num_microbatches
        assert [t.tolist() for t in payload["tokens"]] == [data["tokens"][i] for i in partitions[rank]]
        # Shipped whole, partitioned on the train side.
        assert payload["total_lengths"] == total_lengths


@pytest.mark.unit
def test_split_by_dp_dynamic_cp_commits_cp_sizes_in_straw_schedule(monkeypatch):
    from slime.data import batch_builder

    builder = _builder(monkeypatch, dp_size=1, cp_size=4, transport="straw")
    builder.batch_id, builder.plan_digest = "batch:0", "digest"
    monkeypatch.setattr(
        batch_builder,
        "pack_rollout_payload",
        lambda value, args, rollout_id: SimpleNamespace(manifest=value, root="root"),
    )
    committed = []
    builder._commit_ready = lambda ranks, schedule: committed.append((ranks, schedule))

    refs = [box.inner for box in builder.split_by_dp(_rollout_batch())]

    [(ranks, schedule)] = committed
    assert [ref.rank for ref in refs] == [0, 1, 2, 3]
    assert ranks == refs
    assert schedule["micro_batch_cp_sizes"] == [ref.manifest["micro_batch_cp_sizes"] for ref in refs]
    assert schedule["partitions"] == [ref.manifest["partition"] for ref in refs]


@pytest.mark.unit
def test_process_rollout_data_derives_per_sample_cp_sizes(monkeypatch):
    monkeypatch.setattr(ray, "get", lambda ref: ref)
    builder = _builder(monkeypatch, dp_size=2, cp_size=2)
    refs = builder.split_by_dp(_rollout_batch())
    per_rank = [process_rollout_data(refs, rank, 4) for rank in range(4)]

    # A sample runs on a block of k ranks, so it is held by exactly k ranks, each
    # recording that k.
    holders = Counter(sample for rollout_data in per_rank for sample in rollout_data["partition"])
    for rollout_data in per_rank:
        assert len(rollout_data["cp_sizes"]) == len(rollout_data["response_lengths"])
        assert rollout_data["cp_sizes"] == [holders[sample] for sample in rollout_data["partition"]]
        assert rollout_data["total_lengths"] == [len(t) for t in rollout_data["tokens"]]


@pytest.mark.unit
def test_process_rollout_data_cp_sizes_follow_sample_indices(monkeypatch):
    monkeypatch.setattr(ray, "get", lambda ref: ref)
    payload = {
        "partition": [0, 1, 2],
        "total_lengths": [3, 4, 5],
        "micro_batch_indices": [[1], [0, 2]],
        "micro_batch_cp_sizes": [4, 2],
    }

    assert process_rollout_data([SimpleNamespace(inner=payload)], 0, 1)["cp_sizes"] == [2, 4, 2]


@pytest.mark.unit
def test_process_rollout_data_static_has_no_cp_sizes(monkeypatch):
    monkeypatch.setattr(ray, "get", lambda ref: ref)
    payload = {"partition": [0], "total_lengths": [3, 4], "micro_batch_indices": [[0]]}

    assert "cp_sizes" not in process_rollout_data([SimpleNamespace(inner=payload)], 0, 1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
