"""CPU unit tests for ``slime.backends.megatron_utils.cp_utils.get_sum_of_sample_mean``.

Pins the per-rollout reducer contract: a rollout split into N training
samples (compact / subagent) must contribute exactly one token-weighted
mean to the sum, even when first-fit packing puts those siblings into
different micro-batches at training time.

The CPU-only CI image does not ship megatron — ``_cp_dist_helpers``
stubs ``megatron.core.mpu`` at import time so the subsequent
``cp_utils`` import binds against the stub.

End-to-end report-formula invariance and multi-process distributed
checks live in ``test_metric_report.py`` and ``test_metric_report_dist.py``.
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
    CPContext,
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
    get_sample_cp_contexts,
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
    slice_with_cp,
)


NUM_GPUS = 0


def _make_inputs(per_sample_lengths: list[int]):
    """Build (total_lengths, response_lengths, loss_masks) for samples of the given lengths.

    Each sample has loss_mask = all-ones (so mask sum == length); total length
    is response length + 4 fake prompt tokens (unused by the reducer in
    cp_size==1 mode).
    """
    response_lengths = list(per_sample_lengths)
    total_lengths = [r + 4 for r in response_lengths]
    loss_masks = [torch.ones(r, dtype=torch.float32) for r in response_lengths]
    return total_lengths, response_lengths, loss_masks


def _denoms(*values: int) -> torch.Tensor:
    """Wrap per-sample denoms as the float tensor that the actor side promotes
    them to before calling the reducer."""
    return torch.tensor(values, dtype=torch.float32)


@pytest.mark.unit
def test_default_reduces_to_per_sample_mean():
    """``sample_denoms=None`` reproduces the legacy per-sample-mean."""
    total_lengths, response_lengths, loss_masks = _make_inputs([3, 3, 3])
    reducer = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    # per-sample means: 2, 5, 8 → sum = 15
    assert reducer(x).item() == pytest.approx(15.0)


@pytest.mark.unit
def test_per_rollout_denom_collapses_siblings_into_one_mean():
    """Pre-computed per-rollout mask sums make N sibling samples contribute one
    token-weighted mean instead of N per-sample means."""
    # 4 samples: rollout R0 owns indices 0,1,2 (mask sums 3+3+3=9); rollout R1
    # owns index 3 (mask sum 3). Pre-computed per-sample denom = group sum.
    total_lengths, response_lengths, loss_masks = _make_inputs([3, 3, 3, 3])
    sample_denoms = _denoms(9, 9, 9, 3)
    reducer = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, sample_denoms)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    # R0 token-mean: (1+2+...+9)/9 = 5.  R1 token-mean: (10+11+12)/3 = 11.  Sum = 16.
    assert reducer(x).item() == pytest.approx(16.0)


@pytest.mark.unit
def test_split_across_mbs_recovers_full_per_rollout_mean():
    """The critical contract: when a rollout's samples land in different mbs,
    summing each mb's reducer output equals one whole-step reducer call with
    the same pre-computed denominators. This is exactly the bug that motivated
    the precomputation — if the denom were computed per-mb (partial mask sum),
    the two halves wouldn't add up."""
    # 4 samples (same as above). Whole-step denoms = [9, 9, 9, 3].
    total_lengths, response_lengths, loss_masks = _make_inputs([3, 3, 3, 3])
    sample_denoms = _denoms(9, 9, 9, 3)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    whole = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, sample_denoms)
    whole_value = whole(x).item()

    # mb_a holds samples 0, 1 of R0; mb_b holds sample 2 of R0 and sample 3 (R1).
    # Each mb carries the SAME per-sample denoms (precomputed at step level)
    # — that's what makes the split safe.
    mb_a = get_sum_of_sample_mean(total_lengths[:2], response_lengths[:2], loss_masks[:2], sample_denoms[:2])
    mb_b = get_sum_of_sample_mean(total_lengths[2:], response_lengths[2:], loss_masks[2:], sample_denoms[2:])
    split_value = mb_a(x[:6]).item() + mb_b(x[6:]).item()

    assert split_value == pytest.approx(whole_value)


@pytest.mark.unit
def test_split_with_per_mb_denom_would_be_wrong():
    """Sanity-check the bug we're guarding against: if the caller naively
    computes per-rollout denoms from each mb's own samples (the local mask
    sum, NOT the precomputed whole-rollout sum), the two halves DON'T add up
    to the whole-step value. This pins down WHY the precomputation must
    happen at the step level."""
    total_lengths, response_lengths, loss_masks = _make_inputs([3, 3, 3, 3])
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    whole = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, _denoms(9, 9, 9, 3))
    whole_value = whole(x).item()

    # Wrong denom: each mb only sees its own samples of R0.
    # mb_a's "rollout mask sum" for R0 would be 3+3=6 (instead of 9). mb_b's
    # would be 3. Different from the true whole-rollout total.
    mb_a_wrong = get_sum_of_sample_mean(total_lengths[:2], response_lengths[:2], loss_masks[:2], _denoms(6, 6))
    mb_b_wrong = get_sum_of_sample_mean(total_lengths[2:], response_lengths[2:], loss_masks[2:], _denoms(3, 3))
    wrong_total = mb_a_wrong(x[:6]).item() + mb_b_wrong(x[6:]).item()

    assert wrong_total != pytest.approx(whole_value), (
        "Expected the per-mb denom path to produce a different (incorrect) value; "
        "if these match, the regression test is no longer guarding the precomputation contract."
    )


@pytest.mark.unit
def test_cp_chunking_preserves_per_rollout_mean_report(monkeypatch):
    """Turning CP on must not change the reducer's output.

    Real flow: each CP rank only sees its chunk of the response tokens; the
    reducer's CP>1 branch slices ``loss_mask`` to match. Summing each CP
    rank's reducer output across CP ranks reproduces the cp=1 result, which
    is what train_one_step then divides by ``step_global_batch_size``.
    """
    from megatron.core import mpu as _mpu

    # Use lengths that line up cleanly with the CP chunking
    # (chunk_size = ceil(total_length / (2*cp_size))).
    total_lengths = [12, 12]  # 2 samples
    response_lengths = [8, 8]  # 4 prompt + 8 response each
    loss_masks = [torch.ones(r, dtype=torch.float32) for r in response_lengths]
    sample_denoms = torch.tensor([16.0, 16.0], dtype=torch.float32)  # = sum of both mask totals (one rollout)
    x_full = [
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
    ]
    x_concat = torch.cat(x_full)

    # --- cp=1 baseline ---
    monkeypatch.setattr(_mpu, "get_context_parallel_group", lambda: None, raising=False)
    monkeypatch.setattr(_mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(_mpu, "get_context_parallel_rank", lambda: 0)
    reducer_cp1 = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, sample_denoms)
    baseline = reducer_cp1(x_concat).item()

    # --- cp=2: sum partial reducer outputs across the two CP ranks ---
    monkeypatch.setattr(_mpu, "get_context_parallel_world_size", lambda: 2)
    cp_total = 0.0
    for cp_rank in range(2):
        monkeypatch.setattr(_mpu, "get_context_parallel_rank", lambda r=cp_rank: r)
        # Slice each sample's response-token tensor to the chunks this CP
        # rank owns, mirroring what the forward pass would feed in.
        x_chunks_per_sample = []
        for tl, rl, x in zip(total_lengths, response_lengths, x_full, strict=True):
            prompt_length = tl - rl
            _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(tl, rl)
            chunk_0 = x[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
            chunk_1 = x[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
            x_chunks_per_sample.append(torch.cat([chunk_0, chunk_1]))
        x_for_rank = torch.cat(x_chunks_per_sample)
        reducer_cp2 = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, sample_denoms)
        cp_total += reducer_cp2(x_for_rank).item()

    assert cp_total == pytest.approx(baseline)


def _set_static_cp(monkeypatch, size: int, rank: int) -> None:
    from megatron.core import mpu as _mpu

    monkeypatch.setattr(_mpu, "get_context_parallel_world_size", lambda: size)
    monkeypatch.setattr(_mpu, "get_context_parallel_rank", lambda: rank)
    monkeypatch.setattr(_mpu, "get_context_parallel_group", lambda: None, raising=False)


def _cp_helper_outputs(total_lengths, response_lengths, loss_masks, cp=None):
    """Run every per-sample CP helper; ``cp=None`` reads the static CP state."""
    cps = None if cp is None else [cp] * len(total_lengths)
    offsets = [
        get_logits_and_tokens_offset_with_cp(t, r, cp=cp) for t, r in zip(total_lengths, response_lengths, strict=True)
    ]
    token_slices = [slice_with_cp(torch.arange(t), -1, cp=cp) for t in total_lengths]
    response_slices = [
        slice_log_prob_with_cp(torch.arange(r, dtype=torch.float32), t, r, cp=cp)
        for t, r in zip(total_lengths, response_lengths, strict=True)
    ]
    list_slices = [
        slice_log_prob_with_cp(list(range(r)), t, r, cp=cp)
        for t, r in zip(total_lengths, response_lengths, strict=True)
    ]
    reducer = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, cps=cps)
    return offsets, token_slices, response_slices, list_slices, reducer(torch.cat(response_slices) + 1).item()


@pytest.mark.unit
@pytest.mark.parametrize(("cp_size", "cp_rank"), [(size, rank) for size in (2, 4) for rank in range(size)])
def test_explicit_cp_context_matches_static_cp(monkeypatch, cp_size, cp_rank):
    """An explicit CPContext slices exactly like the same static CP state, and
    does not read the static state at all."""
    # Lengths not divisible by 2 * cp_size exercise the zigzag padding; the
    # short sample leaves some ranks without response tokens.
    total_lengths, response_lengths = [23, 9, 40], [15, 2, 33]
    loss_masks = [(torch.arange(r) % 3 != 0).float() for r in response_lengths]

    _set_static_cp(monkeypatch, cp_size, cp_rank)
    static = _cp_helper_outputs(total_lengths, response_lengths, loss_masks)

    _set_static_cp(monkeypatch, 1, 0)
    explicit = _cp_helper_outputs(
        total_lengths, response_lengths, loss_masks, cp=CPContext(size=cp_size, rank=cp_rank, group=None)
    )

    static_offsets, static_tokens, static_responses, static_lists, static_reduced = static
    offsets, tokens, responses, lists, reduced = explicit
    assert offsets == static_offsets
    assert all(torch.equal(a, b) for a, b in zip(tokens, static_tokens, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(responses, static_responses, strict=True))
    assert lists == static_lists
    assert reduced == static_reduced


@pytest.mark.unit
@pytest.mark.parametrize("cp_size", [2, 4])
def test_cp_slices_from_all_ranks_reassemble_sequence(monkeypatch, cp_size):
    """Rank r holds zigzag chunks r and 2*cp_size-1-r; together the ranks cover
    the padded sequence, and their response slices gather back losslessly."""
    import torch.distributed.nn

    _set_static_cp(monkeypatch, 1, 0)
    total_length, response_length = 23, 15
    group = object()
    cps = [CPContext(size=cp_size, rank=rank, group=group) for rank in range(cp_size)]

    tokens = torch.arange(total_length) + 100
    token_slices = [slice_with_cp(tokens, -1, cp=cp) for cp in cps]
    chunk_size = token_slices[0].size(0) // 2
    chunks = [None] * (2 * cp_size)
    for rank, local in enumerate(token_slices):
        chunks[rank], chunks[2 * cp_size - 1 - rank] = local[:chunk_size], local[chunk_size:]
    padded = torch.cat(chunks)
    assert torch.equal(padded[:total_length], tokens)
    assert (padded[total_length:] == -1).all()

    # Each rank's all_gather input is its slice zero-filled to the full
    # response; summing them stands in for the CP all-reduce.
    reduce_groups = []

    def fake_all_reduce(tensor, group):
        reduce_groups.append(group)
        return tensor

    monkeypatch.setattr(torch.distributed.nn, "all_reduce", fake_all_reduce)
    response = torch.arange(response_length, dtype=torch.float32) + 1
    gathered = sum(
        all_gather_with_cp(
            slice_log_prob_with_cp(response, total_length, response_length, cp=cp),
            total_length,
            response_length,
            cp=cp,
        )
        for cp in cps
    )
    assert torch.equal(gathered.detach(), response)
    assert reduce_groups == [group] * cp_size


@pytest.mark.unit
def test_get_sum_of_sample_mean_accepts_per_sample_contexts(monkeypatch):
    """Samples may sit on different CP contexts: each one is reduced over its
    own chunks, matching the reducer built for that context alone."""
    _set_static_cp(monkeypatch, 1, 0)
    total_lengths, response_lengths = [12, 12], [8, 8]
    loss_masks = [torch.ones(8), torch.tensor([1.0, 0, 1, 1, 0, 1, 1, 1])]
    sample_denoms = _denoms(16, 16)
    x_full = [torch.arange(8, dtype=torch.float32) + 1, torch.arange(8, dtype=torch.float32) * 10]
    cp1, cp2 = CPContext(size=1, rank=0, group=None), CPContext(size=2, rank=1, group=None)

    x_cp2 = slice_log_prob_with_cp(x_full[1], total_lengths[1], response_lengths[1], cp=cp2)
    mixed = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, sample_denoms, cps=[cp1, cp2])
    first = get_sum_of_sample_mean(total_lengths[:1], response_lengths[:1], loss_masks[:1], sample_denoms[:1])
    second = get_sum_of_sample_mean(
        total_lengths[1:], response_lengths[1:], loss_masks[1:], sample_denoms[1:], cps=[cp2]
    )

    expected = first(x_full[0]) + second(x_cp2)
    assert mixed(torch.cat([x_full[0], x_cp2])).item() == pytest.approx(expected.item())


@pytest.mark.unit
def test_sample_cp_contexts_follow_rank_local_samples(monkeypatch):
    """``total_lengths`` arrives un-partitioned; the context count must follow
    the rank-local samples."""
    _set_static_cp(monkeypatch, 2, 1)
    rollout_data = {"total_lengths": [5, 6, 7, 8], "response_lengths": [2, 3]}
    assert get_sample_cp_contexts(rollout_data) == [CPContext(size=2, rank=1, group=None)] * 2

    assert get_sample_cp_contexts({**rollout_data, "cp_sizes": [2, 2]}) == [CPContext(size=2, rank=1, group=None)] * 2
    with pytest.raises(NotImplementedError):
        get_sample_cp_contexts({**rollout_data, "cp_sizes": [2, 4]})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
