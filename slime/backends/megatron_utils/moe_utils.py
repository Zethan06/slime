"""MoE routing and aux-loss scaling under dynamic context parallel (``--dynamic-cp``).

Megatron's routers and its MoE aux-loss gradient scale assume the static CP layout,
where every micro-batch is split over the static TP x CP group. Under dynamic CP each
micro-batch runs on its own TP x CP group (``CPContext.tp_cp_group``), so slime points
the routers at that group and sets the aux-loss gradient scale per micro-batch.
"""

from argparse import Namespace
from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.moe.router import Router

from slime.utils.types import RolloutBatch

from .cp_utils import CPContext


def set_routers_for_dynamic_cp(model: torch.nn.Module, cp: CPContext) -> None:
    """Prepare every MoE router of ``model`` for the micro-batch ``cp`` about to run.

    The routers all-reduce their aux-loss token counts over ``tp_cp_group`` and divide the
    z-loss by its size; point them at the micro-batch's TP x CP group. It stays set until
    the next micro-batch, so a forward recomputed in this micro-batch's backward
    (activation recompute) uses it as well.

    With per-token loss a router also weights its aux losses by a token count, which does
    not add up over a dynamic CP group: with a padding mask the aux-loss path counts the
    whole group's tokens (Megatron's local count aliases the all-reduced one) and the z-loss
    path the rank's. Turn that off; :func:`with_moe_aux_loss_scale` applies the weight.
    """
    assert cp.tp_cp_group is not None, f"no TP x CP group for dynamic CP size {cp.size}"
    for module in model.modules():
        if isinstance(module, Router):
            module.tp_cp_group = cp.tp_cp_group
            module.calculate_per_token_loss = False


def get_moe_aux_loss_step_scale(
    args: Namespace,
    loss_scale: torch.Tensor,
    micro_batch_cp_sizes: Sequence[int],
) -> torch.Tensor:
    """Return the MoE aux-loss gradient scale of a training step under dynamic CP.

    With per-sample loss, Megatron averages the aux loss over the step's micro-batches:
    it sets the scale to ``loss_scale * cp_size / num_microbatches`` and DDP averages the
    gradients over the W = DP x CP ranks, i.e. ``loss_scale * W / num_global_microbatches``
    with the ``W / cp_size * num_microbatches`` micro-batches of static CP. Dynamic CP
    micro-batches run on groups of different sizes, so count them across ranks: a rank
    counts ``1 / cp_size`` per micro-batch. ``micro_batch_cp_sizes`` are this rank's.

    With per-token loss the step scale stays ``loss_scale``; see :func:`with_moe_aux_loss_scale`.
    """
    if args.calculate_per_token_loss:
        return loss_scale
    # Exact in float64: the CP sizes are powers of 2.
    num_global_microbatches = torch.tensor(
        sum(1 / cp_size for cp_size in micro_batch_cp_sizes), dtype=torch.float64, device=loss_scale.device
    )
    dist.all_reduce(num_global_microbatches, group=mpu.get_data_parallel_group(with_context_parallel=True))
    return loss_scale * (mpu.get_data_parallel_world_size(with_context_parallel=True) / num_global_microbatches.item())


class _SetMoEAuxLossScaleInBackward(torch.autograd.Function):
    """Identity on the main loss whose backward sets the MoE aux-loss gradient scale.

    It is the first node of the micro-batch's backward, ahead of every router's aux-loss
    node, which reads the scale.
    """

    @staticmethod
    def forward(ctx, loss: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(scale)
        # A new tensor: Megatron rescales the returned loss in place.
        return loss.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (scale,) = ctx.saved_tensors
        MoEAuxLossAutoScaler.set_loss_scale(scale)
        return grad_output, None


def with_moe_aux_loss_scale(
    loss_func: Callable[[torch.Tensor], tuple],
    args: Namespace,
    batch: RolloutBatch,
    step_scale: torch.Tensor,
) -> Callable[[torch.Tensor], tuple]:
    """Wrap a Megatron loss function to set the MoE aux-loss gradient scale of its micro-batch.

    Megatron sets the scale with the static CP formula after the loss function returns;
    the wrapped loss resets it at the start of the micro-batch's backward.

    With per-token loss Megatron divides every gradient by the step's loss-token count, and
    at CP = 1 a router undoes that for its aux losses by weighting them with the
    micro-batch's token count. Under dynamic CP the routers' own weighting is off
    (:func:`set_routers_for_dynamic_cp`), so the scale carries the micro-batch's (valid)
    token count instead, the same on every rank of its group.
    """
    scale = step_scale
    if args.calculate_per_token_loss:
        scale = step_scale * sum(batch["total_lengths"])

    def loss_func_with_moe_aux_loss_scale(output_tensor: torch.Tensor) -> tuple:
        loss, *outputs = loss_func(output_tensor)
        return (_SetMoEAuxLossScaleInBackward.apply(loss, scale), *outputs)

    return loss_func_with_moe_aux_loss_scale
