import inspect

import torch
import torch.distributed as dist

from .distributed_data_loader import DistributedBatchSampler, DistributedDataLoader


def get_distributed_data_loader(
    dataloader,
    world_size=None,
    rank=None,
):
    world_size = world_size if world_size is not None else dist.get_world_size()
    rank = rank if rank is not None else dist.get_rank()

    assert world_size > 1

    dataset = dataloader.dataset

    # sampler
    sampler_is_batch_sampler = isinstance(
        dataloader.sampler, torch.utils.data.BatchSampler
    )
    batch_sampler = (
        dataloader.sampler if sampler_is_batch_sampler else dataloader.batch_sampler
    )
    sampler = batch_sampler.sampler
    new_batch_sampler = DistributedBatchSampler(
        batch_sampler,
        world_size=world_size,
        rank=rank,
    )

    # generator
    generator = getattr(dataloader, "generator", None)
    if hasattr(sampler, "generator"):
        if sampler.generator is None:
            sampler.generator = torch.Generator()
        generator = sampler.generator

    ignore_keys = [
        "batch_size",
        "shuffle",
        "drop_last",
        "sampler",
        "batch_sampler",
    ]

    signature = inspect.signature(dataloader.__class__.__init__)
    kwargs = {
        k: getattr(dataloader, k, v.default)
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty and k not in ignore_keys
    }
    kwargs_update = {"generator": generator}
    if sampler_is_batch_sampler:
        kwargs_update.update(
            {
                "sampler": new_batch_sampler,
                "batch_size": getattr(dataloader, "batch_size", 1),
            }
        )
    else:
        kwargs_update.update({"batch_sampler": new_batch_sampler})
    kwargs.update(kwargs_update)

    new_dataloader = DistributedDataLoader(
        dataset=dataset,
        **kwargs,
    )

    return new_dataloader
