import inspect

import torch
import torch.distributed as dist


class DistributedBatchSampler(torch.utils.data.BatchSampler):
    """_summary_

    Args:
        data (_type_): _description_
    """

    def __init__(self, batch_sampler, world_size=None, rank=None):
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        self.world_size = (
            world_size if world_size is not None else dist.get_world_size()
        )
        self.rank = rank if rank is not None else dist.get_rank()

        self.epoch = 0

        self.batch_sampler = batch_sampler
        self.batch_size = batch_sampler.batch_size
        self.drop_last = batch_sampler.drop_last
        self.residual = len(self.batch_sampler) % self.world_size + 1

    def __len__(self):
        length = len(self.batch_sampler) // self.world_size
        if len(self.batch_sampler) % self.world_size != 0 and not self.drop_last:
            length += 1

        return length

    def __iter__(self):
        cached_batch = []
        for idx, batch in enumerate(self.batch_sampler):
            if not self.drop_last and idx < self.residual:
                cached_batch += batch
            if idx % self.world_size == self.rank:
                batch_to_yield = batch
            if (
                idx % self.world_size == self.world_size - 1
                and len(batch) == self.batch_size
            ):
                yield batch_to_yield

        if not self.drop_last:
            while idx % self.world_size != 0:
                if len(batch) < self.batch_size:
                    batch += cached_batch
                if idx % self.world_size == self.rank:
                    batch_to_yield = batch[: self.batch_size]
                    batch = batch[self.batch_size :]
                idx += 1
            yield batch_to_yield
            batch = None

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedDataLoader(torch.utils.data.DataLoader):
    """_summary_

    Args:
        torch (_type_): _description_
    """

    def __init__(self, dataset, generator=None, seed=0, **kwargs):
        super().__init__(dataset, **kwargs)

        self.generator = generator
        self.epoch = 0
        self.seed = seed

    def set_epoch(self, epoch):
        self.epoch = epoch

        if self.generator is not None:
            self.generator.manual_seed(self.seed + self.epoch)


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
    new_dataloader.set_epoch(0)

    return new_dataloader
