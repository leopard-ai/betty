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
