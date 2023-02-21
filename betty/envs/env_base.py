import torch

from betty.patch.data_loader import get_distributed_data_loader
from betty.utils import convert_tensor


class Env:
    env = None
    train_data_loader = None

    device = None

    accelerator = None

    _strategy = None
    _world_size = None
    _rank = None
    _local_rank = None

    def initialize(self):
        if self._strategy == "accelerate":
            from accelerate import Accelerator

            self.accelerator = Accelerator()

        if self.train_data_loader is not None:
            self.train_data_loader = self.patch_data_loader(self.train_data_loader)

    def patch_module(self, module):
        """
        Patch module given the systems configuration (e.g., DDP, FSDP)
        """
        module.to(self.device)
        if self._strategy in ["distributed", "zero"]:
            module = torch.nn.parallel.DistributedDataParallel(
                module=module,
                gradient_as_bucket_view=True,
            )
        elif self._strategy == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            module = FSDP(module, device_id=self.device)
        elif self._strategy == "accelerate":
            module = self.accelerator.prepare(module)

        return module

    def patch_data_loader(self, loader):
        """
        Patch data loader given the systems configuration (e.g., DDP, FSDP)
        """
        if self._strategy in ["distributed", "zero", "fsdp"]:
            patched_loader = get_distributed_data_loader(
                loader, world_size=self._world_size, rank=self._rank
            )
        elif self._strategy == "accelerate":
            patched_loader = self.accelerator.prepare(loader)
        else:
            patched_loader = loader

        return patched_loader

    def configure_device(self, device):
        """
        Set the device for the current problem.
        """
        self.device = device

    def configure_distributed_training(self, dictionary):
        """
        Set the configuration for distributed training.

        :param dictionary: Python dictionary of distributed training provided by Engine.
        :type dictionary: dict
        """
        self._strategy = dictionary["strategy"]
        self._world_size = dictionary["world_size"]
        self._rank = dictionary["rank"]
        self._local_rank = dictionary["local_rank"]
