from betty.utils import convert_tensor


class Env:
    def __init__(self, env=None, train_data_loader=None, config=None, *args, **kwargs):
        self.env = env
        self.train_data_loader = train_data_loader
        self.config = config

        # device
        self.device = None

        # distributed training
        self._strategy = None
        self._backend = None
        self._world_size = None
        self._rank = None
        self._local_rank = None

    def reset(self):
        pass

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
        self._backend = dictionary["backend"]
        self._world_size = dictionary["world_size"]
        self._rank = dictionary["rank"]
        self._local_rank = dictionary["local_rank"]
