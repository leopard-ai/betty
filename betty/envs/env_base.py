class Env:
    def step(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass
