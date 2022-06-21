class Env:

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space
