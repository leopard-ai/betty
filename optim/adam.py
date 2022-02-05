from optim.patched_optimizer import PatchedOptimizer


class PatchedAdam(PatchedOptimizer):
    def _update(self):
        raise NotImplementedError