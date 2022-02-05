from optim.patched_optimizer import PatchedOptimizer


class PatchedAdamW(PatchedOptimizer):
    def _update(self):
        raise NotImplementedError