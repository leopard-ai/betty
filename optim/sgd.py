from optim.patched_optimizer import PatchedOptimizer

class PatchedSGD(PatchedOptimizer):
    def _update(self):
        raise NotImplementedError