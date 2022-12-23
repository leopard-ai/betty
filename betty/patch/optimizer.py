from torch.distributed.optim import ZeroRedundancyOptimizer


def patch_optimizer(optimizer, params, is_zero):
    defaults = optimizer.defaults
    new_optimizer = None
    if is_zero:
        new_optimizer = ZeroRedundancyOptimizer(
            params=params, optimizer_class=optimizer.__class__, **defaults
        )
    else:
        new_optimizer = optimizer.__class__(params, **defaults)

    return new_optimizer
