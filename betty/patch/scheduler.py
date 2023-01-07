import inspect


def patch_scheduler(scheduler, optimizer):
    kwargs = {}
    sig = inspect.signature(scheduler.__class__.__init__)
    for param in sig.parameters:
        key = param
        if key == "self":
            continue
        elif key == "optimizer":
            kwargs[key] = optimizer
        elif key == "last_epoch":
            kwargs[key] = getattr(scheduler, key) - 1
        elif key == "lr_lambda":
            kwargs[key] = getattr(scheduler, "lr_lambdas")
        else:
            value = getattr(scheduler, key)
            kwargs[key] = value
    new_scheduler = scheduler.__class__(**kwargs)

    return new_scheduler
