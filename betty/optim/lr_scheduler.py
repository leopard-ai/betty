import inspect

def patch_scheduler(old_scheduler, new_optimizer):
    kwargs = {}
    sig = inspect.signature(old_scheduler.__class__.__init__)
    for param in sig.parameters:
        key = param
        if key == 'self':
            continue
        elif key == 'optimizer':
            kwargs[key] = new_optimizer
        elif key == 'last_epoch':
            kwargs[key] = getattr(old_scheduler, key) - 1
        else:
            value = getattr(old_scheduler, key)
            kwargs[key] = value
    new_scheduler = old_scheduler.__class__(**kwargs)

    return new_scheduler
