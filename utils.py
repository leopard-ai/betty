from typing import List

import torch
import torch.nn as nn
from torch import Tensor


def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])


def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)


def _get_nested_attr(obj: nn.Module, names: List[str]) -> None:
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        _get_nested_attr(getattr(obj, names[0]), names[1:])


def make_split_names(lst):
    return [name.split('.') for name in lst]


def swap_state(mod: nn.Module, split_names: List[str], elems):
    result = []
    for split_name, elem in zip(split_names, elems):
        result.append(_get_nested_attr(mod, split_name))
        _del_nested_attr(mod, split_name)
        _set_nested_attr(mod, split_name, elem)
    return result


def flatten_list(regular_list):
    """[summary]
    Flatten list of lists
    """
    if type(regular_list[0] == list):
        return [item for sublist in regular_list for item in sublist]
    return regular_list
