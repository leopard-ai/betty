import torch


def convert_tensor(item, device=None, fp16=False):
    if not isinstance(item, torch.Tensor):
        return item
    return item.to(device)


def get_grad_norm(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.data.float().norm()
        total_norm += param_norm.item() ** 2

    if (
        total_norm == float("inf")
        or total_norm == -float("inf")
        or total_norm != total_norm
    ):
        total_norm = -1

    return total_norm


def get_weight_norm(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    total_norm = 0.0
    for p in parameters:
        param_norm = torch.norm(p, dtype=torch.float32)
        total_norm += param_norm.item() ** 2

    if (
        total_norm == float("inf")
        or total_norm == -float("inf")
        or total_norm != total_norm
    ):
        total_norm = -1

    return total_norm


def flatten_list(regular_list):
    """[summary]
    Flatten list of lists
    """
    if type(regular_list[0] == list):
        return [item for sublist in regular_list for item in sublist]
    return regular_list


def get_param_index(param, param_list):
    param_list = list(param_list)
    for idx, p in enumerate(param_list):
        if p is param:
            return idx
    print("no corresponding parameter found!")


def get_multiplier(problem):
    if problem.leaf:
        return 1

    assert len(problem.children) > 0
    # stack to store all the nodes of tree
    s1 = []
    # stack to store all the leaf nodes
    s2 = []

    s1.append((problem, 1))
    while len(s1) != 0:
        curr, multiplier = s1.pop(0)

        if len(curr.children) != 0:
            for child in curr.children:
                s1.append((child, multiplier * curr.config.step))
        else:
            s2.append(multiplier)

    assert all(x == s2[0] for x in s2)
    return s2[0]


def log_from_loss_dict(loss_dict):
    outputs = []
    for key, values in loss_dict.items():
        if isinstance(values, dict) or isinstance(values, list):
            for value_idx, value in enumerate(values):
                full_key = key + "_" + str(value_idx)
                if torch.is_tensor(value):
                    value = value.item()
                output = f"{full_key}: {value}"
                outputs.append(output)
        else:
            if torch.is_tensor(values):
                values = values.item()
            output = f"{key}: {values}"
            outputs.append(output)
    return " || ".join(outputs)


def to_vec(tensor_list, alpha=1.0):
    return torch.cat([alpha * t.reshape(-1) for t in tensor_list])


def count_parameters(tensor_list):
    return sum([tensor.numel() for tensor in tensor_list])


def neg_with_none(a):
    if a is None:
        return None
    else:
        return -a


def replace_none_with_zero(tensor_list, reference):
    out = []
    for t, r in zip(tensor_list, reference):
        fixed = t if t is not None else torch.zeros_like(r)
        out.append(fixed)
    return tuple(out)
