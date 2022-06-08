import random
import numpy as np
import torch
from time import sleep


def set_cudnn(device="cuda"):
    torch.backends.cudnn.enabled = device == "cuda"
    torch.backends.cudnn.benchmark = device == "cuda"


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def stop_epoch(time=3):
    try:
        print("can break now")
        for i in range(time):
            sleep(1)
        print("wait for next epoch")
        return False
    except KeyboardInterrupt:
        return True


def compute_loss_accuracy(net, data_loader, criterion, device):
    net.eval()
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            total_loss += criterion(outputs, labels).item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()

    return total_loss / (batch_idx + 1), correct / len(data_loader.dataset)
