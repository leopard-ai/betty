import unittest
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem


class ChildNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.w = torch.nn.Parameter(torch.zeros(20))

    def forward(self, inputs):
        outs = inputs @ self.w
        return outs, self.w


class ParentNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.w = torch.nn.Parameter(torch.ones(20))

    def forward(self):
        return self.w


class Outer(ImplicitProblem):
    def training_step(self, batch):
        inputs, targets = batch
        outs = self.inner(inputs)[0]
        loss = F.binary_cross_entropy_with_logits(outs, targets)
        return loss

    def param_callback(self, params):
        for p in params:
            p.data.clamp_(min=1e-8)


class Inner(ImplicitProblem):
    def training_step(self, batch):
        inputs, targets = batch
        outs, params = self.module(inputs)
        loss = (
            F.binary_cross_entropy_with_logits(outs, targets)
            + 0.5
            * (
                params.unsqueeze(0) @ torch.diag(self.outer()) @ params.unsqueeze(1)
            ).sum()
        )
        return loss

    def on_inner_loop_start(self):
        self.module.w.data.zero_()


class ProblemTest(unittest.TestCase):
    def setUp(self):
        device = "cpu"

        # data preparation
        w_gt = np.random.randn(20)
        x = np.random.randn(100, 20)
        y = x @ w_gt + 0.1 * np.random.randn(100)
        y = (y > 0).astype(float)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.5)
        x_train, y_train = (
            torch.from_numpy(x_train).to(device).float(),
            torch.from_numpy(y_train).to(device).float(),
        )
        x_val, y_val = (
            torch.from_numpy(x_val).to(device).float(),
            torch.from_numpy(y_val).to(device).float(),
        )

        # data_loader
        self.train_loader = [(x_train, y_train)]
        self.valid_loader = [(x_val, y_val)]

        # module
        self.train_module = ChildNet().to(device)
        self.valid_module = ParentNet().to(device)

        # optimizer
        self.train_optimizer = torch.optim.SGD(self.train_module.parameters(), lr=0.1)
        self.valid_optimizer = torch.optim.SGD(
            self.valid_module.parameters(), lr=0.1, momentum=0.9
        )

        self.train_config = Config(unroll_steps=10)
        self.valid_config = Config()
        self.engine_config = EngineConfig(train_iters=20)

        # problem
        self.outer = Outer(
            name="outer",
            module=self.valid_module,
            optimizer=self.valid_optimizer,
            train_data_loader=self.valid_loader,
            config=self.valid_config,
            device=device,
        )
        self.inner = Inner(
            name="inner",
            module=self.train_module,
            optimizer=self.train_optimizer,
            train_data_loader=self.train_loader,
            config=self.train_config,
            device=device,
        )

    def test_add_child(self):
        self.outer.add_child(self.inner)
        self.assertTrue(self.inner in self.outer.children)

    def test_add_parent(self):
        self.inner.add_parent(self.outer)
        self.assertTrue(self.outer in self.inner.parents)


if __name__ == "__main__":
    unittest.main()
