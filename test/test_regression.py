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


class RegressionTest(unittest.TestCase):
    def setUp(self):
        device = "cpu"
        self.device = device

        # data preparation
        w_gt = np.random.randn(20)
        x = np.random.randn(1000, 20)
        y = x @ w_gt + 0.1 * np.random.randn(1000)
        y = (y > 0).astype(float)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.5)
        x_train, y_train = (
            torch.from_numpy(x_train).to(self.device).float(),
            torch.from_numpy(y_train).to(self.device).float(),
        )
        x_val, y_val = (
            torch.from_numpy(x_val).to(self.device).float(),
            torch.from_numpy(y_val).to(self.device).float(),
        )

        # data_loader
        self.train_loader = [(x_train, y_train)]
        self.valid_loader = [(x_val, y_val)]

        # module
        self.valid_module = ParentNet().to(device)

        # optimizer
        self.valid_optimizer = torch.optim.SGD(
            self.valid_module.parameters(), lr=1.0, momentum=0.9
        )

        self.valid_config = Config()
        self.engine_config = EngineConfig(train_iters=2000)

        # problem
        self.outer = Outer(
            name="outer",
            module=self.valid_module,
            optimizer=self.valid_optimizer,
            train_data_loader=self.valid_loader,
            config=self.valid_config,
            device=device,
        )

    def test_darts(self):
        self.train_config = Config(unroll_steps=100)
        self.train_module = ChildNet().to(self.device)
        self.train_optimizer = torch.optim.SGD(self.train_module.parameters(), lr=0.1)
        self.inner = Inner(
            name="inner",
            module=self.train_module,
            optimizer=self.train_optimizer,
            train_data_loader=self.train_loader,
            config=self.train_config,
            device=self.device,
        )
        problems = [self.outer, self.inner]
        u2l = {self.outer: [self.inner]}
        l2u = {self.inner: [self.outer]}
        dependencies = {"l2u": l2u, "u2l": u2l}

        self.engine = Engine(
            config=self.engine_config, problems=problems, dependencies=dependencies
        )
        self.engine.run()
        loss = self.outer.training_step(self.outer.cur_batch)
        self.assertTrue(loss < 0.48)

    def test_cg(self):
        self.train_config = Config(
            type="cg", cg_iterations=3, cg_alpha=0.1, unroll_steps=100
        )
        self.train_module = ChildNet().to(self.device)
        self.train_optimizer = torch.optim.SGD(self.train_module.parameters(), lr=0.1)
        self.inner = Inner(
            name="inner",
            module=self.train_module,
            optimizer=self.train_optimizer,
            train_data_loader=self.train_loader,
            config=self.train_config,
            device=self.device,
        )
        problems = [self.outer, self.inner]
        u2l = {self.outer: [self.inner]}
        l2u = {self.inner: [self.outer]}
        dependencies = {"l2u": l2u, "u2l": u2l}

        self.engine = Engine(
            config=self.engine_config, problems=problems, dependencies=dependencies
        )
        self.engine.run()
        loss = self.outer.training_step(self.outer.cur_batch)
        self.assertTrue(loss < 0.48)

    def test_neumann(self):
        self.train_config = Config(
            type="neumann", neumann_iterations=5, unroll_steps=100
        )
        self.train_module = ChildNet().to(self.device)
        self.train_optimizer = torch.optim.SGD(self.train_module.parameters(), lr=0.1)
        self.inner = Inner(
            name="inner",
            module=self.train_module,
            optimizer=self.train_optimizer,
            train_data_loader=self.train_loader,
            config=self.train_config,
            device=self.device,
        )
        problems = [self.outer, self.inner]
        u2l = {self.outer: [self.inner]}
        l2u = {self.inner: [self.outer]}
        dependencies = {"l2u": l2u, "u2l": u2l}

        self.engine = Engine(
            config=self.engine_config, problems=problems, dependencies=dependencies
        )
        self.engine.run()
        loss = self.outer.training_step(self.outer.cur_batch)
        self.assertTrue(loss < 0.48)


if __name__ == "__main__":
    unittest.main()
