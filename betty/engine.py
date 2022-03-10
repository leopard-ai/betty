import typing

from betty.module import Module


class Engine:
    def __init__(self, config, problems, dependencies=None):
        # config
        self.config = config if config is not None else {}
        self.train_iters = 0
        self.valid_step = 0

        # problem
        self.problems = problems
        self.leaves = []

        # dependencies
        self.dependencies = dependencies

        # initialize
        self.initialize()

    def parse_config(self):
        self.train_iters = self.config.get('train_iters', 50000)
        self.valid_step = self.config.get('valid_step', 200)

    def train_step(self):
        for leaf in self.leaves:
            leaf.step()

    def run(self):
        self.train()
        for it in range(1, self.train_iters + 1):
            self.train_step()

            if it % self.valid_step == 0:
                if self.is_implemented('validation'):
                    self.eval()
                    self.validation()
                    self.train()

    def initialize(self):
        """[summary]
        Initialize dependencies (computational graph) between problems.
        """
        # parse config
        self.parse_config()

        # Set dependencies for problems
        for key, value_list in self.dependencies.items():
            assert key in self.problems
            for value in value_list:
                assert value in self.problems
                key.add_child(value)
                value.add_parent(key)

        # specify leaves
        for problem in self.problems:
            if len(self.dependencies.get(problem, [])) == 0:
                problem.set_leaf()
                self.leaves.append(problem)
            problem.initialize()

    def train(self):
        for problem in self.problems:
            problem.train()

    def eval(self):
        for problem in self.problems:
            problem.eval()

    def is_implemented(self, fn_name):
        return callable(getattr(self, fn_name, None))
