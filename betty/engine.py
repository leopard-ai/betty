import typing
from betty.utils import get_multiplier


class Engine:
    def __init__(self, config, problems, dependencies=None):
        # config
        self.config = config if config is not None else {}
        self.train_iters = 0
        self.valid_step = 0

        # problem
        self.problems = problems
        self._problem_name_dict = {}
        self.leaves = []

        # dependencies
        self.dependencies = dependencies

        # initialize
        self.initialize()

    def parse_config(self):
        self.train_iters = self.config.get('train_iters', 50000)
        self.valid_step = self.config.get('valid_step', 500)

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
        # Parse config
        self.parse_config()

        # Parse dependency
        self.parse_dependency()

        # check & set multiplier for each problem
        for problem in self.problems:
            multiplier = get_multiplier(problem)
            problem.multiplier = multiplier
            problem.initialize()

    def train(self):
        for problem in self.problems:
            problem.train()

    def eval(self):
        for problem in self.problems:
            problem.eval()

    def parse_dependency(self, set_attr=True):
        # Set dependencies for problems
        for key, value_list in self.dependencies.items():
            assert key in self.problems
            for value in value_list:
                assert value in self.problems
                key.add_child(value, set_attr)
                value.add_parent(key, set_attr)

        # Parse problems
        for problem in self.problems:
            if len(self.dependencies.get(problem, [])) == 0:
                problem.leaf = True
                self.leaves.append(problem)
            if set_attr:
                self.set_problem_attr(problem)

    def set_dependency(self, dependencies, set_attr=False):
        self.dependencies = dependencies
        self.leaves = []

        # clear existing dependencies
        for problem in self.problems:
            problem.leaf = False
            problem.clear_dependencies()

        self.parse_dependency(set_attr=set_attr)

    def set_problem_attr(self, problem):
        """[summary]
        Set class attributed for parent/children problems based on their names
        """
        name = problem.name
        if name not in self._problem_name_dict:
            assert not hasattr(self, name), f'Problem already has an attribute named {name}!'
            self._problem_name_dict[name] = 0
            setattr(self, name, problem)
        elif self._problem_name_dict[name] == 0:
            # rename first problem
            first_problem = getattr(self, name)
            delattr(self, name)
            setattr(self, name + '_0', first_problem)

            self._problem_name_dict[name] += 1
            name = name + '_' + str(self._problem_name_dict[name])
            setattr(self, name, problem)
        else:
            self._problem_name_dict[name] += 1
            name = name + '_' + str(self._problem_name_dict[name])
            setattr(self, name, problem)

        return name

    def is_implemented(self, fn_name):
        return callable(getattr(self, fn_name, None))
