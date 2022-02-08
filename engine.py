import module

class Engine:
    def __init__(self, config, problems, dependencies=None):
        self.config = self.parse_config(config)

        self.problems = problems
        self.leaves = []

        self.dependencies = dependencies

        self.initialize()

    def train(self):
        for _ in range(1000):
            for leaf in self.leaves:
                leaf.step()

    def evaluation(self):
        for problem in self.problems:
            problem.valiation_step()

    def parse_config(self, config):
        return config

    def initialize(self):
        """[summary]
        Initialize dependencies (computational graph) between problems.
        """
        # Set dependencies for problems
        for key, value_list in self.dependencies.items():
            assert key in self.problems
            for value in value_list:
                assert value in self.problems
                key.add_children(value)
                value.add_parent(key)

        # specify leaves
        for problem in self.problems:
            if problem.hgconfig().leaf:
                self.leaves.append(problem)
            problem.initialize()
