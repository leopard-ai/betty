class Env:
    def __init__(self):
        self._problem_name_dict = {}

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass

    def set_problem_attr(self, problem):
        """
        Set class attribute for the given ``problem`` based on their names

        :param problem: Problem in multilevel optimization
        :type problem: Problem
        :return: ``problem`` name
        :rtype: str
        """
        name = problem.name
        if name not in self._problem_name_dict:
            assert not hasattr(
                self, name
            ), f"Problem already has an attribute named {name}!"
            self._problem_name_dict[name] = 0
            setattr(self, name, problem)
        elif self._problem_name_dict[name] == 0:
            # rename first problem
            first_problem = getattr(self, name)
            delattr(self, name)
            setattr(self, name + "_0", first_problem)

            self._problem_name_dict[name] += 1
            name = name + "_" + str(self._problem_name_dict[name])
            setattr(self, name, problem)
        else:
            self._problem_name_dict[name] += 1
            name = name + "_" + str(self._problem_name_dict[name])
            setattr(self, name, problem)

        return name
