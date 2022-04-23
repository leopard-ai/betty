import abc

import torch

import betty.hypergradient as hypergradient


class Problem:
    def __init__(self,
                 name,
                 config,
                 module=None,
                 optimizer=None,
                 scheduler=None,
                 train_data_loader=None,
                 device=None):
        self._name = name
        self._config = config
        self.device = device

        # computation graph depedency
        # ! dependency can be defined both in ``Module'' class and ``Engine'' class
        self._parents = []
        self._children = []
        self._paths = []
        self._problem_name_dict = {}

        # data loader
        self.train_data_loader = train_data_loader
        self.train_data_iterator = None
        self.cur_batch = None

        # module
        self.module = module

        # optimizer & lr scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

        # logging
        self.loggers = {}

        # misc
        self._leaf = False
        self._default_grad = False
        self._first_order = False
        self._retain_graph = config.retain_graph
        self._allow_unused = config.allow_unused
        self._inner_loop_start = True
        self._training = True
        self.ready = None
        self._count = 0
        self._multiplier = 1

    def initialize(self):
        """[summary]
        Initialize basic things
        """
        # initialize update ready to False
        if self._leaf:
            assert len(self._children) == 0
        if len(self._paths) == 0:
            self._default_grad = True
        self.ready = [False for _ in range(len(self._children))]
        path_str = [[node.name for node in path] for path in self._paths]
        children_str = [node.name for node in self._children]
        parents_str = [node.name for node in self._parents]
        print('[*] Problem INFO')
        print(f'Name: {self._name}')
        print(f'Parents: {parents_str}')
        print(f'Children: {children_str}')
        print(f'Paths: {path_str}')

        # initialize whether to track higher-order gradient for parameter update
        first_order = []
        for problem in self._parents:
            hgconfig = problem.config
            first_order.append(hgconfig.first_order)
        self._first_order = all(first_order)

        self._inner_loop_start = True

        # set up data loader
        train_data_loader = self.train_data_loader
        if self.is_implemented('configure_train_data_loader'):
            train_data_loader = self.configure_train_data_loader()
        assert train_data_loader is not None, "Train data loader must be specified!"
        self.train_data_iterator = iter(train_data_loader)

        # set up module for the current level
        if self.is_implemented('configure_module'):
            if self.configure_module() is not None:
                self.module = self.configure_module()
        assert self.module is not None, "Module must be specified!"

        # set up optimizer
        if self.is_implemented('configure_optimizer'):
            if self.configure_optimizer() is not None:
                self.optimizer = self.configure_optimizer()

        # set up lr scheduler
        if self.is_implemented('configure_scheduler'):
            if self.configure_scheduler is not None:
                self.scheduler = self.configure_scheduler()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """[summary]
        Users define how forward call is defined for the current problem.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def training_step(self, batch):
        """[summary]
        Users define how loss is calculated for the current problem.
        """
        raise NotImplementedError

    def step(self,
             batch=None,
             backpropagate=True,
             param_update=True):
        """[summary]
        Perform gradient calculation and update parameters accordingly
        """
        if self.check_ready():
            #if self._name == 'outer':
            #    print('name:', self._name)
            #    print('path:', self._paths)
            #    print(self.inner)
            #    print(self.inner.module)
            #    print(self._paths[0][1].module)
            # loop start
            if self._inner_loop_start:
                if self.is_implemented('on_inner_loop_start'):
                    self.on_inner_loop_start()
                self._inner_loop_start = False

                # copy current parameters, buffers, optimizer states
                if not param_update:
                    self.cache_states()

            # increase count
            if self._training and backpropagate:
                self._count += 1


            # load data
            self.cur_batch = self.get_batch() if batch is None else batch

            # calculate loss
            losses = self.get_losses()

            # calculate gradient (a.k.a backward)
            self.backward(losses=losses,
                          params=self.trainable_parameters(),
                          paths=self._paths,
                          config=self._config,
                          create_graph=not self._first_order,
                          retain_graph=self._retain_graph,
                          allow_unused=self._allow_unused)

            # calculate parameter update
            self.optimizer_step()

            if self.is_implemented('param_callback'):
                self.param_callback(self.trainable_parameters())

            # zero-out grad
            self.zero_grad()

            # call parent step function
            if self._training and backpropagate:
                for parent_idx, problem in enumerate(self._parents):
                    if self._count % problem.config.step == 0:
                        idx = problem.children.index(self)
                        problem.ready[idx] = True
                        problem.step()

                        self._inner_loop_start = True

                        if (not param_update) and (parent_idx == len(self._parents) - 1):
                            self.recover_states()

                            losses = self.get_losses()

                            self.backward(losses=losses,
                                          params=self.trainable_parameters(),
                                          paths=self._paths,
                                          config=self._config,
                                          create_graph=not self._first_order,
                                          retain_graph=self._retain_graph,
                                          allow_unused=self._allow_unused)

                            self.optimizer_step()

                            if self.is_implemented('param_callback'):
                                self.param_callback(self.trainable_parameters())

                            self.zero_grad()

            self.ready = [False for _ in range(len(self._children))]

    def get_batch(self):
        try:
            batch = next(self.train_data_iterator)
        except StopIteration:
            train_data_loader = self.train_data_loader
            if self.is_implemented('configure_train_data_loader'):
                train_data_loader = self.configure_train_data_loader()
            self.train_data_iterator = iter(train_data_loader)
            batch = next(self.train_data_iterator)

        return batch

    def get_losses(self):
        losses = self.training_step(self.cur_batch)
        if not (isinstance(losses, tuple) or isinstance(losses, list)):
            losses = (losses,)
        # TODO: Add custom loss aggregation
        # aggregate loss
        losses = tuple(loss / len(losses) for loss in losses)

        return losses

    def backward(self,
                 losses,
                 params,
                 paths,
                 config,
                 create_graph=False,
                 retain_graph=True,
                 allow_unused=True):
        if self._leaf:
            grads = self.get_grads(loss=sum(losses),
                                   params=params,
                                   path=None,
                                   config=config,
                                   create_graph=create_graph,
                                   retain_graph=retain_graph,
                                   allow_unused=allow_unused)
            self.set_grads(params, grads)
        else:
            for idx, path in enumerate(paths):
                loss = losses[0] if len(losses) == 1 else losses[idx]
                grads = self.get_grads(loss=loss,
                                       params=params,
                                       path=path,
                                       config=config,
                                       create_graph=create_graph,
                                       retain_graph=retain_graph,
                                       allow_unused=allow_unused)
                self.set_grads(params, grads)

    def get_grads(self,
                  loss,
                  params,
                  path=None,
                  config=None,
                  create_graph=False,
                  retain_graph=True,
                  allow_unused=True):
        if self._default_grad or self.config.type == 'torch':
            grads = torch.autograd.grad(loss, params,
                                        create_graph=create_graph,
                                        retain_graph=retain_graph,
                                        allow_unused=allow_unused)
        else:
            assert len(self._children) > 0
            grad_fn = hypergradient.get_grad_fn(self.config.type)
            grads = grad_fn(loss, params, path, config,
                            create_graph=create_graph,
                            retain_graph=retain_graph,
                            allow_unused=allow_unused)

        return grads

    def set_grads(self, params, grads):
        for param, grad in zip(params, grads):
            if hasattr(param, 'grad') and param.grad is not None:
                param.grad = param.grad + grad
            else:
                param.grad = grad

    @abc.abstractmethod
    def optimizer_step(self, *args, **kwargs):
        """[summary]
        Update weights as in native PyTorch's optim.step()
        """
        raise NotImplementedError

    def zero_grad(self):
        """[summary]
        Set gradients for trainable parameters for the current problem to 0.
        """
        for param in list(self.trainable_parameters()):
            if hasattr(param, 'grad'):
                del param.grad

    @abc.abstractmethod
    def cache_states(self):
        """[summary]
        Cache params, buffers, optimizer states for the later use.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recover_states(self):
        """[summary]
        Recover params, buffers, optimizer states from the cache.
        """
        raise NotImplementedError

    def is_implemented(self, fn_name):
        """[summary]
        Check if `fn_name` method is implemented in the class
        """
        return callable(getattr(self, fn_name, None))

    def check_ready(self):
        """[summary]
        Check if parameter updates in all children are ready
        """
        return all(self.ready)

    def log(self, dictionary):
        global_step = self._multiplier * self._count
        for key, value in dictionary.items():
            if key not in self.loggers:
                self.loggers[key] = [(global_step, value)]
            else:
                self.loggers[key].append((global_step, value))

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

    def add_child(self, problem):
        """[summary]
        Add a new problem to the children node list.
        """
        assert problem not in self._children
        self._children.append(problem)

    def add_parent(self, problem):
        """[summary]
        Add a new problem to the parent node list.
        """
        assert problem not in self._parents
        self._parents.append(problem)

    def add_paths(self, paths):
        """[summary]
        Add new backpropagation paths.
        """
        self._paths.extend(paths)

    @abc.abstractmethod
    def parameters(self):
        """[summary]
        Return all parameters for the current problem.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def trainable_parameters(self):
        """[summary]
        Return all parameters for the current problem.
        """
        raise NotImplementedError

    def clear_dependencies(self):
        self._children = []
        self._parents = []
        self._paths = []

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    @property
    def name(self):
        """[summary]
        Return the user-defined name of the module
        """
        return self._name

    @property
    def config(self):
        """[summary]
        Return the hypergradient configuration for the current problem.
        """
        return self._config

    @property
    def children(self):
        """[summary]
        Return children problems for the current problem.
        """
        return self._children

    @property
    def parents(self):
        """[summary]
        Return parent problemss for the current problem.
        """
        return self._parents

    @property
    def leaf(self):
        """[summary]
        Return whether the current problem is leaf or not.
        """
        return self._leaf

    @property
    def count(self):
        """[summary]
        Return count for the current problem.
        """
        return self._count

    @property
    def multiplier(self):
        """[summary]
        Return multiplier for the current problem.
        """
        return self._multiplier

    @multiplier.setter
    def multiplier(self, multiplier):
        assert isinstance(multiplier, int)
        self._multiplier = multiplier

    @leaf.setter
    def leaf(self, leaf):
        """[summary]
        Set the current problem as a leaf problem
        """
        self._leaf = leaf
