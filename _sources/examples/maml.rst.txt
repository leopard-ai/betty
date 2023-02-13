Implicit Model Agnostic Meta-Learning
=====================================

Here we re-implement the model-agnostic meta-learning algorithm from
`Meta-Learning with Implicit Gradients <https://arxiv.org/pdf/1909.04630.pdf>`_,
where we learn the initialization weight for convolutional neural networks (CNNs) that allows
quick adaptations to various tasks given only few samples (i.e. few-shot learning).
In this post, we assume that the potential readers are already familiar with the algorithm,
and therefore mainly focus on how to implement MAML with Betty. Also, we note that,
while we focus on the implicit MAML instead of MAML in this blog post, the
modification for MAML is highly straightforward just by replacing ``ImplicitProblem``
with ``IterativeProblem``. Finally, the full version of our code can be found
`here <https://github.com/leopard-ai/betty/tree/main/examples/implicit_maml>`_.

Basics
------
MAML can be interpreted as a bilevel optimization problem, where the upper level learns
the initiazliation weight that allows quick adaptation to new tasks and
the lower level learns to adapt to the task given few training examples. Therefore,
users need to define two ``Problem`` classes for both levels. Following Betty's design
principle, each level can be defined by providing:

1. Module
2. Optimizer
3. Data loader (or data loading function)
4. Loss function
5. ``Problem`` configuration
6. Name
7. (Optional) learning rate scheduler, ...

While data loading is decoupled in most other bi-level optimization problems, in MAML,
it's coupled as the adaptation result from the lower level should be the tested on the
same task in the upper level. We will dive into this in this post.


Environment
~~~~~~~~~~~

As stated above, data loading is coupled for upper and lower levels in MAML. This requires
the user to implement a unified data loading mechanism. Unfortunately, this is hard to achieve
(or the code will get ugly) with the Betty's ``Problem`` class design where users provide
data loader separately for each ``Problem`` class through the class constructor.
To enable the clean implementation for such cases (where data loading is entangled for
multiple ``Problem``s), Betty provides the ``Env`` class where users can specify a unified
data loading mechanism.

More specifically, we use `learn2learn <https://github.com/learnables/learn2learn>`_
to load MAML dataset loading, and define the data loading mechanism in the ``step`` method.
The code is shown below.

.. code:: python

    import learn2learn as l2l
    from betty.envs import Env
    
    tasksets = l2l.vision.benchmarks.get_tasksets(
        args.task,
        train_ways=args.ways,
        train_samples=2 * args.shots,
        test_ways=args.ways,
        test_samples=2 * args.shots,
        num_tasks=args.task_num,
        root="./data",
    )

    def split_data(data, labels, shots, ways):
        out = {"train": None, "test": None}
        adapt_indices = np.zeros(data.size(0), dtype=bool)
        adapt_indices[np.arange(shots * ways) * 2] = True
        eval_indices = torch.from_numpy(~adapt_indices)
        adapt_indices = torch.from_numpy(adapt_indices)
        out["train"] = (data[adapt_indices], labels[adapt_indices])
        out["test"] = (data[eval_indices], labels[eval_indices])

        return out

    class MAMLEnv(Env):
        def __init__(self):
            super().__init__()

            self.tasks = tasksets
            self.batch = {"train": None, "test": None}

        def step(self):
            data, labels = self.tasks.train.sample()
            data, labels = data.to(self.device), labels.to(self.device)
            out = split_data(data, labels, args.shots, args.ways)
            self.batch["train"] = out["train"]
            self.batch["test"] = out["test"]

    env = MAMLEnv()


Upper-Level
~~~~~~~~~~~
Betty allows users to define a custom data loading mechanism through ``get_batch`` method.
Access to ``Env`` will be granted by ``Engine``, and users can simply use ``self.env``.
Also, since the upper problem is updated only after the ``meta_batch_size`` amount of
gradient accumulation, users need to specify this thorugh ``gradient_accumulation``
attribute in ``Config``. The rest part (e.g., defining loss function, module, optimizer,
etc.) is relatively straightforward, and we direct readers who are unfamiliar with this
to our `Tutorial <https://leopard-ai.github.io/betty/tutorial/basic/basic_start.html>`_.

.. code:: python

    class Upper(ImplicitProblem):
        def training_step(self, batch):
            inputs, labels = batch
            out = self.lower(inputs)
            loss = F.cross_entropy(out, labels)
            acc = 100.0 * (out.argmax(dim=1) == labels).float().mean().item()

            return {"loss": loss, "acc": acc}

        def get_batch(self):
            inputs, labels = self.env.batch["test"]

            return inputs, labels

    parent_module = ConvNet(args.ways, args.hidden_size)
    parent_optimizer = optim.AdamW(parent_module.parameters(), lr=3e-4)
    parent_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        parent_optimizer,
        T_max=int(args.meta_batch_size * 7500),
    )
    parent_config = Config(
        log_step=int(args.inner_steps * args.meta_batch_size * 10),
        retain_graph=True,
        gradient_accumulation=args.meta_batch_size,
    )
    parent = Upper(
        name="upper",
        module=parent_module,
        optimizer=parent_optimizer,
        scheduler=parent_scheduler,
        config=parent_config,
    )


Lower-Level
~~~~~~~~~~~

As in the upper level, we also define the custom data loading mechanism through
the ``get_batch`` method. In addition, we need to initialize the weight of inner CNNs
to that of outer CNNs in the beginning of the inner loop. We offer such functionality
via ``on_inner_loop_start`` method.

.. code:: python

    def reg_loss(parameters, reference_parameters, reg_lambda=0.25):
        loss = 0
        for p1, p2 in zip(parameters, reference_parameters):
            loss += torch.sum(torch.pow((p1 - p2), 2))

        return reg_lambda * loss

    class Lower(ImplicitProblem):
        def training_step(self, batch):
            inputs, labels = batch
            out = self.module(inputs)
            loss = F.cross_entropy(out, labels)
            reg = reg_loss(self.parameters(), self.upper.parameters(), args.reg)

            return loss + reg

        def get_batch(self):
            inputs, labels = self.env.batch["train"]

            return inputs, labels

        def on_inner_loop_start(self):
            self.module.load_state_dict(self.upper.module.state_dict())
    
    child_module = model_cls(args.ways, args.hidden_size)
    child_optimizer = optim.SGD(child_module.parameters(), lr=1e-1)
    child_config = Config(type="darts", unroll_steps=args.inner_steps)
    lower = Lower(
        name="lower",
        module=child_module,
        optimizer=child_optimizer,
        config=child_config
    )


Engine
~~~~~~

As illustruated in our
`Tutorial <https://leopard-ai.github.io/betty/tutorial/basic/basic_start.html>`_,
the overall execution of MLO is handled by ``Engine``. Since we handle data loading in
``Env``'s ``step`` method, we have to (1) provide ``MAMLEnv`` to the ``Engine`` and (2)
coordinate the execution order of ``env.step`` with other ``problem.step``. For the
first part, users can simply provide users' custom ``Env`` via the ``Engine`` class
constructor. Coordinating execution order of ``Env`` and ``Problem`` can be achieved in
the ``train_step`` method in ``Engine``. The below code shows how to do these.

.. code:: python

    class MAMLEngine(Engine):
        def train_step(self):
            if self.global_step % args.inner_steps == 1 or args.inner_steps == 1:
                self.env.step()
            for leaf in self.leaves:
                leaf.step(global_step=self.global_step)

        def validation(self):
            self.upper.module.train()
            if not hasattr(self, "best_acc"):
                self.best_acc = -1
            test_net = ConvNet(args.ways, args.hidden_size).to(self.device)
            test_optim = optim.SGD(test_net.parameters(), lr=0.1)
            accs = []
            for i in range(500):
                inputs, labels = tasksets.test.sample()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                out = split_data(inputs, labels, args.shots, args.ways)
                train_inputs, train_labels = out["train"]
                test_inputs, test_labels = out["test"]
                test_net.load_state_dict(self.upper.module.state_dict())
                for _ in range(args.inner_steps):
                    out = test_net(train_inputs)
                    loss = F.cross_entropy(out, train_labels)
                    test_optim.zero_grad()
                    loss.backward()
                    test_optim.step()

                out = test_net(test_inputs)
                accs.append((out.argmax(dim=1) == test_labels).detach())

            acc = 100.0 * torch.cat(accs).float().mean().item()
            if acc > self.best_acc:
                self.best_acc = acc

            return {"acc": acc, "best_acc": self.best_acc}

    u2l = {outer: [inner]}
    l2u = {inner: [outer]}
    dependencies = {"u2l": u2l, "l2u": l2u}
    engine = MAMLEngine(
        config=engine_config, problems=problems, dependencies=dependencies, env=env
    )
    engine.run()

Finally, users can also define the validation mechanism via the ``validation`` method,
and execute MAML training with ``engine.run()``.


Overall, throughout this tutorial, we tried to describe how to handle a unified/entangled
data loading mechanism for multiple ``Problem`` classes via ``Env``. Such use of ``Env``
can also be useful for implementing reinforcement learning related algorithms.