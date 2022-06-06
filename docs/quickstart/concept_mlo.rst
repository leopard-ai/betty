Multilevel Optimization
=======================

To introduce multilevel optimization, we first define an important concept known as a
*constrained problem*.

**Definition 1.** An optimization problem :math:`P` is said to be **constrained** by
:math:`\lambda` when its cost function :math:`\mathcal{C}` has :math:`\lambda` as an
argument in addition to the optimization parameter :math:`\theta` — i.e.
:math:`P:\arg\min_{\theta}\mathcal{C}(\theta, \lambda,\cdots)`.

Multilevel optimization (MLO) refers to a field of study that aims to solve a nested set
of optimization problems defined on a sequence of so-called *levels*, which satisfy two
main criteria: **(A1)** upper-level problems are constrained by the *optimal* parameters
of lower-level problems while **(A2)** lower-level problems are constrained by the
*nonoptimal* parameters of upper-level problems. Formally, an n-level MLO program can be
written as:

.. math::

    \begin{flalign*}
        P_n:\quad&& &\theta_n^* = \underset{\theta_n}{\mathrm{argmin}}\;\mathcal{C}_n(\theta_n, \mathcal{U}_n, \mathcal{L}_n; \mathcal{D}_n)&&\quad\quad\;\text{ $\rhd$ Level $n$}\\
        && &\hspace{8mm}\ddots &&\\
        P_k:\quad&& & \hspace{9mm}\text{s.t.} \hspace{2mm} \theta_k^* = \underset{\theta_k}{\mathrm{argmin}}\; \mathcal{C}_k(\theta_k, \mathcal{U}_k, \mathcal{L}_k; \mathcal{D}_k)&&\quad\quad\;\text{ $\rhd$ Level $k \in \{2, \ldots, n-1\}$}\\
        && &\hspace{23mm}\ddots &&\\
        P_1:\quad&& &\hspace{24mm}\text{s.t.}\hspace{2mm}\theta_1^* = \underset{\theta_1}{\mathrm{argmin}}\; \mathcal{C}_1(\theta_1, \mathcal{U}_1, \mathcal{L}_1; \mathcal{D}_k)&&\quad\quad\;\text{ $\rhd$ Level $1$}
    \end{flalign*}

where :math:`P_k` stands for the level k problem, :math:`\theta_k\,/\,\theta_k^*` for
corresponding nonoptimal / optimal parameters, and
:math:`\mathcal{U}_k\,/\,\mathcal{L}_k` for the sets of constraining parameters from
upper / lower level problems. Here, :math:`\mathcal{D}_k` is the training dataset, and
:math:`\mathcal{C}_k` indicates the cost function. Due to criteria **(A1)** & **(A2)**,
the constraining parameters from upper-level problems should be nonoptimal (i.e.
:math:`\mathcal{U}_k \subseteq \{\theta_{k+1}, \cdots, \theta_n\}`) while the
constraining parameters from lower-level problems should be optimal (i.e.
:math:`\mathcal{L}_k \subseteq \{\theta_{1}^*, \cdots, \theta_{k-1}^*\}`).

Although we denote only one optimization problem per level in the above formulation,
each level could in fact have multiple problems. Therefore, we henceforth discard the
concept of level, and rather assume that problems :math:`\{P_1, P_2, \cdots, P_n\}` of a
general MLO program are topologically sorted in a *reverse* order (i.e. :math:`P_n` /
:math:`P_1` denote uppermost / lowermost problems).

Application Examples
--------------------
Multilevel optimization has found a wide range of applications in machine learning, including, but
not limited to, meta learning [`Finn et al. (MAML) <https://arxiv.org/abs/1703.03400>`_],
hyperparameter optimization (HPO) [`Franceschi et al. <https://arxiv.org/pdf/1703.01785.pdf>`_,
`Lorraine et al. <https://arxiv.org/pdf/1703.01785.pdf>`_], neural architecture search (NAS)
[`Liu et al. (DARTS) <https://arxiv.org/abs/1806.09055>`_], and reinforcement learning (RL)
[`Konda et al. (Actor-Critic)
<https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf>`_].
In particular, each of these problems can be formulated as bilevel optimization, the
simplest case of multilevel optimization with a two-level hierarchy. To better
understand the concept of multilevel optimization, we illustrate how each of these
problems can be formulated under the above mathematical notation and framework.

+---------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------+
|               |                                          Level 2 (Upper)                                          |                                         Level 1 (Lower)                                        |
|               +-------------+------------------+-------------------+------------------------+---------------------+-------------+----------------------+----------------------+-------------------+----------------+
|               | :math:`C_2` | :math:`\theta_2` |    :math:`U_2`    |       :math:`L_2`      |     :math:`D_2`     | :math:`C_1` |   :math:`\theta_1`   |      :math:`U_1`     |    :math:`L_1`    |   :math:`D_1`  |
+===============+=============+==================+===================+========================+=====================+=============+======================+======================+===================+================+
| Meta Learning |      CE     |    init_weight   | :math:`\emptyset` | :math:`\{\theta_1^*\}` | Omniglot_meta-train |      CE     | task-specific weight | :math:`\{\theta_2\}` | :math:`\emptyset` | Omniglot_train |
+---------------+-------------+------------------+-------------------+------------------------+---------------------+-------------+----------------------+----------------------+-------------------+----------------+
|      HPO      |      CE     |  hyperparameter  | :math:`\emptyset` | :math:`\{\theta_1^*\}` |      PTB_valid      |      CE     |     LSTM weights     | :math:`\{\theta_2\}` | :math:`\emptyset` |    PTB_train   |
+---------------+-------------+------------------+-------------------+------------------------+---------------------+-------------+----------------------+----------------------+-------------------+----------------+
|      NAS      |      CE     |   architecture   | :math:`\emptyset` | :math:`\{\theta_1^*\}` |     CIFAR_valid     |      CE     |      CNN weights     | :math:`\{\theta_2\}` | :math:`\emptyset` |   CIFAR_train  |
+---------------+-------------+------------------+-------------------+------------------------+---------------------+-------------+----------------------+----------------------+-------------------+----------------+

While a majority of existing work is built upon bilevel optimization, there have been recent efforts
that go beyond this two-level hierarchy. For example,
[`Raghu et al. <https://arxiv.org/abs/2111.01754>`_] proposed trilevel optimization that combines
hyperparameter optimization with two-level pretraining and finetuning. More generally, conducting
joint optimization over machine learning pipelines consisting of multiple models and hyperparameter
sets can be approached as deeper instances of MLO
[`Such et al. <https://arxiv.org/abs/1912.07768>`_,
`Garg et al. <https://www.aaai.org/AAAI22Papers/AAAI-8716.GargB.pdf>`_].
