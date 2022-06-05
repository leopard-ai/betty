Multilevel Optimization
=======================

To introduce MLO, we first define an important concept known as a *constrained problem*.

**Definition 1.** An optimization problem :math:`P` is said to be **constrained** by
:math:`\lambda` when its cost function :math:`\mathcal{C}` has :math:`\lambda` as an argument in
addition to the optimization parameter :math:`\theta` 
(i.e. :math:`P:\arg\min_{\theta}\mathcal{C}(\theta, \lambda,\cdots)`). 

Multilevel optimization refers to a field of study that aims to solve a nested set of optimization
problems defined on a sequence of so-called *levels*, which satisfy two main criteria: **(A1)**
upper-level problems are constrained by the *optimal* parameters of lower-level problems while
**(A2)** lower-level problems are constrained by the *nonoptimal* parameters of upper-level
problems. Formally, an n-level MLO program can be written as:

.. math::

    \begin{flalign*}
        P_n:&& &\theta_n^* = \underset{\theta_n}{\mathrm{argmin}}\;\mathcal{C}_n(\theta_n, \mathcal{U}_n, \mathcal{L}_n; \mathcal{D}_n)&& \text{ $\rhd$ Level $n$}\\
        && &\hspace{8mm}\ddots &&\\
        P_k:&& & \hspace{9mm}\text{s.t.} \hspace{2mm} \theta_k^* = \underset{\theta_k}{\mathrm{argmin}}\; \mathcal{C}_k(\theta_k, \mathcal{U}_k, \mathcal{L}_k; \mathcal{D}_k)&& \text{ $\rhd$ Level $k \in \{2, \ldots, n-1\}$}\\
        && &\hspace{23mm}\ddots &&\\
        P_1:&& &\hspace{24mm}\text{s.t.}\hspace{2mm}\theta_1^* = \underset{\theta_1}{\mathrm{argmin}}\; \mathcal{C}_1(\theta_1, \mathcal{U}_1, \mathcal{L}_1; \mathcal{D}_k)&& \text{ $\rhd$ Level $1$}
    \end{flalign*}

where, :math:`P_k` stands for the level k problem, :math:`\theta_k\,/\,\theta_k^*` for
corresponding nonoptimal / optimal parameters, and :math:`\mathcal{U}_k\,/\,\mathcal{L}_k` for the
sets of constraining parameters from upper / lower level problems. Here, :math:`\mathcal{D}_k` is
the training dataset, and :math:`\mathcal{C}_k` indicates the cost function. Due to criteria
**A1** & **A2**, constraining parameters from upper-level problems should be nonoptimal (i.e.
:math:`\mathcal{U}_k \subseteq \{\theta_{k+1}, \cdots, \theta_n\}`) while constraining parameters
from lower-level problems should be optimal (i.e.
:math:`\mathcal{L}_k \subseteq \{\theta_{1}^*, \cdots, \theta_{k-1}^*\}`). Although we denote only
one optimization problem per level in the above formulation, each level could in fact have multiple
problems. Therefore, we henceforth discard the concept of level, and rather assume that problems
:math:`\{P_1, P_2, \cdots, P_n\}` of a general MLO program are topologically sorted in a
"reverse" order (i.e. :math:`P_n` / :math:`P_1` denote uppermost / lowermost problems).

For example, in hyperparameter optimization formulated as bilevel optimization, hyperparameters and
network parameters correspond to upper and lower level parameters (:math:`\theta_2` and
:math:`\theta_1`). Train / validation losses correspond to :math:`\mathcal{C}_1` /
:math:`\mathcal{C}_2`, and validation loss is dependent on optimal network parameters
:math:`\theta_1^*` obtained given :math:`\theta_2`. Thus, constraining sets for each level are
:math:`\mathcal{U}_1=\{\theta_2\}` and :math:`\mathcal{L}_2=\{\theta_1^*\}`.