for i in range(task_num):
    # 1. run the i-th task and compute loss for k=0
    logits = self.net(x_spt[i], vars=None, bn_training=True)
    loss = F.cross_entropy(logits, y_spt[i])
    grad = torch.autograd.grad(loss, self.net.parameters())
    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

    for k in range(1, self.update_step):
        # 1. run the i-th task and compute loss for k=1~K-1
        logits = self.net(x_spt[i], fast_weights, bn_training=True)
        loss = F.cross_entropy(logits, y_spt[i])
        # 2. compute grad on theta_pi
        grad = torch.autograd.grad(loss, fast_weights)
        # 3. theta_pi = theta_pi - train_lr * grad
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

        logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
        # loss_q will be overwritten and just keep the loss_q on last update step.
        loss_q = F.cross_entropy(logits_q, y_qry[i])
        losses_q[k + 1] += loss_q

    self.meta_optim.zero_grad()
    loss_q.backward()
    self.meta_optim.step()

