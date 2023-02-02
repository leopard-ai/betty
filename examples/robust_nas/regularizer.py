from __future__ import division
import torch
import torch.nn as nn
import torch.autograd as autograd

import hessianflow as hf
import hessianflow.optimizer.optm_utils as hf_optm_utils
import hessianflow.optimizer.progressbar as hf_optm_pgb


def zero_gradients(i):
    for t in iter_gradients(i):
        t.zero_()


class JacobianReg(nn.Module):
    """
    Loss criterion that computes the trace of the square of the Jacobian.

    Arguments:
        n (int, optional): determines the number of random projections.
            If n=-1, then it is set to the dimension of the output
            space and projection is non-random and orthonormal, yielding
            the exact result.  For any reasonable batch size, the default
            (n=1) should be sufficient.
    """

    def __init__(self, n=1):
        assert n == -1 or n > 0
        self.n = n
        super(JacobianReg, self).__init__()

    def forward(self, x, y):
        """
        computes (1/2) tr |dy/dx|^2
        """
        B, C = y.shape
        if self.n == -1:
            num_proj = C
        else:
            num_proj = self.n
        J2 = 0
        for ii in range(num_proj):
            if self.n == -1:
                # orthonormal vector, sequentially spanned
                v = torch.zeros(B, C)
                v[:, ii] = 1
            else:
                # random properly-normalized vector for each sample
                v = self._random_vector(C=C, B=B)
            if x.is_cuda:
                v = v.cuda()
            Jv = self._jacobian_vector_product(y, x, v, create_graph=True)
            J2 += C * torch.norm(Jv) ** 2 / (num_proj * B)
        R = (1 / 2) * J2
        return R

    def _random_vector(self, C, B):
        """
        creates a random vector of dimension C with a norm of C^(1/2)
        (as needed for the projection formula to work)
        """
        if C == 1:
            return torch.ones(B)
        v = torch.randn(B, C)
        arxilirary_zero = torch.zeros(B, C)
        vnorm = torch.norm(v, 2, 1, True)
        v = torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)
        return v

    def _jacobian_vector_product(self, y, x, v, create_graph=False):
        """
        Produce jacobian-vector product dy/dx dot v.

        Note that if you want to differentiate it,
        you need to make create_graph=True
        """
        flat_y = y.reshape(-1)
        flat_v = v.reshape(-1)
        (grad_x,) = torch.autograd.grad(
            flat_y, x, flat_v, retain_graph=True, create_graph=create_graph
        )
        return grad_x


class PJacobiNormReg(nn.Module):
    """
    Loss criterion that computes the trace of the square of the Jacobian.
    Arguments:
        n (int, optional): determines the number of random projections.
            If n=-1, then it is set to the dimension of the output
            space and projection is non-random and orthonormal, yielding
            the exact result.  For any reasonable batch size, the default
            (n=1) should be sufficient.
    """

    def __init__(self, n=2, p=1):
        assert n == -1 or n > 0
        self.n = n
        self.p = p
        super(PJacobiNormReg, self).__init__()

    def forward(self, x, y):
        """
        computes (1/2) tr |dy/dx|^2
        """

        B, C = y.shape
        if self.n == -1:
            num_iter = C
        else:
            num_iter = self.n
        J2 = 0
        index = torch.argsort(y, dim=1, descending=True)
        v = torch.zeros(B, C)
        for ii in range(num_iter):
            v += torch.eye(C)[index[:, ii]]
        Jv = self._jacobian_vector_product(y, x, v.cuda(), create_graph=True)
        J2 += torch.norm(Jv, self.p) / (num_iter * B)
        return J2

    def _jacobian_vector_product(self, y, x, v, create_graph=False):
        """
        Produce jacobian-vector product dy/dx dot v.
        Note that if you want to differentiate it,
        you need to make create_graph=True
        """
        flat_y = y.reshape(-1)
        flat_v = v.reshape(-1)
        (grad_x,) = torch.autograd.grad(
            flat_y, x, flat_v, retain_graph=True, create_graph=create_graph
        )
        return grad_x


class JacobiNormReg(nn.Module):
    """
    Loss criterion that computes the trace of the square of the Jacobian.
    Arguments:
        n (int, optional): determines the number of random projections.
            If n=-1, then it is set to the dimension of the output
            space and projection is non-random and orthonormal, yielding
            the exact result.  For any reasonable batch size, the default
            (n=1) should be sufficient.
    """

    # def __init__(self, n=2, p=1):
    def __init__(self, n=2, p="fro"):
        assert n == -1 or n > 0
        self.n = n
        self.p = p
        super(JacobiNormReg, self).__init__()

    def forward(self, x, y):
        """
        computes (1/2) tr |dy/dx|^2
        """

        B, C = y.shape
        if self.n == -1:
            num_iter = C
        else:
            num_iter = self.n
        J2 = 0
        index = torch.argsort(y, dim=1, descending=True)

        for ii in range(num_iter):
            v = torch.eye(C)[index[:, ii]]
            Jv = self._jacobian_vector_product(y, x, v.cuda(), create_graph=True)
            J2 += torch.norm(Jv, self.p) / (num_iter * B)
        return J2

    def _jacobian_vector_product(self, y, x, v, create_graph=False):
        """
        Produce jacobian-vector product dy/dx dot v.
        Note that if you want to differentiate it,
        you need to make create_graph=True
        """
        flat_y = y.reshape(-1)
        flat_v = v.reshape(-1)
        (grad_x,) = torch.autograd.grad(
            flat_y, x, flat_v, retain_graph=True, create_graph=create_graph
        )
        return grad_x


class JacobiLossNormReg(nn.Module):
    """
    Loss criterion that computes the trace of the square of the Jacobian.
    Arguments:
        n (int, optional): determines the number of random projections.
            If n=-1, then it is set to the dimension of the output
            space and projection is non-random and orthonormal, yielding
            the exact result.  For any reasonable batch size, the default
            (n=1) should be sufficient.
    """

    def __init__(self, n=2, p=1):
        assert n == -1 or n > 0
        self.n = n
        self.p = p
        super(JacobiLossNormReg, self).__init__()

    def forward(self, x, loss):
        """
        computes (1/2) tr |dy/dx|^2
        """

        B = x.shape[0]

        Jv = self._jacobian_vector_product(loss, x, create_graph=True)
        J2 = torch.norm(Jv, self.p) / B
        return J2

    def _jacobian_vector_product(self, loss, x, create_graph=False):
        """
        Produce jacobian-vector product dy/dx dot v.
        Note that if you want to differentiate it,
        you need to make create_graph=True
        """
        flat_loss = loss.reshape(-1)
        (grad_x,) = torch.autograd.grad(
            flat_loss, x, retain_graph=True, create_graph=create_graph
        )
        return grad_x


class JacobiAngularReg(nn.Module):
    """
    Loss criterion that computes the trace of the square of the Jacobian.
    Arguments:
        n (int, optional): determines the number of random projections.
            If n=-1, then it is set to the dimension of the output
            space and projection is non-random and orthonormal, yielding
            the exact result.  For any reasonable batch size, the default
            (n=1) should be sufficient.
    """

    def __init__(self, n=2, p=1):
        assert n == -1 or n > 0
        self.n = n
        self.p = p
        super(JacobiAngularReg, self).__init__()

    def forward(self, x, y):
        """
        computes (1/2) tr |dy/dx|^2
        """

        B, C = y.shape
        if self.n == -1:
            num_iter = C
        else:
            num_iter = self.n
        J2 = 0
        index = torch.argsort(y, dim=1, descending=True)

        for ii in range(num_iter):
            v = torch.eye(C)[index[:, ii]]
            Jv = self._jacobian_vector_product(y, x, v.cuda(), create_graph=True)
            J2 += torch.norm(Jv, self.p) / B
        return J2

    def _jacobian_vector_product(self, y, x, v, create_graph=False):
        """
        Produce jacobian-vector product dy/dx dot v.
        Note that if you want to differentiate it,
        you need to make create_graph=True
        """
        flat_y = y.reshape(-1)
        flat_v = v.reshape(-1)
        (grad_x,) = torch.autograd.grad(
            flat_y, x, flat_v, retain_graph=True, create_graph=create_graph
        )
        return grad_x


class loss_curv:
    def __init__(self, net, criterion, lambda_, device="cuda"):
        self.net = net
        self.criterion = criterion
        self.lambda_ = lambda_
        self.device = device

    def _find_z(self, inputs, alphas, targets, h):
        inputs.requires_grad_()
        outputs = self.net.eval()(inputs, alphas)
        # print(targets.size()[0])
        loss_z = self.criterion(outputs, targets)  # self.net.eval()(inputs)
        # loss_z = self.net.module.loss(inputs, alphas, targets)

        loss_z.backward(
            torch.ones(targets.size(), dtype=torch.float)[0].to(self.device)
        )  # torch.ones(targets.size(), dtype=torch.float).to(self.device)
        grad = inputs.grad.data + 0.0
        norm_grad = grad.norm().item()
        z = torch.sign(grad).detach() + 0.0
        z = (
            1.0
            * (h)
            * (z + 1e-7)
            / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None] + 1e-7)
        )
        inputs.grad.detach()
        inputs.grad.zero_()
        # zero_gradients(inputs)
        self.net.zero_grad()

        return z, norm_grad

    def regularizer(self, inputs, alphas, targets, h=3.0, lambda_=4):
        z, norm_grad = self._find_z(inputs, alphas, targets, h)

        inputs.requires_grad_()
        outputs_pos = self.net.eval()(inputs + z, alphas)
        outputs_orig = self.net.eval()(inputs, alphas)

        loss_pos = self.criterion(outputs_pos, targets)
        loss_orig = self.criterion(outputs_orig, targets)
        grad_diff = torch.autograd.grad(
            (loss_pos - loss_orig),
            inputs,
            grad_outputs=torch.ones(targets.size())[0].to(self.device),
            create_graph=True,
        )[0]
        reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
        self.net.zero_grad()

        return torch.sum(self.lambda_ * reg) / float(inputs.size(0)), norm_grad


class loss_eigen:
    def __init__(
        self,
        net,
        test_loader,
        input,
        target,
        criterion,
        full_eigen,
        maxIter=10,
        tol=1e-2,
    ):
        self.net = net
        self.test_loader = test_loader
        self.criterion = criterion
        self.full_eigen = full_eigen
        self.max_iter = maxIter
        self.tol = tol
        self.input = input
        self.target = target
        self.cuda = True

    def regularizer(self):
        if self.full_eigen:
            eigenvalue, eigenvector = hf.get_eigen_full_dataset(
                self.net, self.test_loader, self.criterion, self.max_iter, self.tol
            )
        else:
            eigenvalue, eigenvector = hf.get_eigen(
                self.net,
                self.input,
                self.target,
                self.criterion,
                self.cuda,
                self.max_iter,
                self.tol,
            )

        return eigenvalue, eigenvector
