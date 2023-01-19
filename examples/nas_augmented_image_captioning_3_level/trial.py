# # import torch
# # import numpy as np
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.autograd import Variable
# # import matplotlib.pyplot as plt
# # from soft_argmax import SoftArgmax1D

# # def plot_grad_flow(named_parameters):
# #     ave_grads = []
# #     layers = []
# #     for n, p in named_parameters:
# #         if(p.requires_grad) and ("bias" not in n):
# #             layers.append(n)
# #             ave_grads.append(p.grad.abs().mean())
# #     plt.plot(ave_grads, alpha=0.3, color="b")
# #     plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
# #     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
# #     plt.xlim(xmin=0, xmax=len(ave_grads))
# #     plt.xlabel("Layers")
# #     plt.ylabel("average gradient")
# #     plt.title("Gradient flow")
# #     plt.grid(True)

# # class ArgMax(torch.autograd.Function):

# # 	@staticmethod
# # 	def forward(ctx, input):
# # 		idx = torch.argmax(input, 1)

# # 		output = torch.zeros_like(input)
# # 		output.scatter_(1, idx, 1)
# # 		ctx.save_for_backward(output)
# # 		return output


# # 	@staticmethod
# # 	def backward(ctx, grad_output):
# # 		return grad_output

# # class Net1(nn.Module):

# # 	def __init__(self):
# # 		super().__init__()
# # 		self.model = nn.Sequential(
# # 			nn.Linear(1000,100),
# # 			nn.ReLU(),
# # 			nn.Linear(100,100))

# # 	def forward(self, x):
# # 		return self.model(x)


# # class Net2(nn.Module):

# # 	def __init__(self):
# # 		super().__init__()
# # 		self.model = nn.Sequential(
# # 			nn.Linear(1000,100),
# # 			nn.ReLU(),
# # 			nn.Linear(100,100))

# # 	def forward(self, x):
# # 		return self.model(x)

# # train_x, train_y = np.random.randn(10,1000), np.random.choice(100,size=10)
# # valid_x, valid_y = np.random.randn(10,1000), np.random.choice(100,size=10)
# # pseudo_x = np.random.randn(10,1000)
# # train_x, train_y  = torch.tensor(train_x).float(), torch.tensor(train_y)
# # valid_x, valid_y  = torch.tensor(valid_x).float(), torch.tensor(valid_y)
# # pseudo_x = torch.tensor(pseudo_x).float()
# # n1 = Net1()
# # n2 = Net2()
# # optim1 = torch.optim.Adam(n1.parameters(),lr = 0.001)
# # optim2 = torch.optim.Adam(n2.parameters(),lr = 0.001)
# # criterion = nn.CrossEntropyLoss()
# # argmax = SoftArgmax1D()
# # for i in range(100):
# # 	optim1.zero_grad()
# # 	logits = n1(train_x)
# # 	loss = criterion(logits, train_y)
# # 	loss.mean()
# # 	loss.backward()
# # 	# plot_grad_flow(n1.named_parameters())
# # 	optim1.step()
# # 	print("initial logits from n1: ")
# # 	print(n1(train_x))

# # 	pseudo_logits = n1(pseudo_x)
# # 	pseudo_y = argmax(pseudo_logits).round()
# # 	print("pseudo_y: ",pseudo_y)

# # 	optim2.zero_grad()
# # 	logits = n2(pseudo_x)
# # 	loss = criterion(logits, pseudo_y.long())
# # 	loss.mean()
# # 	loss.backward()
# # 	optim2.step()


# # 	optim1.zero_grad()
# # 	logits = n2(valid_x)
# # 	loss = criterion(logits, valid_y)
# # 	loss.mean()
# # 	loss.backward()
# # 	plot_grad_flow(n1.named_parameters())
# # 	optim1.step()

# # plt.show()
# # print("after n2 step: ")
# # print(n1(train_x))


# import torch.nn as nn
# import torch
# from functools import reduce
# from operator import mul
# # from utils import get_logger

# """Implements the EmbeddingMul class
# Author: Noémien Kocher
# Date: Fall 2018
# Unit test: embedding_mul_test.py
# """

# # logger = None


# # A pytorch module can not have a logger as its attrbute, because
# # it then cannot be serialized.
# # def set_logger(alogger):
# #     global logger
# #     logger = alogger


# class EmbeddingMul(nn.Module):
#     """This class implements a custom embedding mudule which uses matrix
#     multiplication instead of a lookup. The method works in the functional
#     way.
#     Note: this class accepts the arguments from the original pytorch module
#     but only with values that have no effects, i.e set to False, None or -1.
#     """

#     def __init__(self, depth, device):
#         super(EmbeddingMul, self).__init__()
#         # i.e the dictionnary size
#         self.depth = depth
#         self.device = device
#         self.ones = torch.eye(depth, requires_grad=False, device=self.device)
#         self._requires_grad = True
#         # "oh" means One Hot
#         self.last_oh = None
#         self.last_weight = None

#     @property
#     def requires_grad(self):
#         return self._requires_grad

#     @requires_grad.setter
#     def requires_grad(self, value):
#         self._requires_grad = value
#         logger.info(
#             f"(embedding mul) requires_grad set to {self.requires_grad}. ")

#     def forward(self, input, weight, padding_idx=None, max_norm=None,
#                 norm_type=2., scale_grad_by_freq=False, sparse=False):
#         """Declares the same arguments as the original pytorch implementation
#         but only for backward compatibility. Their values must be set to have
#         no effects.
#         Args:
#             - input: of shape (bptt, bsize)
#             - weight: of shape (dict_size, emsize)
#         Returns:
#             - result: of shape (bptt, bsize, dict_size)
#         """
#         # ____________________________________________________________________
#         # Checks if unsupported argument are used
#         if padding_idx != -1:
#             raise NotImplementedError(
#                 f"padding_idx must be -1, not {padding_idx}")
#         if max_norm is not None:
#             raise NotImplementedError(f"max_norm must be None, not {max_norm}")
#         if scale_grad_by_freq:
#             raise NotImplementedError(f"scale_grad_by_freq must be False, "
#                                       f"not {scale_grad_by_freq}")
#         if sparse:
#             raise NotImplementedError(f"sparse must be False, not {sparse}")
#         # ____________________________________________________________________

#         if self.last_oh is not None:
#             del self.last_oh
#         self.last_oh = self.to_one_hot(input)

#         with torch.set_grad_enabled(self.requires_grad):
#             result = torch.stack(
#                 [torch.mm(batch.float(), weight)
#                  for batch in self.last_oh], dim=0)
#         self.last_weight = weight.clone()
#         return result

#     def to_one_hot(self, input):
#         # Returns a new tensor that doesn't share memory
#         result = torch.index_select(
#             self.ones, 0, input.view(-1).long()).view(
#             input.size()+(self.depth,))
#         result.requires_grad = self.requires_grad
#         return result

#     def __repr__(self):
#         return self.__class__.__name__ + "({})".format(self.depth)


# if __name__ == "__main__":
# 	input = torch.tensor([[1., 2., 0.], [3., 4., 5.]])
# 	dim = 10
# 	mod = EmbeddingMul(dim, 'cpu')
# 	emmatrix = torch.rand(10, 5, requires_grad=True)
# 	print(emmatrix)
# 	output = mod(input, emmatrix, -1)
# 	output = output.mean()
# 	print(torch.autograd.grad(output,emmatrix))
# 	print(output)

import torch

# class HotEmbedding(torch.nn.Module):
# 	def __init__(self, max_val, embedding_dim, eps=1e-2):
# 		super(HotEmbedding, self).__init__()
# 		self.A = torch.arange(max_val, requires_grad=False)
# 		self.B = torch.randn((max_val, embedding_dim), requires_grad=True)
# 		self.eps = eps

# 	def forward(self, x):
# 		print(x.shape, self.A.shape, self.B.shape)
# 		return 1/((x.unsqueeze(-1)**2 - self.A.unsqueeze(0).repeat(x.size(0),1,1)**2)+self.eps) @ self.B

# layer = HotEmbedding(10331, 512)
# x = torch.tensor([[1.,2.,3.,1.,2.,3.],[1.,2.,3.,1.,2.,3.]], requires_grad=True)
# y = layer(x)
# print(y.shape)

a = torch.Tensor([1.0, 2.0, 3.0, 4.0])
# a.requires_grad=True
# b = torch.eye(5)
# for i in range(4):
# 	b[i][i] = a[i]/a.detach()[i]

# print(b)

# c = torch.randn(5,512, requires_grad=True)

# y = b@c
# print(torch.autograd.grad(y.mean(), a, retain_graph=True))
