# https://github.com/MWPainter/cvpr2019/blob/master/stitched/soft_argmax.py
from __future__ import division

import torch


def _make_radial_window(width, height, cx, cy, fn, window_width=10.0):
    """
    Returns a grid, where grid[i,j] = fn((i**2 + j**2)**0.5)
    :param width: Width of the grid to return
    :param height: Height of the grid to return
    :param cx: x center
    :param cy: y center
    :param fn: The function to apply
    :return:
    """
    # The length of cx and cy is the number of channels we need
    channels = cx.size(0)

    # Explicitly tile cx and cy, ready for computing the distance matrix below, because pytorch doesn't broadcast very well
    # Make the shape [channels, height, width]
    cx = cx.repeat(height, width, 1).permute(2, 0, 1)
    cy = cy.repeat(height, width, 1).permute(2, 0, 1)

    # Compute a grid where dist[i,j] = (i-cx)**2 + (j-cy)**2, need to view and repeat to tile and make shape [channels, height, width]
    xs = torch.arange(width).view((1, width)).repeat(channels, height, 1).float().cuda()
    ys = (
        torch.arange(height).view((height, 1)).repeat(channels, 1, width).float().cuda()
    )
    delta_xs = xs - cx
    delta_ys = ys - cy
    dists = torch.sqrt((delta_ys**2) + (delta_xs**2))

    # apply the function to the grid and return it
    return fn(dists, window_width)


def _parzen_scalar(delta, width):
    """For reference"""
    del_ovr_wid = math.abs(delta) / width
    if delta <= width / 2.0:
        return 1 - 6 * (del_ovr_wid**2) * (1 - del_ovr_wid)
    elif delta <= width:
        return 2 * (1 - del_ovr_wid) ** 3


def _parzen_torch(dists, width):
    """
    A PyTorch version of the parzen window that works a grid of distances about some center point.
    See _parzen_scalar to see the
    :param dists: The grid of distances
    :param window: The width of the parzen window
    :return: A 2d grid, who's values are a (radial) parzen window
    """
    hwidth = width / 2.0
    del_ovr_width = dists / hwidth

    near_mode = (dists <= hwidth / 2.0).float()
    in_tail = ((dists > hwidth / 2.0) * (dists <= hwidth)).float()

    return near_mode * (
        1 - 6 * (del_ovr_width**2) * (1 - del_ovr_width)
    ) + in_tail * (2 * ((1 - del_ovr_width) ** 3))


def _uniform_window(dists, width):
    """
    A (radial) uniform window function
    :param dists: A grid of distances
    :param width: A width for the window
    :return: A 2d grid, who's values are 0 or 1 depending on if it's in the window or not
    """
    hwidth = width / 2.0
    return (dists <= hwidth).float()


def _identity_window(dists, width):
    """
    An "identity window". (I.e. a "window" which when multiplied by, will not change the input).
    """
    return torch.ones(dists.size())


def round_func_BPDA(input):
    # This is equivalent to replacing round function (non-differentiable) with
    # an identity function (differentiable) only when backward.
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out


class SoftArgmax1D(torch.nn.Module):
    """
    Implementation of a 1d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """

    def __init__(self, base_index=0, step_size=1):
        """
        The "arguments" are base_index, base_index+step_size, base_index+2*step_size, ... and so on for
        arguments at indices 0, 1, 2, ....
        Assumes that the input to this layer will be a batch of 1D tensors (so a 2D tensor).
        :param base_index: Remember a base index for 'indices' for the input
        :param step_size: Step size for 'indices' from the input
        """
        super(SoftArgmax1D, self).__init__()
        self.base_index = base_index
        self.step_size = step_size
        self.softmax = torch.nn.Softmax(dim=1).cuda()

    def forward(self, x):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:
        SoftArgMax(x) = \sum_i (i * softmax(x)_i)
        :param x: The input to the soft arg-max layer
        :return: Output of the soft arg-max layer
        """
        smax = self.softmax(x)
        end_index = self.base_index + x.size()[1] * self.step_size
        indices = torch.arange(
            start=self.base_index, end=end_index, step=self.step_size
        ).cuda()
        # print(smax, indices)
        return round_func_BPDA(torch.matmul(smax.float(), indices.float()))


class SoftArgmax2D(torch.nn.Module):
    """
    Implementation of a 1d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """

    def __init__(
        self,
        base_index=0,
        step_size=1,
        window_fn=None,
        window_width=10,
        softmax_temp=1.0,
    ):
        """
        The "arguments" are base_index, base_index+step_size, base_index+2*step_size, ... and so on for
        arguments at indices 0, 1, 2, ....
        Assumes that the input to this layer will be a batch of 3D tensors (so a 4D tensor).
        For input shape (B, C, W, H), we apply softmax across the W and H dimensions.
        We use a softmax, over dim 2, expecting a 3D input, which is created by reshaping the input to (B, C, W*H)
        (This is necessary because true 2D softmax doesn't natively exist in PyTorch...
        :param base_index: Remember a base index for 'indices' for the input
        :param step_size: Step size for 'indices' from the input
        :param window_function: Specify window function, that given some center point produces a window 'landscape'. If
            a window function is specified then before applying "soft argmax" we multiply the input by a window centered
            at the true argmax, to enforce the input to soft argmax to be unimodal. Window function should be specified
            as one of the following options: None, "Parzen", "Uniform"
        :param window_width: How wide do we want the window to be? (If some point is more than width/2 distance from the
            argmax then it will be zeroed out for the soft argmax calculation, unless, window_fn == None)
        """
        super(SoftArgmax2D, self).__init__()
        self.base_index = base_index
        self.step_size = step_size
        self.softmax = torch.nn.Softmax(dim=2)
        self.softmax_temp = softmax_temp
        self.window_type = window_fn
        self.window_width = window_width
        self.window_fn = _identity_window
        if window_fn == "Parzen":
            self.window_fn = _parzen_torch
        elif window_fn == "Uniform":
            self.window_fn = _uniform_window

    def _softmax_2d(self, x, temp):
        """
        For the lack of a true 2D softmax in pytorch, we reshape each image from (C, W, H) to (C, W*H) and then
        apply softmax, and then restore the original shape.
        :param x: A 4D tensor of shape (B, C, W, H) to apply softmax across the W and H dimensions
        :param temp: A scalar temperature to apply as part of the softmax function
        :return: Softmax(x, dims=(2,3))
        """
        B, C, W, H = x.size()
        x_flat = x.view((B, C, W * H)) / temp
        x_softmax = self.softmax(x_flat)
        return x_softmax.view((B, C, W, H))

    def forward(self, x):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:
        SoftArgMax2d(x) = (\sum_i \sum_j (i * softmax2d(x)_ij), \sum_i \sum_j (j * softmax2d(x)_ij))
        :param x: The input to the soft arg-max layer
        :return: Output of the 2D soft arg-max layer, x_coords and y_coords, in the shape (B, C, 2), which are the soft
            argmaxes per channel
        """
        # Compute windowed softmax
        # Compute windows using a batch_size of "batch_size * channels"
        batch_size, channels, height, width = x.size()
        argmax = torch.argmax(x.view(batch_size * channels, -1), dim=1)
        argmax_x, argmax_y = torch.remainder(argmax, width).float(), torch.floor(
            torch.div(argmax.float(), float(width))
        )
        windows = _make_radial_window(
            width, height, argmax_x, argmax_y, self.window_fn, self.window_width
        )
        windows = windows.view(batch_size, channels, height, width).cuda()
        smax = self._softmax_2d(x, self.softmax_temp) * windows
        smax = smax / torch.sum(smax.view(batch_size, channels, -1), dim=2).view(
            batch_size, channels, 1, 1
        )

        # compute x index (sum over y axis, produce with indices and then sum over x axis for the expectation)
        x_end_index = self.base_index + width * self.step_size
        x_indices = torch.arange(
            start=self.base_index, end=x_end_index, step=self.step_size
        ).cuda()
        x_coords = torch.sum(torch.sum(smax, 2) * x_indices, 2)

        # compute y index (sum over x axis, produce with indices and then sum over y axis for the expectation)
        y_end_index = self.base_index + height * self.step_size
        y_indices = torch.arange(
            start=self.base_index, end=y_end_index, step=self.step_size
        ).cuda()
        y_coords = torch.sum(torch.sum(smax, 3) * y_indices, 2)

        # For debugging (testing if it's actually like the argmax?)
        # argmax_x = argmax_x.view(batch_size, channels)
        # argmax_y = argmax_y.view(batch_size, channels)
        # print("X err in soft argmax: {err}".format(err=torch.mean(torch.abs(argmax_x - x_coords))))
        # print("Y err in soft argmax: {err}".format(err=torch.mean(torch.abs(argmax_y - y_coords))))

        # Put the x coords and y coords (shape (B,C)) into an output with shape (B,C,2)
        return torch.cat(
            [torch.unsqueeze(x_coords, 2), torch.unsqueeze(y_coords, 2)], dim=2
        )
