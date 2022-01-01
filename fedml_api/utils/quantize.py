import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def uniform_quantize(k, g=32):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, param=None):
      ctx.save_for_backward(input, param)
      if param is None:
        if k == 32:
          out = input
        elif k == 1:
          out = torch.sign(input)
        else:
          n = float(2 ** k - 1)
          out = torch.round(input * n) / n
      else:
        assert input.size() == param.size()
        if k == 32:
          out = input
        else:
          n = float(2 ** k - 1)
          out = (torch.floor(input * n)+ torch.bernoulli(param))/n

      return out

    @staticmethod
    def backward(ctx, grad_output):
      input, param = ctx.saved_variables
      # if param is None:
      #   grad_param = None
      # else:
      #   grad_param = grad_output.clone()
      # return grad_output.clone(), grad_param

      grad = grad_output.clone()
      if param is None:
        if g == 32:
          out = grad
        elif g == 1:
          out = torch.sign(grad)
        else:
          n = float(2 ** g - 1)
          out = torch.round(grad * n) / n
      else:
        assert grad.size() == param.size()
        if g == 32:
          out = grad
        else:
          n = float(2 ** g - 1)
          out = (torch.floor(input * n)+ torch.bernoulli(param))/n

      if param is None:
        grad_param = None
      else:
        n = float(2 ** g - 1)
        grad_param = out.clone()/n
      return out, grad_param

  return qfn().apply


def uniform_quantize_new(k, g=32):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, param=None):
      ctx.save_for_backward(input, param)
      if param is None:
        if k == 32:
          out = input
        elif k == 1:
          out = torch.sign(input)
        else:
          n = float(2 ** k - 1)
          out = torch.round(input * n) / n
      else:
        assert input.size() == param.size()
        if k == 32:
          out = input
        else:
          n = float(2 ** k - 1)
          out = (torch.floor(input * n) + (torch.sign(param) + 1)/2)/n

      return out

    @staticmethod
    def backward(ctx, grad_output):
      input, param = ctx.saved_variables
      # if param is None:
      #   grad_param = None
      # else:
      #   grad_param = grad_output.clone()
      # return grad_output.clone(), grad_param

      grad = grad_output.clone()
      if param is None:
        if g == 32:
          out = grad
        elif g == 1:
          out = torch.sign(grad)
        else:
          n = float(2 ** g - 1)
          out = torch.round(grad * n) / n
      else:
        assert grad.size() == param.size()
        if g == 32:
          out = grad
        else:
          n = float(2 ** g - 1)
          out = (torch.floor(input * n) + (torch.sign(param) + 1)/2)/n

      if param is None:
        grad_param = None
      else:
        n = float(2 ** g - 1)
        grad_param = out.clone()/2/n
      return out, grad_param

  return qfn().apply


class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit, g_bit=32):
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit
    # self.uniform_q = uniform_quantize(k=w_bit, g=g_bit)
    self.uniform_q = uniform_quantize_new(k=w_bit, g=g_bit)
  def forward(self, x, param=None):
    if self.w_bit == 32:
      weight_q = x
    elif self.w_bit == 1:
      E = torch.mean(torch.abs(x)).detach()
      weight_q = self.uniform_q(x / E) * E
    else:
      weight = torch.tanh(x)
      max_w = torch.max(torch.abs(weight)).detach()
      weight = weight / 2 / max_w + 0.5
      weight_q = max_w * (2 * self.uniform_q(weight, param) - 1)
    return weight_q


class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit):
    super(activation_quantize_fn, self).__init__()
    assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = uniform_quantize(k=a_bit)

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = x
    else:
      activation_q = self.uniform_q(torch.clamp(x, 0, 1))
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q


def conv2d_Q_fn(w_bit, g_bit, random):
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
      self.w_bit = w_bit
      if random:
        self.param = nn.Parameter(torch.rand(self.weight.size()) - 0.5)
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit, g_bit=g_bit)

    def forward(self, input, order=None):
      if random:
        # weight_q = self.quantize_fn(self.weight, torch.sigmoid(self.param))
        weight_q = self.quantize_fn(self.weight, self.param)
      else:
        weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv2d_Q


def linear_Q_fn(w_bit, g_bit, random=False):
  class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
      super(Linear_Q, self).__init__(in_features, out_features, bias)
      self.w_bit = w_bit
      if random:
        self.param = nn.Parameter(torch.rand(self.weight.size()) - 0.5)
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit, g_bit=g_bit)

    def forward(self, input):
      if random:
        # weight_q = self.quantize_fn(self.weight, torch.sigmoid(self.param))
        weight_q = self.quantize_fn(self.weight, self.param)
      else:
        weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.linear(input, weight_q, self.bias)

  return Linear_Q