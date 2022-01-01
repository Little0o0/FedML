import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def uniform_quantize(k, k_g=32):
    class qfn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2 ** k - 1)
                out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            if k_g == 32:
                return grad_output
            else:
                x = grad_output.clone()
                rank = len(x.shape)
                assert rank is not None
                maxx = torch.max(torch.abs(x), 1, keepdim=True)
                for i in range(2, rank):
                    maxx = torch.max(torch.abs(maxx[0]), i, keepdim=True)
                x = x / maxx[0]
                n = float(2 ** k_g - 1)
                x = x * 0.5 + 0.5 + torch.FloatTensor(x.shape).uniform_(-0.5 / n, 0.5 / n).cuda()
                x = torch.clamp(x, 0.0, 1.0)
                x = torch.round(x * n) / n - 0.5
                return x * maxx[0] * 2

    return qfn().apply


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        assert w_bit <= 8 or w_bit == 32
        self.w_bit = w_bit
        self.uniform_q = uniform_quantize(k=w_bit)

    def forward(self, x):
        if self.w_bit == 32:
            weight_q = x
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(x)).detach()
            weight_q = self.uniform_q(x / E) * E
        else:
            weight = torch.tanh(x)
            weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
            weight_q = 2 * self.uniform_q(weight) - 1
        return weight_q

    def extra_repr(self):
        return "w_bit=%s" % (self.w_bit)


class activation_quantize_fn(nn.Module):
    def __init__(self, a_bit, g_bit):
        super(activation_quantize_fn, self).__init__()
        assert a_bit <= 8 or a_bit == 32
        self.a_bit = a_bit
        self.uniform_q = uniform_quantize(k = a_bit, k_g = g_bit)

    def forward(self, x):
        if self.a_bit == 32:
            activation_q = x
        else:
            activation_q = self.uniform_q(torch.clamp(x, 0, 1))
            # print(np.unique(activation_q.detach().numpy()))
        return activation_q

    def extra_repr(self):
        return "a_bit=%s" % (self.a_bit)

class DRF_QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, wbit=8, abit=8, gbit=8):
        super(DRF_QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, groups, bias)
        self.wbit = wbit
        self.abit = abit
        self.weight_quantize_fn = weight_quantize_fn(w_bit=wbit)
        self.activation_quantize_fn = activation_quantize_fn(a_bit=abit, g_bit=gbit)

    def forward(self, input, num_grad_bits=32, order=None):
        weight_q = self.weight_quantize_fn(self.weight)
        input_q = self.activation_quantize_fn(input)
        return F.conv2d(input_q, weight_q, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)



# class DRF_QLinear(nn.Linear):
#     def __init__(self, in_features, out_features, bias=True):
#         super(DRF_QLinear, self).__init__(in_features, out_features, bias)
#         self.bit = bit
#         self.weight_quantize_fn = weight_quantize_fn(w_bit=bit)
#         self.activation_quantize_fn = activation_quantize_fn(a_bit=bit)
#
#     def forward(self, input):
#         weight_q = self.weight_quantize_fn(self.weight)
#         input_q = self.activation_quantize_fn(input)
#         return F.linear(input_q, weight_q, self.bias)

# if __name__ == '__main__':
#   import numpy as np
#   import matplotlib.pyplot as plt

#   a = torch.rand(1, 3, 32, 32)

#   Conv2d = conv2d_Q_fn(w_bit=2)
#   conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
#   act = activation_quantize_fn(a_bit=3)

#   b = conv(a)
#   b.retain_grad()
#   c = act(b)
#   d = torch.mean(c)
#   d.retain_grad()

#   d.backward()
#   pass
