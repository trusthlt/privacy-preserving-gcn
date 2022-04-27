import numpy as np
import torch
from torch.optim import SGD, Adam
from torch.nn.utils.clip_grad import clip_grad_norm_
import math
import sys


class DPSGD(SGD):

    def __init__(self, params, noise_scale, gradient_norm_bound, lot_size,
                 sample_size, lr=0.01):
        super(DPSGD, self).__init__(params, lr=lr)

        self.noise_scale = noise_scale
        self.gradient_norm_bound = gradient_norm_bound
        self.lot_size = lot_size
        self.sample_size = sample_size
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p.accumulated_grads = []

    def per_sample_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    per_sample_grad = p.grad.detach().clone()

                    ## Clipping gradient
                    clip_grad_norm_(per_sample_grad,
                                    max_norm=self.gradient_norm_bound)
                    p.accumulated_grads.append(per_sample_grad)

    def zero_accum_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.accumulated_grads = []

    def zero_sample_grad(self):
        super(DPSGD, self).zero_grad()

    def step(self, device, *args, **kwargs):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # DP:
                p.grad.data = torch.stack(p.accumulated_grads, dim=0).clone()

                ## Adding noise and aggregating each element of the lot:
                p.grad.data += torch.empty(p.grad.data.shape).normal_(mean=0.0, std=(self.noise_scale*self.gradient_norm_bound)).to(device)
                p.grad.data = torch.sum(p.grad.data, dim=0) * self.sample_size / self.lot_size
        super(DPSGD, self).step(*args, **kwargs)


class DPAdam(Adam):

    def __init__(self, params, noise_scale, gradient_norm_bound, lot_size,
                 sample_size, lr=0.01):
        super(DPAdam, self).__init__(params, lr=lr)

        self.noise_scale = noise_scale
        self.gradient_norm_bound = gradient_norm_bound
        self.lot_size = lot_size
        self.sample_size = sample_size
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p.accumulated_grads = []

    def per_sample_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    per_sample_grad = p.grad.detach().clone()

                    ## Clipping gradient
                    clip_grad_norm_(per_sample_grad, max_norm=self.gradient_norm_bound)
                    p.accumulated_grads.append(per_sample_grad)

    def zero_accum_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.accumulated_grads = []

    def zero_sample_grad(self):
        super(DPAdam, self).zero_grad()

    def step(self, device, *args, **kwargs):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # DP:
                p.grad.data = torch.stack(p.accumulated_grads, dim=0).clone()

                ## Adding noise and aggregating each element of the lot:
                p.grad.data += torch.empty(p.grad.data.shape).normal_(mean=0.0, std=(self.noise_scale*self.gradient_norm_bound)).to(device)
                p.grad.data = torch.sum(p.grad.data, dim=0) * self.sample_size / self.lot_size
        super(DPAdam, self).step(*args, **kwargs)
