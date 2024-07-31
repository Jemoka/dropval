"""
models. 

code adapted or with reference to
https://github.com/eric-mitchell/mend
https://github.com/RobertCsordas/modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def update_counter(x, m, s, k):
    new_m = m + (x - m) / k
    new_s = s + (x - m) * (x - new_m)

    return new_m, new_s


class FactoredLinear(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, bias=True):
        super().__init__()

        self.u = nn.Parameter(torch.zeros(in_dim, hidden_dim))
        self.v = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim, out_dim)))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

    def forward(self, x):
        out = (self.u @ (self.v @ x.T)).T
        if self.bias != None:
            out += self.bias
        return out

class ShiftedFactoredLayerWithPassthrough(nn.Module):

    def __init__(self,
                 in_dim, out_dim, hidden_dim,
                 target_layer_count=0, bias=True):
        super().__init__()

        self.FL = FactoredLinear(in_dim, out_dim, hidden_dim, bias)

        if target_layer_count > 0:
            self.scale_embedding = nn.Embedding(target_layer_count, out_dim)
            self.scale_embedding.weight.data.fill_(1)
            self.shift_embedding = nn.Embedding(target_layer_count, out_dim)
            self.shift_embedding.weight.data.fill_(0)

    def forward(self, x, target_layer_id=None):
        out = self.FL(x)

        if target_layer_id != None:
            shift, scale = (self.shift_embedding(target_layer_id),
                            self.scale_embedding(target_layer_id))
            out = scale*out + shift

        # this is relu, but we preseve gradients from 
        # the backpropegation. any function σ st σ(0) = 0 
        # should work
        out = out.clamp(min=0) 

        return out+x

class MEND(nn.Module):
    """MEND Network from Eric + Friends
    
    Description
    -----------
    To enable some layer-wise specialization, MEND applies a layer-specific scale s and
    offset o to the editor network hidden state and output, similar to FiLM layers
    (Perez et al., 2018).

    Putting everything together, a MEND network computes
    g(z) where z = concat(u, δ)
    as
       h(z) = z + σ(s1 (U1 V1 z + b) + o1)
       g(z) = h(z) + σ(s2 (U2 V2 h) + o2) 

    Parameters
    ----------
    activation_size : int
        the size of the target activations 
    grad_size : int
        the size of the target gradients 
    mend_hidden_size : int
        the hidden size of the factored MLP for MEND
    target_layer_count : int
        how many layers of this size we want to edit using this
        network (i.e. how many o,s pairs to generate)
    """

    def __init__(self, activation_size, grad_size,
                 mend_hidden_size, target_layer_count):

        super().__init__()

        target_activation_size = activation_size + grad_size

        self.register_buffer("has_initialized", torch.tensor(False))
        self.register_buffer("u_mean", torch.full((activation_size,), float("nan")))
        self.register_buffer("v_mean", torch.full((grad_size,), float("nan")))
        self.register_buffer("u_std", torch.full((activation_size,), float("nan")))
        self.register_buffer("v_std", torch.full((grad_size,), float("nan")))
        self.register_buffer("u_s", torch.full((activation_size,), float("nan")))
        self.register_buffer("v_s", torch.full((grad_size,), float("nan")))
        self.register_buffer("count", torch.full((1,), float("nan")))

        # trainable learning rate
        self.lr = nn.Parameter(nn.init.zeros_(torch.empty(target_layer_count,)))

        self.h = ShiftedFactoredLayerWithPassthrough(target_activation_size,
                                                     target_activation_size,
                                                     mend_hidden_size,
                                                     target_layer_count,
                                                     True)
        self.g = ShiftedFactoredLayerWithPassthrough(target_activation_size,
                                                     target_activation_size,
                                                     mend_hidden_size,
                                                     target_layer_count,
                                                     False)

    def forward(self, u, v, target_layer_id):
        lid = torch.tensor(target_layer_id, device=u.device)
        # adapted from
        # https://github.com/eric-mitchell/mend/blob/e04fdb9cc784188906feffeb171025872933a5a8/algs/mend.py#L26

        u_, v_ = u.to(torch.float32), v.to(torch.float32)

        # because we are about to mean center it
        # if we are training, keep track of gradients
        if self.training:
            for batch_idx in range(u_.size(0)):
                if not self.has_initialized.all():
                    self.u_mean = u_[batch_idx].clone().detach()
                    self.v_mean = v_[batch_idx].clone().detach()
                    self.u_s.zero_()
                    self.v_s.zero_()
                    self.count[:] = 1
                    self.has_initialized = torch.tensor(True, device=self.has_initialized.device)
                else:
                    self.count += 1
                    self.u_mean, self.u_s = update_counter(u_[batch_idx], self.u_mean,
                                                            self.u_s, self.count)
                    self.v_mean, self.v_s = update_counter(v_[batch_idx], self.v_mean,
                                                           self.v_s, self.count)
            if self.count < 2:
                raise RuntimeError(f"Too few samples in this batch to normalise with: {self.count}")

            # -1 for student's t test
            self.u_std = (self.u_s / (self.count - 1)) ** 0.5
            self.v_std = (self.v_s / (self.count - 1)) ** 0.5

        # finally norm it
        u_ = (u_ - self.u_mean) / (self.u_std + 1e-7) 
        v_ = (v_ - self.v_mean) / (self.v_std + 1e-7) 

        x = torch.cat((u_, v_), dim=1)
        
        h = self.h(x, target_layer_id=lid)
        g = self.g(h, target_layer_id=lid)

        u_tilde, d_tilde = g[:,:u.size(1)], g[:,u.size(1):]

        return u_tilde, d_tilde, self.lr[target_layer_id]
        
class BinaryMaskedLinear(nn.Module):
    """A Binary Weight Mask to a Linear

    For learned binary masking from Robert and friends.
    """

    def __init__(self, tau=1, *args, **kwargs):
        super().__init__()

        self.linear = nn.Linear(*args, **kwargs)
        self.l = nn.Parameter(torch.ones_like(self.linear.weight), requires_grad=True)

        for param in self.linear.parameters():
            param.requires_grad = False

        self.tau = tau

    def get_mask(self, shape):
        u1 = torch.rand(list(shape) + list(self.l.shape), device=self.l.device)
        u2 = torch.rand(list(shape) + list(self.l.shape), device=self.l.device)

        # gumbel time
        si = F.sigmoid((self.l - ((u1.log()/u2.log()).log()) / self.tau))
        bi = ((si > 0.5).float() - si).detach() + si

        return bi

    def forward(self, x):
        if self.training:
            masked_weights = (self.get_mask((x.size(0),))*self.linear.weight.detach())
            res = torch.einsum("b...i,b...oi -> b...o", x, masked_weights)
        else:
            masked_weight = (F.sigmoid(self.l) > 0.5).float() * self.linear.weight.detach()
            res = torch.einsum("...i,oi -> ...o", x, masked_weight)

        if self.linear.bias != None:
            res += self.linear.bias.detach()

        return res

