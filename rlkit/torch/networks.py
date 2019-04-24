"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm


def identity(x):
    return x


class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class MlpEncoder(FlattenMlp):
    '''
    encode context via MLP
    '''

    def reset(self, num_tasks=1):
        pass

def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, torch.sqrt(sigma_squared)

class Info_bottleneck:

    def information_bottleneck(self, z):
        # assume input and output to be task x batch x feat
        mu = z[..., :self.z_dim]
        sigma_squared = F.softplus(z[..., self.z_dim:])
        z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
        # if not self.det_z:
        z_dists = [torch.distributions.Normal(m, s) for m, s in z_params]
        self.z_dists = z_dists
        z = [d.rsample() for d in z_dists]
        # else:
        #     z = [p[0] for p in z_params]
        z = torch.stack(z)
        return z

class OracleEncoder(MlpEncoder, Info_bottleneck):
    def forward(self, *inputs, **kwargs):
        """
        make sure the inputs are of shape b,dim
        :param inputs:
        :param kwargs:
        :return:
        """
        in_ = inputs[0]
        new_z = super(OracleEncoder, self).forward(*inputs, **kwargs)
        new_z = new_z.view(in_.size(0), -1, self.task_enc.output_size)

        return self.information_bottleneck(new_z)

class SeqEncoder(MlpEncoder, Info_bottleneck):
    def __init__(self, z_dim, *inputs, **kwargs):
        self.save_init_params(locals())
        super().__init__(*inputs, **kwargs)
        self.z_collection = list()
        self.z_dim = z_dim

    def forward(self, *inputs, **kwargs):
        """
        make sure the inputs are of shape 1,dim
        :param inputs:
        :param kwargs:
        :return:
        """
        single_z = super().forward(*inputs, **kwargs) # mb,1,dim
        # TODO: # otherwise append the result: new_z
        self.z_collection.append(single_z)
        inp = torch.stack(self.z_collection, dim=1) # mb,n,dim
        # TODO all information bottleneck should make sure its input of shape mb,b,dim
        new_z = self.information_bottleneck(inp) # mb,dim
        return new_z # mb,dim

    def clear(self):
        self.z_collection.clear()

class OracleEncoder2(OracleEncoder):
    def __init__(self, encoder1, encoder2, *inputs, **kwargs):
        self.save_init_params(locals())
        super(OracleEncoder2, self).__init__(*inputs, **kwargs)
        self.encoder1 = encoder1
        self.encoder2 = encoder2 # make sure its inputs shape is 1,dim

    def forward(self, sar, phy):
        zs = self.encoder1.forward(sar) # b,dim
        phi = self.encoder2.forward(phy) # 1,dim
        phi_exp = phi.repeat(zs.size(0),1) # b,dim
        # inp = torch.cat((zs,phi_exp), dim=-1)
        return super().forward((zs,phi_exp))

class RecurrentEncoder(FlattenMlp):
    '''
    encode context via recurrent network
    '''

    def __init__(self,
                 *args,
                 **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out, (hn, cn) = self.lstm(out, (self.hidden, torch.zeros(self.hidden.size()).to(ptu.device)))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out[:, -1, :]

        # output layer
        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)




