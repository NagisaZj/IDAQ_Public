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
# import rlkit.torch.transformer as transformer

def identity(x):
    return x

class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            batch_attention=False,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            use_dropout=False,
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
        self.use_dropout = use_dropout
        self.fcs = []
        self.layer_norms = []
        self.dropouts = []
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
            
            if self.use_dropout:
                dropout_n = nn.Dropout(0.1)
                self.__setattr__("drop_out{}".format(i), dropout_n)
                self.dropouts.append(dropout_n)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)
        
        self.batch_attention = batch_attention
        self.transition_attention = transformer.BatchTransitionAttention(
            hidden=100,
            input_size=input_size,
            output_size=input_size,
            n_layers=3,
            attn_heads=1,
            dropout=0.1
        ) if self.batch_attention else None

    def forward(self, input, return_preactivations=False):
        if self.batch_attention:
            input = self.transition_attention(input)
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
            if self.use_dropout and i < len(self.fcs) - 1:
                h = self.dropouts[i](h)
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

    def forward(self, meta_size=16, batch_size=256, *inputs, **kwargs):
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


class MlpEncoder(Mlp):
    '''
    encode context via MLP
    '''

    def reset(self, num_tasks=1):
        pass


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




import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import scipy.spatial
import os
import tarfile

class DynamicsNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_neurons = 512, activation = nn.ReLU):
        super(DynamicsNet, self).__init__()

        # Validate inputs
        assert input_dim > 0
        assert output_dim > 0
        assert n_neurons > 0

        # Store configuration parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neurons = n_neurons

        # Create layers for the net
        self.input_layer = nn.Linear(input_dim, n_neurons)
        self.h0 = nn.Linear(n_neurons, n_neurons)
        self.h0_act = activation()
        self.h1 = nn.Linear(n_neurons, n_neurons)
        self.h1_act = activation()
        self.output_layer = nn.Linear(n_neurons, output_dim)


    def forward(self, x):
        x = self.input_layer(x)
        x = self.h0(x)
        x = self.h0_act(x)
        x = self.h1(x)
        x = self.h1_act(x)
        x = self.output_layer(x)

        return x

class DynamicsEnsemble():
    def __init__(self, input_dim, output_dim, n_models = 4, n_neurons = 512, threshold = 1.5, n_layers = 2, activation = nn.ReLU, cuda = True):
        self.n_models = 4

        self.threshold = threshold

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.models = []

        for i in range(n_models):
            if(cuda):
                self.models.append(DynamicsNet(input_dim,
                                            output_dim,
                                            n_neurons = n_neurons,
                                            activation = activation).cuda())
            else:
                self.models.append(DynamicsNet(input_dim,
                                            output_dim,
                                            n_neurons = n_neurons,
                                            activation = activation))

    def forward(self, model, x):
        return self.models[model](x)

    def train_step(self, model_idx, feed, target):
        # Reset Gradients
        self.optimizers[model_idx].zero_grad()

        # Feed forward
        next_state_pred = self.models[model_idx](feed)
        output = self.losses[model_idx](next_state_pred, target)

        # Feed backwards
        output.backward()

        # Weight update
        self.optimizers[model_idx].step()

        # Tensorboard
        return output


    def train(self, dataloader, epochs = 5, optimizer = torch.optim.Adam, loss = nn.MSELoss, summary_writer = None, comet_experiment = None):

        hyper_params = {
            "dynamics_n_models":  self.n_models,
            "usad_threshold": self.threshold,
            "dynamics_epochs" : 5
        }
        if(comet_experiment is not None):
            comet_experiment.log_parameters(hyper_params)

        # Define optimizers and loss functions
        self.optimizers = [None] * self.n_models
        self.losses = [None] * self.n_models

        for i in range(self.n_models):
            self.optimizers[i] = optimizer(self.models[i].parameters())
            self.losses[i] = loss()

        # Start training loop
        for epoch in range(epochs):
            for i, batch in enumerate(tqdm(dataloader)):
                # Split batch into input and output
                feed, target = batch

                loss_vals = list(map(lambda i: self.train_step(i, feed, target), range(self.n_models)))

                # Tensorboard
                if(summary_writer is not None):
                    for j, loss_val in enumerate(loss_vals):
                        summary_writer.add_scalar('Loss/dynamics_{}'.format(j), loss_val, epoch*len(dataloader) + i)

                if(comet_experiment is not None and i % 10 == 0):
                    for j, loss_val in enumerate(loss_vals):
                        comet_experiment.log_metric('dyn_model_{}_loss'.format(j), loss_val, epoch*len(dataloader) + i)
                        comet_experiment.log_metric('dyn_model_avg_loss'.format(j), sum(loss_vals)/len(loss_vals), epoch*len(dataloader) + i)


    def usad(self, predictions):
        # Compute the pairwise distances between all predictions
        distances = scipy.spatial.distance_matrix(predictions, predictions)

        # If maximum is greater than threshold, return true
        return (np.amax(distances) > self.threshold)

    def predict(self, x):
        # Generate prediction of next state using dynamics model
        with torch.set_grad_enabled(False):
            return torch.stack(list(map(lambda i: self.forward(i, x), range(self.n_models))))

    def save(self, save_dir):
        for i in range(self.n_models):
            torch.save(self.models[i].state_dict(), os.path.join(save_dir, "dynamics_{}.pt".format(i)))

    def load(self, load_dir):
        for i in range(self.n_models):
            self.models[i].load_state_dict(torch.load(os.path.join(load_dir, "dynamics_{}.pt".format(i))))