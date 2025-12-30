from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from .vae import ModelVAE
from ...data import VaeDataset
from ..components import Component

from ml_collections import ConfigDict
from typing import Optional

EPS = 1e-6
MAX_SIGMA_SQUARE = 1e10


class NBVAE(ModelVAE):

    def __init__(self, h_dim: int, components: List[Component],
                 mask,
                 dataset: VaeDataset,
                 config: ConfigDict,
                 component_subspaces=None,
                 component_no_grads=None, 
                 component_batch=None) -> None:
        """
        h_dim: The last dimension of the encoder.
        components: The parsed list of latent components, one component for each subspace.
        mask: If there are k latent components, the mask will also have k components where the kth component in the mask
          will be used to output the parameters of the posterior distribution for the kth latent component.
        dataset: The dataset to train the model on.
        config: A ConfigDict which specifies the hyperparameters of the model. The list is in get_config.py.
        component_subspaces: Specifies which latent components to reconstruct the components with.
        component_no_grads: A general parameter for specifying which latent components to stop the gradient for.
        """
        super().__init__(h_dim,
                         components,
                         mask,
                         dataset,
                         config,
                         component_subspaces,
                         component_no_grads,
                         component_batch)

        self.in_dim = dataset.in_dim
        self.h_dim = h_dim

        self.device = config.device
                     
        n_batch = config.n_batch
        if type(n_batch) != list:
            n_batch = [n_batch]
        self.n_batch = n_batch
        self.batch_invariant = config.batch_invariant
        
        total_num_of_batches = sum(self.n_batch)
        self.total_num_of_batches = total_num_of_batches
        
        # multi-layer
        # http://adamlineberry.ai/vae-series/vae-code-experiments
        
        self.activation = config.activation
        
        # construct the encoder
        input_dim = dataset.in_dim
        if not self.batch_invariant:
            input_dim = input_dim + self.total_num_of_batches
        encoder_szs = [input_dim] + [128, 64, h_dim]
        encoder_layers = []
        for in_sz, out_sz in zip(encoder_szs[:-1], encoder_szs[1:]):
            encoder_layers.append(nn.Linear(in_sz, out_sz, bias=True))
            if config.use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(out_sz, momentum=config.momentum, eps=config.eps))
            if self.activation == "relu":
                encoder_layers.append(nn.ReLU())
            elif self.activation == "leaky_relu":
                encoder_layers.append(nn.LeakyReLU())
            elif self.activation == "tanh":
                encoder_layers.append(nn.Tanh())
            else:
                encoder_layers.append(nn.GELU())
            
        self.encoder = nn.Sequential(*encoder_layers)

        # construct the decoder
        hidden_sizes = [self.total_z_dim + self.total_num_of_batches] + [64, 128]
        decoder_layers = []
        for in_sz, out_sz in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            decoder_layers.append(nn.Linear(in_sz, out_sz, bias=True))
            if config.use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(out_sz, momentum=config.momentum, eps=config.eps))
            if self.activation == "relu":
                decoder_layers.append(nn.ReLU())
            elif self.activation == "leaky_relu":
                decoder_layers.append(nn.LeakyReLU())
            elif self.activation == "tanh":
                decoder_layers.append(nn.Tanh())
            else:
                decoder_layers.append(nn.GELU())
            
        self.decoder = nn.Sequential(*decoder_layers)
        
        output_dim = dataset.in_dim
        self.fc_mu = nn.Linear(128, output_dim)
        self.fc_sigma = nn.Linear(128, output_dim)

        if config.init == "normal":
            self.apply(self._init_weights_normal)
        elif config.init == "xavier_uniform":
            self.apply(self._init_weights_xavier_uniform)
        elif config.init == "xavier_normal":
            self.apply(self._init_weights_xavier_normal)
        elif config.init == "he_uniform":
            self.apply(self._init_weights_he_uniform)
        elif config.init == "he_normal":
            self.apply(self._init_weights_he_normal)
        elif config.init == "custom":
            self.components[0].apply(self._init_weights_xavier_uniform)
        elif config.init == "custom_xavier_normal":
            self.components[0].apply(self._init_weights_xavier_normal)

    def encode(self, x: Tensor, batch: Tensor) -> Tensor:
        x = x.squeeze()

        assert len(x.shape) == 2
        bs, dim = x.shape

        assert dim == self.in_dim
        x = x.view(bs, self.in_dim)

        if not self.batch_invariant and self.total_num_of_batches != 0:
            x = torch.concat((x, batch), dim=1)
        x = self.encoder(x)

        return x.view(bs, -1)  # such that x is batch * dim, similar to reshape (no need)

    def decode(self, concat_z: Tensor, batch: Tensor, use_z_batch: bool = True):
        assert len(concat_z.shape) >= 2
        bs = concat_z.size(-2)

        if not use_z_batch:
            batch = torch.zeros(batch.shape).to(self.device)
        concat_z = torch.concat((concat_z, batch), dim=1)
        x = self.decoder(concat_z)

        mu = torch.nn.functional.softmax(self.fc_mu(x), -1)

        sigma_square = torch.nn.functional.softplus(self.fc_sigma(x))
        sigma_square = torch.mean(sigma_square, 0)
        sigma_square = torch.clamp(sigma_square, EPS, MAX_SIGMA_SQUARE)

        mu = mu.view(-1, bs, self.in_dim)  # flatten
        
        return mu.squeeze(dim=0), sigma_square
