# Copyright 2019 Ondrej Skopek.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTICE: This file has been changed from its original form.
# ==============================================================================

from collections import defaultdict
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .vae import ModelVAE
from ..components import StereographicallyProjectedSphereComponent, PoincareComponent
from ..components import SphericalComponent, HyperbolicComponent
from ..ops import hyperbolics as H
from ..stats import Stats, EpochStats
from ..utils import CurvatureOptimizer


class Trainer:

    def __init__(self,
                 model: ModelVAE) -> None:
        self.model = model
        self.stats = Stats()
        
        self.epoch_train_results = {}
        self.epoch_train_stats = {}

    @property
    def epoch(self) -> int:
        return self.stats.epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self.stats.epoch = value

    def get_beta(self, betas: Optional[Sequence[float]]) -> float:
        if betas is None:
            return 1.0
        elif self.epoch >= len(betas):
            return betas[-1]
        else:
            return betas[self.epoch]

    def train_epochs(self,
                       optimizer: Any,
                       train_data: DataLoader,
                       betas: Sequence[float],
                       likelihood_n: int = 500,
                       max_epochs: int = 1000,
                       visualize_information = None
                       ) -> Dict[int, EpochStats]:
        """
        optimizer: The optimizer.
        train_data: The data loader for loading the data.
        betas: Weights to specify the strength of the KL-divergence. The same beta is used for each latent component.
        likelihood_n: How many samples to use to perform Monte Carlo integration to compute the log-likelihood.
        max_epochs: The number of epochs to train the model for.
        visualize_information: None to save no intermediate embeddings when training. A dictionary if saving
         the intermediate embeddings is desired. Need the keys "x" and "y" where "x" is the full scRNA-seq dataset and "y" is the batch vector.
         The keys "embeddings_save_path" and "model_name" are also needed where "embeddings_save_path" is the path to save the embeddings to
          and "model_name" is the name of the model being used such as "r2, e10". The key "epochs" is needed and the value is a list
           of integers which speicfy which epochs to save the embeddings for. The key "device" is needed to specify the device being used such as
            torch.device("cpu") or torch.device("cuda").
          In summary, visualization_information should contain the keys "x", "y", "embeddings_save_path", "model_name", "epochs", and "device"
           if the intermediate embeddings during training are desired.
        """
        train_results = dict()
        test_results = dict()

        count = 0
        if visualize_information:
            self.check_visualize_information(visualize_information)
            x = visualize_information["x"]
            y = visualize_information["y"]
            device = visualization_information["device"]
            a = self.model(torch.log1p(torch.tensor(x,device=device)), torch.tensor(y,device=device))
            b = a[4]
            embeddings_save_path = visualize_information["embeddings_save_path"]
            model_name = visualize_information["model_name"]
            bb = b.detach().to(torch.device("cpu")).numpy()
            np.savetxt(os.path.join(embeddings_save_path, f'{model_name}_all_encode_v63_initialization_z_params.txt'), bb)
        
        for _ in range(max_epochs):
            beta = self.get_beta(betas)
            train_results[self.epoch] = self._train_epoch(optimizer, train_data, beta=beta, likelihood_n=likelihood_n)
            self.epoch += 1
            
            if visualize_information:
                self.check_visualize_information(visualize_information)
                if count in visualize_information["epochs"]:
                    x = visualize_information["x"]
                    y = visualize_information["y"]
                    device = visualization_information["device"]
                    a = self.model(torch.log1p(torch.tensor(x,device=device)), torch.tensor(y,device=device))
                    b = a[4]
                    embeddings_save_path = visualize_information["embeddings_save_path"]
                    model_name = visualize_information["model_name"]
                    bb = b.detach().to(torch.device("cpu")).numpy()
                    np.savetxt(os.path.join(embeddings_save_path, f'{model_name}_all_encode_v63_epoch_{count}_z_params.txt'), bb)
            count = count + 1

    def _train_epoch(self, optimizer: torch.optim.Optimizer, train_data: DataLoader, beta: float, likelihood_n: int = 500) -> EpochStats:
        print(f"\tTrainEpoch {self.epoch}:\t", end="")
        self.model.train()

        batch_stats = []
        for x_mb, y_mb in train_data:
            stats, (reparametrized, _, _) = self.model.train_step(optimizer, x_mb, y_mb, beta=beta, epoch_num=self.epoch, likelihood_n=likelihood_n)

            self.stats.global_step += 1
            batch_stats.append(stats)

        epoch_stats = EpochStats(batch_stats, length=len(train_data.dataset))
        epoch_dict = epoch_stats.to_print()
        for i, component in enumerate(self.model.components):
            name = f"{component.summary_name(i)}/curvature"
            epoch_dict[name] = float(component.manifold.curvature)
        
        rounded_epoch_dict = {}
        for key in epoch_dict:
            rounded_epoch_dict[key] = round(epoch_dict[key], 3)
        print(rounded_epoch_dict, flush=True)
        self.epoch_train_results[self.epoch] = epoch_dict
        self.epoch_train_stats[self.epoch] = epoch_stats

        return epoch_stats

    def build_optimizer(self, learning_rate: float, fixed_curvature: bool, use_adamw: bool = False, weight_decay: Optional[float] = 0) -> torch.optim.Optimizer:

        def ncurvature_param_cond(key: str) -> bool:
            return "nradius" in key or "curvature" in key

        def pcurvature_param_cond(key: str) -> bool:
            return "pradius" in key

        net_params = [
            v for key, v in self.model.named_parameters()
            if not ncurvature_param_cond(key) and not pcurvature_param_cond(key)
        ]
        neg_curv_params = [v for key, v in self.model.named_parameters() if ncurvature_param_cond(key)]
        pos_curv_params = [v for key, v in self.model.named_parameters() if pcurvature_param_cond(key)]
        curv_params = neg_curv_params + pos_curv_params

        if use_adamw:
            if weight_decay is not None:
                net_optimizer = torch.optim.AdamW(net_params, lr=learning_rate, weight_decay=weight_decay)
            else:
                net_optimizer = torch.optim.AdamW(net_params, lr=learning_rate)
        else:
            net_optimizer = torch.optim.Adam(net_params, lr=learning_rate)
        if not fixed_curvature and not curv_params:
            warnings.warn("Fixed curvature disabled, but found no curvature parameters. Did you mean to set "
                          "fixed=True, or not?")
        if not pos_curv_params:
            c_opt_pos = None
        else:
            c_opt_pos = torch.optim.SGD(pos_curv_params, lr=1e-4)

        if not neg_curv_params:
            c_opt_neg = None
        else:
            c_opt_neg = torch.optim.SGD(neg_curv_params, lr=1e-4)

        def condition() -> bool:
            return (not fixed_curvature) and (self.epoch >= 10) and (self.stats.global_step % 1 == 0)

        return CurvatureOptimizer(net_optimizer, neg=c_opt_neg, pos=c_opt_pos, should_do_curvature_step=condition)

    def check_visualize_information(self, visualize_information):
        required_keys = ["x", "y", "embeddings_save_path", "model_save_name", "epochs", "device"]
        for each_key in required_keys:
            if each_key not in visualize_information:
                raise AssertionError(f"Need to specify {each_key} if want to save intermediate embeddings")
    
    def plot_loss(self, desired_loss="elbo", save=None):
        epochs = sorted(self.epoch_train_results.keys())

        losses = [self.epoch_train_results[epoch_num][desired_loss] for epoch_num in epochs]
            
        plt.plot(epochs, losses)
        plt.xlabel("Epoch")
        plt.ylabel(desired_loss)
            
        if save:
            plt.savefig(save)
        plt.show()
        plt.close()
