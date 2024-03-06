import argparse
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Type

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import tqdm
from matplotlib.gridspec import GridSpec
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR

#stuff for b-rex
from .bayesian_rex import compute_l2, calc_linearized_pairwise_ranking_loss, mcmc_map_search
from .lexicase_rex import calc_lexicase_scores, lexicase_search, select_one

import inspect

class BaseRewardModel(nn.Module):
    network: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        *,
        state_dim: int,
        output_dim: int = 1,
        num_layers: int = 4,
        hidden_dim: int = 512,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        layers: List[nn.Module] = []
        for layer_index in range(num_layers - 1):
            if layer_index == 0:
                layers.append(nn.Linear(state_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_index < num_layers - 1:
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
        
        if (num_layers > 1):
            self.network = nn.Sequential(*layers)
            self.last_layer = nn.Linear(hidden_dim, output_dim) ##this is seperated for lexicase stuff
        else:
            #identity
            self.network = nn.Identity()
            self.last_layer = nn.Linear(state_dim, output_dim) ##this is seperated for lexicase stuff

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.last_layer(F.relu(self.network(state))).squeeze(-1)

    def get_penultimate_layer(self, state: torch.Tensor) -> torch.Tensor:
        return F.relu(self.network(state)).squeeze(-1)

    def preference_logp(
        self, state0: torch.Tensor, state1: torch.Tensor, preferences: torch.Tensor
    ) -> torch.Tensor:
        """
        Return the log probability of the given preference comparisons according to the
        model. If preferences[i] == 0, then state0 is preferred to state1, and vice
        versa.
        """

        reward0 = self.forward(state0)
        reward1 = self.forward(state1)
        reward_diff = reward0 - reward1
        reward_diff[preferences == 1] *= -1
        return -F.softplus(-reward_diff)


# class BaseModelWithPenultimateLayer(BaseRewardModel):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def forward(self, state: torch.Tensor) -> torch.Tensor:
#         return self.network(state)  
    
#     def get_penultimate_layer(self, state: torch.Tensor) -> torch.Tensor:
#         return self.network(state).squeeze(-1)
    
#     def preference_logp(
#         self, state0: torch.Tensor, state1: torch.Tensor, preferences: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Return the log probability of the given preference comparisons according to the
#         model. If preferences[i] == 0, then state0 is preferred to state1, and vice
#         versa.
#         """

#         reward0 = self.forward(state0)
#         reward1 = self.forward(state1)
#         reward_diff = reward0 - reward1
#         reward_diff[preferences == 1] *= -1
#         return -F.softplus(-reward_diff)
    

class MeanAndVarianceRewardModel(BaseRewardModel):
    def __init__(self, *args, max_std=np.inf, **kwargs):
        super().__init__(*args, output_dim=2, **kwargs)
        self.max_std = max_std

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        mean_and_log_std = self.network(state)
        # return mean_and_log_std
        mean = mean_and_log_std[:, 0]
        log_std = mean_and_log_std[:, 1] - 2
        log_std = log_std.clamp(max=np.log(self.max_std))
        return torch.stack([mean, log_std], dim=1)

    def preference_logp(
        self, state0: torch.Tensor, state1: torch.Tensor, preferences: torch.Tensor
    ) -> torch.Tensor:
        output0 = self.forward(state0)
        output1 = self.forward(state1)
        mean0 = output0[:, 0]
        log_std0 = output0[:, 1]
        mean1 = output1[:, 0]
        log_std1 = output1[:, 1]

        diff_mean = mean0 - mean1
        diff_mean[preferences == 1] *= -1
        var_combined = torch.exp(log_std0) ** 2 + torch.exp(log_std1) ** 2
        # p: torch.Tensor = Normal(0, torch.sqrt(var_combined)).cdf(diff_mean)
        z = diff_mean / torch.sqrt(var_combined)
        # Based on approximation here: https://stats.stackexchange.com/a/452121
        return -F.softplus(-z * np.sqrt(2 * np.pi))
        # logp = torch.log(p.clamp(min=1e-4))
        # return logp


class CategoricalRewardModel(BaseRewardModel):
    comparison_matrix: torch.Tensor

    def __init__(
        self, *args, num_atoms: Optional[int] = None, state_dim: int, **kwargs
    ):
        if num_atoms is None:
            if state_dim == 1:
                num_atoms = 20
            else:
                num_atoms = 8
        super().__init__(
            *args,
            output_dim=num_atoms,
            use_batchnorm=True,
            state_dim=state_dim,
            **kwargs,
        )

        comparison_matrix = torch.empty((num_atoms, num_atoms))
        atom_values = torch.linspace(0, 1, num_atoms)
        comparison_matrix[:] = atom_values[None, :] > atom_values[:, None]
        comparison_matrix[atom_values[None, :] == atom_values[:, None]] = 0.5
        self.register_buffer("comparison_matrix", comparison_matrix)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.network(state), dim=-1)

    def preference_logp(
        self, state0: torch.Tensor, state1: torch.Tensor, preferences: torch.Tensor
    ) -> torch.Tensor:
        dist0 = self.forward(state0)
        dist1 = self.forward(state1)
        prob1 = ((dist0 @ self.comparison_matrix) * dist1).sum(dim=1)
        prob = prob1.clone()
        prob[preferences == 0] = (1 - prob1)[preferences == 0]
        return prob.log()


class ClassifierRewardModel(BaseRewardModel):
    def __init__(self, *args, state_dim, **kwargs):
        super().__init__(*args, state_dim=state_dim * 2, **kwargs)

    def forward(self, state0: torch.Tensor, state1: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.network(torch.cat([state0, state1], dim=-1))

    def preference_logp(
        self, state0: torch.Tensor, state1: torch.Tensor, preferences: torch.Tensor
    ) -> torch.Tensor:
        """
        Return the log probability of the given preference comparisons according to the
        model. If preferences[i] == 0, then state0 is preferred to state1, and vice
        versa.
        """

        logits = self.forward(state0, state1)
        logits[preferences == 0] *= -1
        return -F.softplus(-logits)


def train_rlhf(
    *,
    reward_model: BaseRewardModel,
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    sample_state: Callable[[int], torch.Tensor],
    batch_size: int,
    lr: float,
    num_iterations: int,
    device: torch.device,
    flip_percentage: float = 0,
) -> BaseRewardModel:
    """
    Trains and returns a reward function using RLHF.
    Args:
        reward_fn: The ground truth reward/utility function, which can be randomized.
        state_dim: The dimension of the state space.
        sample_state: A function that samples states from the state space, i.e.
            sample_state(n) returns n samples from the state space.
        batch_size: The batch size for training the reward function.
        num_iterations: The number of iterations to train the reward function.
    Returns:
        The trained reward function as a PyTorch neural network.
    """

    optimizer = optim.Adam(reward_model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=(1e-5 / lr) ** (1 / num_iterations))
    reward_model.to(device).train()
    progress_bar = tqdm.tqdm(range(num_iterations))
    for _ in progress_bar:
        optimizer.zero_grad()
        state0 = sample_state(batch_size).to(device)
        state1 = sample_state(batch_size).to(device)
        if (len(inspect.signature(reward_fn).parameters.keys()) == 2):
            identity = torch.randint(0, 2, state0.shape, device=state0.device)
            rewards0 = reward_fn(state0, identity)
            rewards1 = reward_fn(state1, identity)
        else:
            rewards0 = reward_fn(state0)
            rewards1 = reward_fn(state1)
        preferences = (rewards1 > rewards0).long()

        #randomly flip
        flip = torch.rand(preferences.squeeze(-1).shape, device=preferences.device) < flip_percentage
        # if 0 -> 1, if 1 -> 0
        preferences[flip] = 1 - preferences[flip]

        loss = -reward_model.preference_logp(state0, state1, preferences).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        progress_bar.set_description(
            f"loss = {loss.item():.2f}    lr = {scheduler.get_lr()[0]:.2e}"  # type: ignore
        )

    return reward_model

def train_brex(
    *,
    reward_model: BaseRewardModel,
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    sample_state: Callable[[int], torch.Tensor],
    batch_size: int,
    lr: float,
    num_iterations: int,
    device: torch.device,
    likelihood: str = "bradley-terry",
    flip_percentage: float = 0,
) -> BaseRewardModel:
    """Trains and returns a reward model using Bayesian Reward Extrapolation.
    Args:
        reward_fn: The ground truth reward/utility function, which can be randomized.
        state_dim: The dimension of the state space.
        sample_state: A function that samples states from the state space, i.e.
            sample_state(n) returns n samples from the state space.
        batch_size: The batch size for training the reward function.
        num_iterations: The number of iterations to train the reward function.
    """
    #first, we will train a model with RLHF, and delete the last layer
    #reward_model = train_rlhf(reward_model=reward_model,
    #                          reward_fn= reward_fn,
    #                        sample_state = sample_state,
    #                        batch_size = batch_size,
    #                        lr = lr,
    #                        num_iterations = int(num_iterations),
    #                        device = device)

    state0 = sample_state(batch_size).to(device)
    state1 = sample_state(batch_size).to(device)
    features0 = reward_model.get_penultimate_layer(state0)
    features1 = reward_model.get_penultimate_layer(state1)


    if (len(inspect.signature(reward_fn).parameters.keys()) == 2):
        identity = torch.randint(0, 2, state0.shape, device=state0.device)
        #identity = torch.rand(state0.shape, device=state0.device) < 0.5
        rewards0 = reward_fn(state0, identity)
        rewards1 = reward_fn(state1, identity)
    else:
        rewards0 = reward_fn(state0)
        rewards1 = reward_fn(state1)
    
    preferences = (rewards1 > rewards0).long()

    #randomly flip
    flip = torch.rand(preferences.shape, device=preferences.device) < flip_percentage
    # if 0 -> 1, if 1 -> 0
    preferences[flip] = 1 - preferences[flip]

    #print(f"labels: {labels}")
    #print(f"labels shape: {labels.shape}")

    best_reward_lastlayer, chain, logliks = mcmc_map_search(reward_model, preferences, features0, features1, num_iterations*10, 0.01, 1, likelihood)

    #interpolating the features to get variances
    state_interpolated = torch.linspace(0, 1, 100).to(device)
    features_interpolated = reward_model.get_penultimate_layer(state_interpolated[:, None])
    chain = torch.from_numpy(chain).squeeze(1)
    print(f"features interpolated shape: {features_interpolated.shape}")
    print(f"chain shape: {chain.shape}")
    returns = features_interpolated @ chain.T
    mean_model = returns.mean(axis=1)
    variances = torch.var(returns, axis=1)


    if (len(inspect.signature(reward_fn).parameters.keys()) == 2):
        true_rewards_1 = reward_fn(state_interpolated, torch.tensor(0))
        true_rewards_2 = reward_fn(state_interpolated, torch.tensor(1))

        preference_matrix_1 = torch.zeros((100, 100), dtype=torch.long)
        preference_matrix_2 = torch.zeros((100, 100), dtype=torch.long)

        for i in range(100):
            for j in range(100):
                if (true_rewards_1[i] >= true_rewards_1[j]):
                    preference_matrix_1[i, j] = 1

                if (true_rewards_2[i] >= true_rewards_2[j]):
                    preference_matrix_2[i, j] = 1
            
        
        #get preference passes
        passes_1 = preference_pass_matrix(returns, preference_matrix_1)
        passes_2 = preference_pass_matrix(returns, preference_matrix_2)

        #sum of passes, save to text file
        sum_passes_1 = torch.sum(passes_1)
        sum_passes_2 = torch.sum(passes_2)

        s0 = sample_state(40).to(device)
        s1 = sample_state(40).to(device)
        f0 = reward_model.get_penultimate_layer(s0)
        f1 = reward_model.get_penultimate_layer(s1)
        r0_z1 = reward_fn(s0, torch.tensor(1))
        r1_z1 = reward_fn(s1, torch.tensor(1))
        r0_z0 = reward_fn(s0, torch.tensor(0))
        r1_z0 = reward_fn(s1, torch.tensor(0))
        pp0 = (r1_z0 > r0_z0).long()
        pp1 = (r1_z1 > r0_z1).long()

        ind0, score0 = select_one(chain, f0, f1, pp0) 
        
        ret0 = features_interpolated @ ind0.T
        ind1, score1 = select_one(chain, f0, f1, pp1)
        ret1 = features_interpolated @ ind1.T

        print(f"ind0: {ind0}")
        print(f"scores0: {score0}")
        print(f"ind1: {ind1}")
        print(f"scores1: {score1}")

        #normalize ret0 and ret1
        ret0 = ret0 / torch.norm(ret0)
        ret1 = ret1 / torch.norm(ret1)

        #graph these individuals
        #make it pretty
        #plt.plot(state_interpolated.detach().numpy(), ret0.detach().numpy(), label="0")
        #plt.plot(state_interpolated.detach().numpy(), ret1.detach().numpy(), label="1")
        #scale each line by its own min and max
        fig, ax1 = plt.subplots(figsize=(5, 5))
        ax2 = ax1.twinx()
        ax1.plot(state_interpolated.detach().numpy(), ret0.detach().numpy(), label="0", color='blue')
        ax2.plot(state_interpolated.detach().numpy(), ret1.detach().numpy(), label="1", color='red')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)

        plt.savefig(f'./results/1d_identity/{batch_size}_{lr}_{num_iterations}/b_rex_individuals_{likelihood}_{num_iterations}_{flip_percentage}.png')
        plt.close()

        with open(f'./results/1d_identity/{batch_size}_{lr}_{num_iterations}/b_rex_num_passes_{likelihood}_{num_iterations}_{flip_percentage}.txt', 'w') as f:
            f.write(f"sum_passes_1: {sum_passes_1}\n")
            f.write(f"sum_passes_2: {sum_passes_2}\n")

        #save heatmap
        #plt.imshow(passes_1, cmap='hot', interpolation='nearest', aspect='auto')
        #colorbar should go between 0 and 10,000
        plt.pcolor(passes_1, cmap='hot', vmin=0, vmax=10000)
        plt.colorbar()
        plt.savefig(f'./results/1d_identity/{batch_size}_{lr}_{num_iterations}/b_rex_passes_1_{likelihood}_{num_iterations}_{flip_percentage}.png')
        plt.close()
        
        plt.pcolor(passes_2, cmap='hot', vmin=0, vmax=10000)
        plt.colorbar()
        plt.savefig(f'./results/1d_identity/{batch_size}_{lr}_{num_iterations}/b_rex_passes_2_{likelihood}_{num_iterations}_{flip_percentage}.png')
        plt.close()

    else:
        true_rewards = reward_fn(state_interpolated)

        #get the ground truth preference matrix for all states
        preference_matrix  = torch.zeros((100, 100), dtype=torch.long)
        for i in range(100):
            for j in range(100):
                if (true_rewards[i] >= true_rewards[j]):
                    preference_matrix[i, j] = 1

        #get preference passes
        passes = preference_pass_matrix(returns, preference_matrix)

        #sum of passes, save to text file
        sum_passes = torch.sum(passes)
        with open(f'./results/1d_identity/{batch_size}_{lr}_{num_iterations}/b_rex_num_passes_{likelihood}_{num_iterations}_{flip_percentage}.txt', 'w') as f:
            f.write(f"sum_passes: {sum_passes}\n")

        #save heatmap
        plt.pcolor(passes, cmap='hot', vmin=0, vmax=10000)
        plt.colorbar()

        plt.savefig(f'./results/1d_identity/{batch_size}_{lr}_{num_iterations}/b_rex_passes_{likelihood}_{num_iterations}_{flip_percentage}.png')
        plt.close()


    best_reward_model = reward_model
    best_reward_lastlayer = best_reward_lastlayer.to(device)
    best_reward_model.last_layer = best_reward_lastlayer

    

    
    #unique_indicies = remove_same_rankings(returns)

    #all_models = returns[:, unique_indicies]
    all_models = returns
    
    #print(f"best_reward_lastlayer: {[p for p in best_reward_lastlayer.parameters()]}")
    #print(f"best_reward_model: {[p for p in best_reward_model.last_layer.parameters()]}")

    #evaluate the model
    state0 = sample_state(batch_size).to(device)
    state1 = sample_state(batch_size).to(device)
    if (len(inspect.signature(reward_fn).parameters.keys()) == 2):
        identity = torch.randint(0, 2, state0.shape, device=state0.device)
        rewards0 = reward_fn(state0, identity)
        rewards1 = reward_fn(state1, identity)
    else:
        rewards0 = reward_fn(state0)
        rewards1 = reward_fn(state1)
    loss = -best_reward_model.preference_logp(state0, state1, (rewards1 > rewards0).long()).mean()
    print(f"loss for B_REX MAP: {loss.item()}")

    return best_reward_model, variances, mean_model, all_models

def train_lex(
    *,
    reward_model: BaseRewardModel,
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    sample_state: Callable[[int], torch.Tensor],
    batch_size: int,
    lr: float,
    num_iterations: int,
    device: torch.device,
    flip_percentage: float = 0,
) -> BaseRewardModel:
    """Trains and returns a reward model using lexicase selection.
    Args:
        reward_fn: The ground truth reward/utility function, which can be randomized.
        state_dim: The dimension of the state space.
        sample_state: A function that samples states from the state space, i.e.
            sample_state(n) returns n samples from the state space.
        batch_size: The batch size for training the reward function.
        num_iterations: The number of iterations to train the reward function.
    """
    #first, we will train a model with RLHF, and delete the last layer
    #reward_model = train_rlhf(reward_model=reward_model,
    #                          reward_fn= reward_fn,
    #                        sample_state = sample_state,
    #                        batch_size = batch_size,
    #                        lr = lr,
    #                        num_iterations = int(num_iterations),
    #                        device = device)

    state0 = sample_state(batch_size).to(device)
    state1 = sample_state(batch_size).to(device)
    features0 = reward_model.get_penultimate_layer(state0)
    features1 = reward_model.get_penultimate_layer(state1)

    if (len(inspect.signature(reward_fn).parameters.keys()) == 2):
        identity = torch.randint(0, 2, state0.shape, device=state0.device)
        rewards0 = reward_fn(state0, identity)
        rewards1 = reward_fn(state1, identity)
    else:
        rewards0 = reward_fn(state0)
        rewards1 = reward_fn(state1)
    preferences = (rewards1 > rewards0).long()

    #randomly flip
    flip = torch.rand(preferences.shape, device=preferences.device) < flip_percentage
    # if 0 -> 1, if 1 -> 0
    preferences[flip] = 1 - preferences[flip]

    #print(f"labels: {labels}")
    #print(f"labels shape: {labels.shape}")

    population, scores, best_scores = lexicase_search(preferences, features0, features1, num_iterations//10, num_iterations, 0.01, normalize=True, downsample_level=5)

    sum_scores = np.sum(scores, axis=1)
    best_reward_lastlayer = torch.from_numpy(population[np.argmax(sum_scores)])
    print(f"best_reward_lastlayer shape: {best_reward_lastlayer.shape}")
    print(f"best_reward_lastlayer: {best_reward_lastlayer}")
    best_reward_lastlayer = best_reward_lastlayer.to(device)

    #interpolating the features to get variances
    state_interpolated = torch.linspace(0, 1, 100).to(device)
    

    features_interpolated = reward_model.get_penultimate_layer(state_interpolated[:, None])
    # chain = torch.from_numpy(chain).squeeze(1)
    population = torch.from_numpy(population).to(device).squeeze(1).to(torch.float32)
    print(f"features interpolated dtype: {features_interpolated.dtype}")
    print(f"chain dtype: {population.dtype}")
    returns = features_interpolated @ population.T

    if (len(inspect.signature(reward_fn).parameters.keys()) == 2):
        true_rewards_1 = reward_fn(state_interpolated, torch.tensor(0))
        print(f"true_rewards_1 shape: {true_rewards_1.shape}")
        print(f"true_rewards_1: {true_rewards_1}")

        true_rewards_2 = reward_fn(state_interpolated, torch.tensor(1))
        print(f"true_rewards_2 shape: {true_rewards_2.shape}")
        print(f"true_rewards_2: {true_rewards_2}")

        preference_matrix_1 = torch.zeros((100, 100), dtype=torch.long)
        preference_matrix_2 = torch.zeros((100, 100), dtype=torch.long)

        for i in range(100):
            for j in range(100):
                if (true_rewards_1[i] >= true_rewards_1[j]):
                    preference_matrix_1[i, j] = 1

                if (true_rewards_2[i] >= true_rewards_2[j]):
                    preference_matrix_2[i, j] = 1

        #pick one more individual

        s0 = sample_state(40).to(device)
        s1 = sample_state(40).to(device)
        f0 = reward_model.get_penultimate_layer(s0)
        f1 = reward_model.get_penultimate_layer(s1)
        r0_z1 = reward_fn(s0, torch.tensor(1))
        r1_z1 = reward_fn(s1, torch.tensor(1))
        r0_z0 = reward_fn(s0, torch.tensor(0))
        r1_z0 = reward_fn(s1, torch.tensor(0))
        pp0 = (r1_z0 > r0_z0).long()
        pp1 = (r1_z1 > r0_z1).long()

        ind0, score0 = select_one(population, f0, f1, pp0) 
        
        ret0 = features_interpolated @ ind0.T
        ind1, score1 = select_one(population, f0, f1, pp1)
        ret1 = features_interpolated @ ind1.T

        print(f"ind0: {ind0}")
        print(f"scores0: {score0}")
        print(f"ind1: {ind1}")
        print(f"scores1: {score1}")

        #normalize ret0 and ret1
        ret0 = ret0 / torch.norm(ret0)
        ret1 = ret1 / torch.norm(ret1)

        #graph these individuals
        #make it pretty
        #plt.plot(state_interpolated.detach().numpy(), ret0.detach().numpy(), label="0")
        #plt.plot(state_interpolated.detach().numpy(), ret1.detach().numpy(), label="1")
        fig, ax1 = plt.subplots(figsize=(5, 5))
        ax2 = ax1.twinx()
        ax1.plot(state_interpolated.detach().numpy(), ret0.detach().numpy(), label="0", color='blue')
        ax2.plot(state_interpolated.detach().numpy(), ret1.detach().numpy(), label="1", color='red')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        #plt.legend()
        plt.savefig(f'./results/1d_identity/{batch_size}_{lr}_{num_iterations}/lexicase_individuals_{num_iterations}_{flip_percentage}.png')
        plt.close()


        #plot preference matrix
        plt.pcolor(preference_matrix_1, cmap='hot', vmin=0, vmax=100)
        plt.colorbar()
        plt.savefig(f'./results/1d_identity/{batch_size}_{lr}_{num_iterations}/lexicase_preference_1_{num_iterations}_{flip_percentage}.png')
        plt.close()

        plt.pcolor(preference_matrix_2, cmap='hot', vmin=0, vmax=100)
        plt.colorbar()
        plt.savefig(f'./results/1d_identity/{batch_size}_{lr}_{num_iterations}/lexicase_preference_2_{num_iterations}_{flip_percentage}.png')
        plt.close()
        
        #get preference passes
        passes_1 = preference_pass_matrix(returns, preference_matrix_1)
        passes_2 = preference_pass_matrix(returns, preference_matrix_2)

        #sum of passes, save to text file
        print(f"passes_1 shape: {passes_1.shape}")
        print(f"passes_2 shape: {passes_2.shape}")
        
        sum_passes_1 = torch.sum(passes_1)
        sum_passes_2 = torch.sum(passes_2)

        with open(f'./results/1d_identity/{batch_size}_{lr}_{num_iterations}/lexicase_num_passes_{num_iterations}_{flip_percentage}.txt', 'w') as f:
            f.write(f"sum_passes_1: {sum_passes_1}\n")
            f.write(f"sum_passes_2: {sum_passes_2}\n")

        #save heatmap
        plt.pcolor(passes_1, cmap='hot', vmin=0, vmax=100)
        plt.colorbar()
        plt.savefig(f'./results/1d_identity/{batch_size}_{lr}_{num_iterations}/lexicase_passes_1_{num_iterations}_{flip_percentage}.png')
        plt.close()
        
        plt.pcolor(passes_2, cmap='hot', vmin=0, vmax=100)
        plt.colorbar()
        plt.savefig(f'./results/1d_identity/{batch_size}_{lr}_{num_iterations}/lexicase_passes_2_{num_iterations}_{flip_percentage}.png')
        plt.close()

    else:
        true_rewards = reward_fn(state_interpolated)
        preference_matrix = torch.zeros((100, 100), dtype=torch.long)
        for i in range(100):
            for j in range(100):
                if (true_rewards[i] >= true_rewards[j]):
                    preference_matrix[i, j] = 1 #we prefer i over j

        #get preference passes
        passes = preference_pass_matrix(returns, preference_matrix)

        #sum of passes, save to text file
        sum_passes = torch.sum(passes)
        with open(f'./results/1d_identity/{batch_size}_{lr}_{num_iterations}/lexicase_num_passes_{num_iterations}_{flip_percentage}.txt', 'w') as f:
            f.write(f"sum_passes: {sum_passes}\n")

        #save heatmap
        plt.pcolor(passes, cmap='hot', vmin=0, vmax=100)
        plt.colorbar()

        plt.savefig(f'./results/1d_identity/{batch_size}_{lr}_{num_iterations}/lexicase_passes_{num_iterations}_{flip_percentage}.png')
        plt.close()

    #unique_indicies = remove_same_rankings(returns)

    mean_model = returns.mean(axis=1)
    print(f"mean_model shape: {mean_model.shape}")
    print(f"returns shape: {returns.shape}")
    #print(f"unique_indicies shape: {unique_indicies.shape}")
    #all_models = returns[:, unique_indicies]
    all_models = returns
    # print(f"returns shape: {returns.shape}")
    variances = torch.var(returns, axis=1)

    best_reward_model = reward_model
    best_reward_model.last_layer = torch.nn.Linear(reward_model.last_layer.in_features, 1, bias=False).to(device)
    
    print(f"best_reward_lastlayer shape: {best_reward_lastlayer.shape}")
    print(f"best_reward_lastlayer: {best_reward_lastlayer}")
    for param in best_reward_model.last_layer.parameters():
        print(f"param before: {param}")
        #copy param from best_reward_lastlayer
        param.data = best_reward_lastlayer
        print(f"param after: {param}" )

    #evaluate the model
    state0 = sample_state(batch_size).to(device)
    state1 = sample_state(batch_size).to(device)
    if (len(inspect.signature(reward_fn).parameters.keys()) == 2):
        identity = torch.randint(0, 2, state0.shape, device=state0.device)
        rewards0 = reward_fn(state0, identity)
        rewards1 = reward_fn(state1, identity)
    else:
        rewards0 = reward_fn(state0)
        rewards1 = reward_fn(state1)
    #loss = -best_reward_model.preference_logp(state0, state1, (rewards1 > rewards0).long()).mean()
    #print(f"loss for LEX BEST: {loss.item()}")

    return best_reward_model, variances, mean_model, all_models


def reward_fn_1d_identity(state: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    state = state.squeeze(-1)
    identity = identity.squeeze(-1)
    rewards = state.clone()
    rewards[(state >= 0.8) & (identity == 1)] *= 2
    rewards[(state >= 0.8) & (identity == 0)] *= 0

    return rewards

def reward_fn_1d(state: torch.Tensor) -> torch.Tensor:
    state = state.squeeze(-1)
    rewards = state.clone()
    double_rewards = torch.rand(rewards.shape, device=rewards.device) < 0.5
    rewards[(state >= 0.8) & double_rewards] *= 2
    rewards[(state >= 0.8) & ~double_rewards] *= 0
    return rewards

def reward_fn_2d(state: torch.Tensor) -> torch.Tensor:
    x, y = state[..., 0], state[..., 1]
    b = torch.rand(x.shape, device=x.device) < 1 - x * y
    rewards = torch.empty_like(x)
    rewards[b] = (y / (1 - x * y))[b]
    rewards[~b] = 0
    return rewards

#this reward function implements cyclical preferences.
#if identity is 0, then the reward is equal to the state
# if identity is 1, then the reward is equal to a + 0.66 if a < 0.33 and (a-0.33) otherwise
# if identity is 2, then the reward is equal to a + 0.33 if a < 0.66 and (a-0.66) otherwise
def reward_fn_cycles(state: torch.Tensor, identity: torch.tensor) -> torch.Tensor:
    state = state.squeeze(-1)
    identity = identity.squeeze(-1)
    rewards = state.clone()
    rewards[identity == 1] = (state[identity == 1] + 0.66) % 1
    rewards[identity == 2] = (state[identity == 2] + 0.33) % 1
    return rewards

def reward_fn_default(state: torch.Tensor) -> torch.Tensor:
    return state.squeeze(-1).clone()

#this function removes all reward hypotheses that are equivalent to each other based on how they rank
def remove_same_rankings(returns):
    #returns is a matrix of all returns for each hypothesis, there is a vector showing return for a = [0, 0.01, 0.02, ... 1]
    # returns a vector of all_hypotheses that are not equivalent
    print(f"SHAPE OF RETURNS: {returns.shape}")

    #for each hypotheses, we need to rank the states
    ranks = torch.argsort(returns, dim=0)
    print(f"ranks: {ranks}")

    #now, we need to combine all hypotheses that have the same ranking
    unique_ranks_output = torch.unique(ranks, dim=1, return_inverse=True)
    print(f"unique_ranks: {unique_ranks_output}")
    print(f"inverse: {unique_ranks_output[1]}")

    print(f"number unique: {unique_ranks_output[0].shape[1]}")

    return unique_ranks_output[1]

def sample_state_fn(n: int) -> torch.Tensor:
    #return a nroamlly distributed sample from [0, 1] with center 0.2
    return torch.rand((n, 1))
    #return torch.clip(torch.randn((n, 1)) * 0.1 + 0.2, 0, 1)


#given a set of returns, and a set of preferences,
# we calculate whether, for each preference, there is at least one hypothesis that is consistent with the preference
def preference_pass_matrix(returns, preference_matrix):
    print(f"returns shape: {returns.shape}")
    print(f"preference_matrix shape: {preference_matrix.shape}")
    print(f"preference_matrix: {preference_matrix}")
    passes = torch.zeros((preference_matrix.shape[0], preference_matrix.shape[1]), dtype=torch.int)
    for i in range(preference_matrix.shape[0]):
        for j in range(preference_matrix.shape[1]):
            if (preference_matrix[i, j] == 1):
                passes[i, j] = torch.sum(returns[i,:] >= returns[j,:])
            else:
                passes[i, j] = torch.sum(returns[i,:] <= returns[j,:])

    #get sum of passes
    return passes
#in the categorical case, we simply sample many times and see if the preference is consistent with the hypothesis
def preference_pass_matrix_categorical(returns, preferences):
    pass


def main(  # noqa: C901
    *,
    env_name: Literal["1d", "2d"],
    batch_size: int,
    lr: float,
    num_iterations: int,
    out_dir: str,
    reward_model_kwargs: Dict[str, Any] = {},
    flips: float,
):
    reward_fn: Callable[[torch.Tensor], torch.Tensor]
    state_dim: int

    if env_name == "1d":
        reward_fn = reward_fn_1d
        state_dim = 1
    elif env_name == "1d_identity":
        reward_fn = reward_fn_1d_identity
        state_dim = 1
    elif env_name == "2d":
        reward_fn = reward_fn_2d
        state_dim = 2
    elif env_name == "cycles":
        reward_fn = reward_fn_cycles
        state_dim = 1
    elif env_name == "default":
        reward_fn = reward_fn_default
        state_dim = 1
    else:
        raise ValueError(f"Unknown environment name: {env_name}")

    sample_state = lambda n: sample_state_fn(n)
    print(f"state_dim: {state_dim}")
    print(f"state_shape: {sample_state(1).shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reward_models: Dict[str, BaseRewardModel] = {}
    reward_model_class: Type[BaseRewardModel]
    for reward_model_class in [
        BaseRewardModel,
        #MeanAndVarianceRewardModel,
        #CategoricalRewardModel,
        #ClassifierRewardModel,
    ]:
        print(f"Training {reward_model_class.__name__}...")
        kwargs = dict(reward_model_kwargs)
        # if env_name == "1d" and reward_model_class is MeanAndVarianceRewardModel:
        #     kwargs["max_std"] = 1
        reward_model = reward_model_class(state_dim=state_dim, **kwargs)
        reward_model = train_rlhf(
            reward_model=reward_model,
            reward_fn=reward_fn,
            sample_state=sample_state,
            batch_size=batch_size,
            lr=lr,
            num_iterations=num_iterations,
            device=device,
            flip_percentage = flips
        )
        reward_model.eval()
        reward_models[reward_model_class.__name__] = reward_model

    #Bayesian REX
    reward_model_kwargs["num_layers"] = 2
    b_rex_model = BaseRewardModel(state_dim = state_dim, **(dict(reward_model_kwargs)))
    b_rex_model, b_rex_variances, brex_mean, brex_all = train_brex(
        reward_model=b_rex_model,
        reward_fn=reward_fn,
        sample_state=sample_state,
        batch_size=batch_size,
        lr=lr,
        num_iterations=num_iterations,
        device=device,
        flip_percentage=flips
    )
    b_rex_model.eval()
    reward_models["B_REX"] = b_rex_model

    #print(f"brex variances: {b_rex_variances}")
    #print(f"brex variances shape: {b_rex_variances.shape}")
   # print(f"brex mean: {brex_mean}")
    #print(f"brex mean shape: {brex_mean.shape}")

    #Lexicase
    lex_model = BaseRewardModel(state_dim = state_dim, **(dict(reward_model_kwargs)))
    lex_model, lex_variances, lex_mean, lex_all = train_lex(
        reward_model=lex_model,
        reward_fn=reward_fn,
        sample_state=sample_state,
        batch_size=batch_size,
        lr=lr,
        num_iterations=num_iterations,
        device=device,
        flip_percentage = flips
    )
    lex_model.eval()
    reward_models["Lexicase"] = lex_model

    #print(f"lex variances: {lex_variances}")
    #print(f"lex variances shape: {lex_variances.shape}")
    #print(f"lex mean: {lex_mean}")
    #print(f"lex mean shape: {lex_mean.shape}")


    lex_mcmc_model = BaseRewardModel(state_dim = state_dim, **(dict(reward_model_kwargs)))
    lex_mcmc_model, lex_mcmc_variances, lex_mcmc_mean, lex_mcmc_all = train_brex(
        reward_model=lex_mcmc_model,
        reward_fn=reward_fn,
        sample_state=sample_state,
        batch_size=batch_size,
        lr=lr,
        num_iterations=num_iterations,
        device=device,
        likelihood = "lexicase",
        flip_percentage=flips,
    )

    lex_mcmc_model.eval()
    reward_models["Lexicase MCMC"] = lex_mcmc_model
    
    #-=============================
    
    experiment_dir = os.path.join(
        out_dir,
        env_name,
        f"{batch_size}_{lr}_{num_iterations}",
    )
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Saving results to {experiment_dir}...")

    # Matplotlib configuration
    latex_preamble = r"""
    \usepackage{times}
    \usepackage{amsmath}
    \usepackage{mathptmx}
    \usepackage{xcolor}

    \newcommand{\oracle}[1]{{O_{#1}}}
    \newcommand{\utility}{u}
    \newcommand{\exutility}{\bar{\utility}}
    \newcommand{\learnedutility}{\hat{\utility}}
    \newcommand{\loss}{L}
    \newcommand{\noise}{\epsilon}
    \newcommand{\noisedist}{\mathcal{D}_\noise}
    \newcommand{\bordacount}{\text{BC}}
    \newcommand{\altspace}{\mathcal{A}}
    \newcommand{\alta}{a}
    \newcommand{\altb}{b}
    \newcommand{\altc}{c}
    \newcommand{\unseenspace}{\mathcal{Z}}
    \newcommand{\unseen}{z}
    \newcommand{\unseendist}{\mathcal{D}_\unseen}
    \newcommand{\comparisonprob}{p}
    \newcommand{\btlprob}{\comparisonprob^\text{BTL}}
    \newcommand{\uniformdist}{\text{Unif}}
    \newcommand{\bernoulli}{\mathcal{B}}
    \newcommand{\learneddist}{\smash{\hat{\mathcal{D}}}}
    """
    matplotlib.rc("text", usetex=True)
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rc("text.latex", preamble=latex_preamble)
    matplotlib.rc("font", size=9)
    matplotlib.rc("pgf", rcfonts=False, texsystem="pdflatex", preamble=latex_preamble)
    figure = plt.figure(figsize=(8.5, 4.2))
    cmap = "Greys"
    color = "k"
    title_kwargs = dict(fontsize=9)

    if env_name == "1d" or env_name == "1d_identity" or env_name == "cycles" or env_name == "default":
        x = torch.linspace(0, 1, 100, device=device)
        num_rows = 3
        num_cols = 3

        # # Combine first three subplots into one
        # combined_ax = figure.add_subplot(num_rows, num_cols, 1)
        # combined_ax.set_title("Normal preference learning", **title_kwargs)
        # combined_ax.set_xlabel(r"$\alta$")
        # combined_ax.set_xlim(0, 1)

        # # Base Reward Model Output
        # rewards = reward_models["BaseRewardModel"](x[:, None]).detach().cpu().tolist()
        # combined_ax.plot(
        #     x.cpu(),
        #     rewards,
        #     color="k",
        #     label=r"$\learnedutility(\alta)$",
        # )
        # combined_ax.set_ylim(
        #     [
        #         -0.2 * max(rewards) + 1.2 * min(rewards),
        #         1.2 * max(rewards) - 0.2 * min(rewards),
        #     ]
        # )
        # combined_ax.annotate(
        #     r"$\learnedutility(\alta)$",
        #     xy=(x[65].item(), rewards[65]),
        #     textcoords="offset points",
        #     xytext=(-3, 3),
        #     color="k",
        #     ha="right",
        # )

        # combined_ax.set_ylabel(r"$\text{Learned Utility}(\alpha)$", color=color)
        # combined_ax.tick_params(axis='y', labelcolor=color)

        # Creating twin X-axis for Borda Count and Expected Utility
        #twin_ax = combined_ax.twinx()

        # Borda Count
        # bc = torch.empty_like(x)
        # bc[x < 0.8] = (0.1 + x)[x < 0.8]
        # bc[x >= 0.8] = (0.275 + 0.25 * x)[x >= 0.8]
        # twin_ax.plot(x.cpu(), bc.cpu(), color="tab:blue", label=r"$\bordacount(\alta)$")
        # twin_ax.annotate(
        #     r"$\bordacount(\alta)$",
        #     xy=(x[35].item(), bc[35].item()),
        #     textcoords="offset points",
        #     xytext=(-3, 3),
        #     color="tab:blue",
        #     ha="right",
        # )

        # # Expected Utility Function
        # twin_ax.plot(x.cpu(), x.cpu(), color="tab:red", label=r"$\exutility(\alta)$")
        # twin_ax.set_yticks([])
        # twin_ax.annotate(
        #     r"$\exutility(\alta)$",
        #     xy=(x[50].item(), x[50].item()),
        #     textcoords="offset points",
        #     xytext=(3, -5),
        #     color="tab:red",
        #     ha="left",
        # )
        # combined_ax.set_yticks([])

        # twin_ax.set_ylabel(r"$\text{Utility Measures}$", color='C1')
        # twin_ax.tick_params(axis='y', labelcolor='C1')

        # Adding legend
        # lines, labels = combined_ax.get_legend_handles_labels()
        # lines2, labels2 = twin_ax.get_legend_handles_labels()
        # twin_ax.legend(
        #     lines + lines2,
        #     labels + labels2,
        #     loc="upper left",
        #     handlelength=1,
        #     frameon=False,
        #     borderaxespad=0.1,
        # )

        #print(f"brex model: {b_rex_model(x[:, None]).detach().cpu().shape}")

        #BREX Subplot
        brex_ax = figure.add_subplot(num_rows, num_cols, 1)
        brex_ax.set_title("B-REX MAP", **title_kwargs)
        brex_ax.set_xlabel(r"$\alta$")
        brex_ax.set_ylabel(r"$\hat{\mu}(\alta)$")
        brex_ax.set_xlim(0, 1)
        if len(b_rex_model(x[:,None]).shape) > 1:
            brex_ax.plot(x.cpu(), b_rex_model(x[:, None]).detach().cpu()[:, 0], color=color)
        else:
            brex_ax.plot(x.cpu(), b_rex_model(x[:, None]).detach().cpu(), color=color)
        b_rex_std = torch.sqrt(b_rex_variances).detach().cpu()
        #plot std from 0
        #brex_ax.fill_between(x.cpu(), 0, b_rex_std, alpha=0.2, color=color)

        brex_ax = figure.add_subplot(num_rows, num_cols, 2)
        brex_ax.set_title("B-REX All", **title_kwargs)
        brex_ax.set_xlabel(r"$\alta$")
        brex_ax.set_ylabel(r"$\hat{\mu}(\alta)$")
        brex_ax.set_xlim(0, 1)
        brex_ax.plot(x.cpu(), brex_all.detach().cpu(), color=color, alpha=0.005)

        #plot mean of B-REX
        brex_ax = figure.add_subplot(num_rows, num_cols, 3)
        brex_ax.set_title("B-REX Mean", **title_kwargs)
        brex_ax.set_xlabel(r"$\alta$")
        brex_ax.set_ylabel(r"$\hat{\mu}(\alta)$")
        brex_ax.set_xlim(0, 1)
        brex_ax.plot(x.cpu(), brex_mean.detach().cpu(), color="tab:red", label=r"$\exutility(\alta)$")


        # brex_ax = figure.add_subplot(num_rows, num_cols, 8)
        # brex_ax.set_title("B-REX Samples", **title_kwargs)
        # brex_ax.set_xlabel(r"$\alta$")
        # brex_ax.set_ylabel(r"$\hat{\mu}(\alta)$")
        # brex_ax.set_xlim(0, 1)
        # print(f"brex all shape: {brex_all.shape}")
        # #randomly sample 10 models
        # brex_samples = brex_all[:,np.random.choice(brex_all.shape[1], 10, replace=False)]
        # brex_ax.plot(x.cpu(), brex_samples.detach().cpu(), color=color, alpha=0.5)


        #plot std
        #brex_ax.fill_between(x.cpu(), b_rex_model(x[:, None]).detach().cpu() - b_rex_std, b_rex_model(x[:, None]).detach().cpu() + b_rex_std, alpha=0.2, color=color)
        #brex mean
        #brex_ax.plot(x.cpu(), brex_mean.detach().cpu(), color="tab:red", label=r"$\exutility(\alta)$")
        #brex_ax.fill_between(x.cpu(), brex_mean.detach().cpu() - b_rex_std, brex_mean.detach().cpu() + b_rex_std, alpha=0.2, color="tab:red")
        #plot all brex_models
        

        #lexicase Subplot
        lex_ax = figure.add_subplot(num_rows, num_cols, 4)
        lex_ax.set_title("Lexicase Best", **title_kwargs)
        lex_ax.set_xlabel(r"$\alta$")
        lex_ax.set_ylabel(r"$\hat{\mu}(\alta)$")
        lex_ax.set_xlim(0, 1)

        print(f"x shape: {x.shape}")
        print(f"x type: {x.dtype}")
        #convert lex model weights to float
        lex_model.to(torch.float32)
        print(f"lex model: {lex_model(x[:, None]).detach().cpu().shape}")

        if len(lex_model(x[:,None]).shape) > 1:
            lex_ax.plot(x.cpu(), lex_model(x[:, None]).detach().cpu()[:, 0], color=color)
        else:
            lex_ax.plot(x.cpu(), lex_model(x[:, None]).detach().cpu(), color=color)
        #lex_ax.plot(x.cpu(), lex_model(x[:, None]).detach().cpu()[:, 0], color=color)
        #lex_std = torch.sqrt(lex_variances).detach().cpu()
        #plot std from 0    

       # lex_ax.fill_between(x.cpu(), 0, lex_std, alpha=0.2, color=color)
        
        #std
        #lex_ax.fill_between(x.cpu(), lex_model(x[:, None]).detach().cpu() - lex_std, lex_model(x[:, None]).detach().cpu() + lex_std, alpha=0.2, color=color)

        #lex all
        lex_ax = figure.add_subplot(num_rows, num_cols, 5)
        lex_ax.set_title("Lexicase All", **title_kwargs)
        lex_ax.set_xlabel(r"$\alta$")
        lex_ax.set_ylabel(r"$\hat{\mu}(\alta)$")
        lex_ax.set_xlim(0, 1)
        lex_ax.plot(x.cpu(), lex_all.detach().cpu(), color=color, alpha=0.5)

        # lex_ax = figure.add_subplot(num_rows, num_cols, 9)
        # lex_ax.set_title("Lexicase Samples", **title_kwargs)
        # lex_ax.set_xlabel(r"$\alta$")
        # lex_ax.set_ylabel(r"$\hat{\mu}(\alta)$")
        # lex_ax.set_xlim(0, 1)
        # #randomly sample 10 models
        # lex_samples = lex_all[:,np.random.choice(lex_all.shape[1], 10, replace=False)]
        # lex_ax.plot(x.cpu(), lex_samples.detach().cpu(), color=color, alpha=0.5)

        #lexicase mean
        lex_ax = figure.add_subplot(num_rows, num_cols, 6)
        lex_ax.set_title("Lexicase Mean", **title_kwargs)
        lex_ax.set_xlabel(r"$\alta$")
        lex_ax.set_ylabel(r"$\hat{\mu}(\alta)$")
        lex_ax.set_xlim(0, 1)
        lex_ax.plot(x.cpu(), lex_mean.detach().cpu(), color="tab:red", label=r"$\exutility(\alta)$")
        #lex_ax.fill_between(x.cpu(), lex_mean.detach().cpu() - lex_std, lex_mean.detach().cpu() + lex_std, alpha=0.2, color="tab:red")

        #lex_ax.plot(x.cpu(), lex_mean.detach().cpu(), color="tab:red", label=r"$\exutility(\alta)$")
        #lex_ax.fill_between(x.cpu(), lex_mean.detach().cpu() - lex_std, lex_mean.detach().cpu() + lex_std, alpha=0.2, color="tab:red")
        
        # Mean and Variance Subplot
        # mean_and_variance_ax = figure.add_subplot(num_rows, num_cols, 2)
        # mean_and_variance_ax.set_title("DPL (mean and variance)", **title_kwargs)
        # mean_and_variance_ax.set_xlabel(r"$\alta$")
        # mean_and_variance_ax.set_ylabel(r"$\hat{\mu}(\alta) \pm \hat{\sigma}(\alta)$")
        # mean_and_variance_ax.set_xlim(0, 1)
        # mean_and_log_std = (
        #     reward_models["MeanAndVarianceRewardModel"](x[:, None]).detach().cpu()
        # )
        # mean = mean_and_log_std[:, 0]
        # std = torch.exp(mean_and_log_std[:, 1])
        # mean_and_variance_ax.plot(x.cpu(), mean, color=color)
        # ymin = mean.min() - 1
        # ymax = mean.max() + 1
        # mean_and_variance_ax.fill_between(
        #     x.cpu(),
        #     np.clip(mean - std, ymin, ymax),
        #     np.clip(mean + std, ymin, ymax),
        #     alpha=0.2,
        #     color=color,
        # )
        # mean_and_variance_ax.set_ylim(mean.min() - 1, mean.max() + 1)

        # # Categorical Subplot
        # categorical_ax = figure.add_subplot(num_rows, num_cols, 3)
        # categorical_ax.set_title("DPL (categorical)", **title_kwargs)
        # categorical_ax.set_xlabel(r"$\alta$")
        # categorical_ax.set_ylabel(r"$\hat{p}(\utility \mid \alta)$")
        # categorical_ax.set_xlim(0, 1)
        # dist = reward_models["CategoricalRewardModel"](x[:, None]).detach().cpu()
        # categorical_ax.imshow(
        #     dist.transpose(0, 1),
        #     origin="lower",
        #     extent=(0, 1, 0, 1),
        #     aspect="auto",
        #     cmap=cmap,
        # )

        #brex w/ lexicase likelihood
        lexmcmc_ax = figure.add_subplot(num_rows, num_cols, 7)
        lexmcmc_ax.set_title("Lexicase MCMC MAP", **title_kwargs)
        lexmcmc_ax.set_xlabel(r"$\alta$")
        lexmcmc_ax.set_ylabel(r"$\hat{\mu}(\alta)$")
        lexmcmc_ax.set_xlim(0, 1)
        if len(lex_mcmc_model(x[:,None]).shape) > 1:
            lexmcmc_ax.plot(x.cpu(), lex_mcmc_model(x[:, None]).detach().cpu()[:, 0], color=color)
        else:
            lexmcmc_ax.plot(x.cpu(), lex_mcmc_model(x[:, None]).detach().cpu(), color=color)
        lex_mcmc_std = torch.sqrt(lex_mcmc_variances).detach().cpu()
        #plot std from 0
        #brex_ax.fill_between(x.cpu(), 0, lex_mcmc_std, alpha=0.2, color=color)

        lexmcmc_ax = figure.add_subplot(num_rows, num_cols, 8)
        lexmcmc_ax.set_title("Lexicase MCMC All", **title_kwargs)
        lexmcmc_ax.set_xlabel(r"$\alta$")
        lexmcmc_ax.set_ylabel(r"$\hat{\mu}(\alta)$")
        lexmcmc_ax.set_xlim(0, 1)
        lexmcmc_ax.plot(x.cpu(), lex_mcmc_all.detach().cpu(), color=color, alpha=0.005)

        #plot mean of B-REX
        lexmcmc_ax = figure.add_subplot(num_rows, num_cols, 9)
        lexmcmc_ax.set_title("Lexicase MCMC Mean", **title_kwargs)
        lexmcmc_ax.set_xlabel(r"$\alta$")
        lexmcmc_ax.set_ylabel(r"$\hat{\mu}(\alta)$")
        lexmcmc_ax.set_xlim(0, 1)
        lexmcmc_ax.plot(x.cpu(), lex_mcmc_mean.detach().cpu(), color="tab:red", label=r"$\exutility(\alta)$")


        figure.tight_layout(pad=0.1)
        figure.subplots_adjust(wspace=0.7, left=0.15, right=0.95)
    elif env_name == "2d":
        x = torch.linspace(0, 1, 20, device=device)
        y = torch.linspace(0, 1, 20, device=device)
        xx, yy = torch.meshgrid(x, y)
        xxyy = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        num_rows = 2
        num_cols = 3

        def setup_ax(ax):
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        true_reward_ax = figure.add_subplot(num_rows, num_cols, 1)
        true_reward_ax.set_title(
            "True utility function\n$\\utility(x, y)$", **title_kwargs
        )
        setup_ax(true_reward_ax)
        true_reward_ax.contourf(xx.cpu(), yy.cpu(), yy.cpu(), cmap=cmap)

        borda_count_ax = figure.add_subplot(num_rows, num_cols, 2)
        borda_count_ax.set_title("Borda count\n$\\bordacount(x, y)$", **title_kwargs)
        setup_ax(borda_count_ax)
        bc = (
            0.25
            + yy
            - 0.25 * yy**2
            - 0.125 * xx * yy
            - xx * yy**2
            + 0.25 * xx * yy**3
        )
        borda_count_ax.contourf(xx.cpu(), yy.cpu(), bc.cpu(), cmap=cmap)

        base_ax = figure.add_subplot(num_rows, num_cols, 3)
        base_ax.set_title(
            "Preference learning\n$\\learnedutility(x, y)$", **title_kwargs
        )
        setup_ax(base_ax)
        uhat = reward_models["BaseRewardModel"](xxyy).detach().cpu().reshape(xx.size())
        base_ax.contourf(xx.cpu(), yy.cpu(), uhat, cmap="Blues")

        mean_ax = figure.add_subplot(num_rows, num_cols, 4)
        var_ax = figure.add_subplot(num_rows, num_cols, 5)
        mean_ax.set_title("\n$\\learnedutility(x, y)$", **title_kwargs)
        var_ax.set_title("\n$\\hat{\\sigma}(x, y)$", **title_kwargs)
        setup_ax(mean_ax)
        setup_ax(var_ax)
        mean_and_log_std = (
            reward_models["MeanAndVarianceRewardModel"](xxyy)
            .detach()
            .cpu()
            .reshape(xx.size() + (2,))
        )
        mean = mean_and_log_std[..., 0]
        std = torch.exp(mean_and_log_std[..., 1])
        mean_ax.contourf(xx.cpu(), yy.cpu(), mean, cmap=cmap)
        var_ax.contourf(xx.cpu(), yy.cpu(), std, cmap=cmap)
        mean_and_var_ax = figure.add_subplot(
            GridSpec(num_rows, num_cols - 1, width_ratios=[2, 1])[2], frameon=False
        )
        mean_and_var_ax.set_title(
            "Preference learning (mean and variance)\n", **title_kwargs
        )
        mean_and_var_ax.xaxis.set_visible(False)
        mean_and_var_ax.yaxis.set_visible(False)

        categorical_ax = figure.add_subplot(num_rows, num_cols, 6)
        categorical_ax.set_title("Preference learning (categorical)", **title_kwargs)
        setup_ax(categorical_ax)
        bins = 4
        x = (torch.arange(bins, device=device).float() + 0.5) / bins
        y = (torch.arange(bins, device=device).float() + 0.5) / bins
        xx, yy = torch.meshgrid(x, y)
        xxyy = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        dists = (
            reward_models["CategoricalRewardModel"](xxyy)
            .detach()
            .cpu()
            .reshape(*xx.size(), -1)
        )
        for i in range(1, bins):
            categorical_ax.axvline(i / bins, color="black", linewidth=1)
            categorical_ax.axhline(i / bins, color="black", linewidth=1)
        for i in range(bins):
            for j in range(bins):
                dist = dists[i, j]
                num_atoms = dist.size()[0]
                categorical_ax.bar(
                    np.linspace(i / bins, (i + 1) / bins, num_atoms + 3)[1:-2],
                    dist / dists.max() / bins,
                    width=1 / bins / num_atoms,
                    bottom=j / bins,
                    align="edge",
                    color=color,
                )

        figure.tight_layout(pad=0.1)
        figure.subplots_adjust(hspace=1, wspace=0.5)

    figure.savefig(os.path.join(experiment_dir, f"{env_name}_{flips}.png"), dpi=300)
    figure.savefig(os.path.join(experiment_dir, f"{env_name}_{flips}.pgf"), dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="1d")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--flips", type=float, default=0)
    args = parser.parse_args()

    main(
        env_name=args.env,
        batch_size=args.batch_size,
        lr=args.lr,
        num_iterations=args.num_iterations,
        out_dir=args.out_dir,
        flips=args.flips
    )
