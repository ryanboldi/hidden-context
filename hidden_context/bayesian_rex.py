import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os, sys

import argparse
# coding: utf-8

# Take as input a compressed pretrained network or run T_REX before hand
# Then run MCMC and save posterior chain
import pickle
import copy
import time
import random

from tqdm import trange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_l2(last_layer):
    linear = last_layer.weight.data

    with torch.no_grad():
        weights = linear.squeeze().cpu().numpy()

    return np.linalg.norm(weights)

def calc_linearized_pairwise_ranking_loss(last_layer, pairwise_prefs, demo_cnts, confidence=1):
    '''use (i,j) indices and precomputed feature counts to do faster pairwise ranking loss'''
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    #print(device)
    #don't need any gradients
    with torch.no_grad():

        #do matrix multiply with last layer of network and the demo_cnts
        linear = last_layer.weight.data

        weights = linear.squeeze() 

        demo_returns = confidence * torch.mv(demo_cnts, weights)

        loss_criterion = nn.CrossEntropyLoss(reduction="sum")

        cum_log_likelihood = 0.0
        outputs = torch.zeros((len(pairwise_prefs),2)) #each row is a new pair of returns
        for p, ppref in enumerate(pairwise_prefs):
            i,j = ppref
            outputs[p,:] = torch.tensor([demo_returns[i], demo_returns[j]])
        labels = torch.ones(len(pairwise_prefs)).long()

        #print(f"number correct: {torch.sum(outputs[:,1] > outputs[:,0]).item()} out of {len(pairwise_prefs)}")

        cum_log_likelihood = -loss_criterion(outputs, labels)

    return cum_log_likelihood



def calc_pref_ranking_loss(last_layer, features0, features1, prefs, confidence=1):
    with torch.no_grad():
        linear = last_layer.weight.data
        weights = linear.squeeze()

        ##print(f"weights: {weights}")
        #print(f"features0: {features0}")
        #print(f"features1: {features1}")

        if len(weights.shape) > 1:
            weights = weights[0]

        returns0 = confidence * torch.mv(features0, weights)
        returns1 = confidence * torch.mv(features1, weights)

        loss_criterion = nn.CrossEntropyLoss(reduction="sum")
        
        cum_log_likelihood = 0.0
        outputs = torch.zeros((len(prefs),2)) #each row is a new pair of returns
        outputs[:, 0] = returns0
        outputs[:, 1] = returns1

        labels = torch.tensor(prefs).long()

        cum_log_likelihood = -loss_criterion(outputs, labels)

    return cum_log_likelihood

def calc_lexicase_pairwise_ranking_loss(prop_layer, cur_layer, features0, features1, prefs, confidence=1):
    """use (i,j) indices and precomputed feature counts to do faster pairwise ranking loss"""
    # device = torch.device("cuda")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    # print(device)
    # don't need any gradients
    all_scores = []
    for last_layer in [prop_layer, cur_layer]:
        linear = last_layer.weight.data  # not using bias
        # print(linear)
        # print(bias)
        weights = (
            linear.squeeze()
        )  # append bias and weights from last fc layer together
        # print('weights',weights)
        # print('demo_cnts', demo_cnts)
        returns0 = weights @ features0.T 
        returns1 = weights @ features1.T

        #removed positivity prior

        outputs = np.array((returns0 > returns1), dtype=np.float32)
        scores = np.array(np.abs(prefs - outputs))
        #swap 0s and 1s
        scores = np.abs(scores - 1)

        all_scores.append(scores)

    cur_scores, prop_scores = all_scores

    # calc lexicase likelihood
    cur_likelihood = torch.tensor(np.sum((cur_scores - prop_scores) == 1))
    prop_likelihood = torch.tensor(np.sum((prop_scores - cur_scores) == 1))

    return cur_likelihood, prop_likelihood, np.sum(prop_scores)


def mcmc_map_search(reward_net, pairwise_prefs, features0, features1, num_steps, step_stdev, confidence, likelihood = "bradley-terry"):
    '''run metropolis hastings MCMC and record weights in chain'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #last_layer = reward_net.last_layer 
    last_layer = torch.nn.Linear(reward_net.last_layer.in_features, 1, bias=False).to(device)

    #copy first dim of last layuer over
    #last_layer.weight.data = reward_net.last_layer.weight.data[0]
    #print(f"last layer: {last_layer}")  

    #random initialization
    with torch.no_grad():
        linear = last_layer.weight.data
        linear.add_(torch.randn(linear.size(), device=device) * step_stdev)

        l2_norm = np.array([compute_l2(last_layer)])

        linear.div_(torch.from_numpy(l2_norm).float().to(device))

    #normalize the weight vector to have unit 2-norm
    #last_layer = euclidean_proj_l1ball(last_layer)

    #import time
    #start_t = time.time()
    #starting_loglik = calc_pairwise_ranking_loss(reward_net, demo_pairs, preference_labels)
    #end_t = time.time()
    #print("slow likelihood", starting_loglik, "time", 1000*(end_t - start_t))
    #start_t = time.time()s
    if likelihood == "bradley-terry":
        starting_loglik = calc_pref_ranking_loss(last_layer, features0, features1, pairwise_prefs, confidence)
    elif likelihood == "lexicase": #start with accept
        starting_loglik = torch.tensor(0)
    else:
        print("likelihood not recognized")
    #starting_loglik = calc_linearized_pairwise_ranking_loss(last_layer, pairwise_prefs, demo_cnts, confidence)
    #end_t = time.time()
    #print("new fast? likelihood", new_starting_loglik, "time", 1000*(end_t - start_t))
    #print(bunnY)

    map_loglik = starting_loglik
    map_reward = copy.deepcopy(last_layer)

    cur_reward = copy.deepcopy(last_layer)
    cur_loglik = starting_loglik

    #print(cur_reward)

    reject_cnt = 0
    accept_cnt = 0
    chain = []
    log_liks = []
    for i in trange(num_steps, desc="MCMC Steps"):
        #take a proposal step
        proposal_reward = copy.deepcopy(cur_reward)

        #add noise to the weights
        with torch.no_grad():
            proposal_reward.weight.add_(torch.randn(proposal_reward.weight.size(), device=device) * step_stdev)
        #with torch.no_grad():
        #    for param in proposal_reward.parameters():
        #        param.add_(torch.randn(param.size(), device=device) * step_stdev)
        #    
            l2_norm = np.array([compute_l2(proposal_reward)])

            for param in proposal_reward.parameters():
                param.div_(torch.from_numpy(l2_norm).float().to(device))
            
        if likelihood == "bradley-terry":
            prop_loglik = calc_pref_ranking_loss(proposal_reward, features0, features1, pairwise_prefs, confidence)
        else:
            cur_loglik, prop_loglik, prop_scores = calc_lexicase_pairwise_ranking_loss(proposal_reward, cur_reward, features0, features1, pairwise_prefs, confidence)
        #print(f"step {i} loglik: {prop_loglik}")

        if (likelihood == "lexicase"):
            if np.random.rand() < (prop_loglik/(cur_loglik + prop_loglik)):
                accept_cnt += 1
                cur_reward = copy.deepcopy(proposal_reward)
                cur_loglik = prop_loglik
            else:
                #reject and stick with cur_reward
                reject_cnt += 1
            
            if prop_loglik > map_loglik:
                map_loglik = prop_loglik
                map_reward = copy.deepcopy(proposal_reward)
                print("step", i)

                print("updating map loglikelihood to ", prop_loglik)
                print("MAP reward so far", map_reward)
        elif prop_loglik > cur_loglik:
            #accept always
            accept_cnt += 1
            cur_reward = copy.deepcopy(proposal_reward)
            cur_loglik = prop_loglik

            #check if this is best so far
            if prop_loglik > map_loglik:
                map_loglik = prop_loglik
                map_reward = copy.deepcopy(proposal_reward)
                print("step", i)

                print("updating map loglikelihood to ", prop_loglik)
                print("MAP reward so far", map_reward)
        else:
            #accept with prob exp(prop_loglik - cur_loglik)
            if np.random.rand() < torch.exp(prop_loglik - cur_loglik).item():
                accept_cnt += 1
                cur_reward = copy.deepcopy(proposal_reward)
                cur_loglik = prop_loglik
            else:
                #reject and stick with cur_reward
                reject_cnt += 1
        chain.append(cur_reward.weight.detach().numpy())
        log_liks.append(cur_loglik)

    print("num rejects", reject_cnt)
    print("num accepts", accept_cnt)
    return map_reward, np.array(chain), np.array(log_liks)

def generate_feature_counts(demos, reward_net):
    feature_cnts = torch.zeros(len(demos), reward_net.fc2.in_features) #no bias
    for i in range(len(demos)):
        traj = np.array(demos[i])
        traj = torch.from_numpy(traj).float().to(device)
        #print(len(trajectory))
        feature_cnts[i,:] = reward_net.state_features(traj).squeeze().float().to(device)
    return feature_cnts.to(device)