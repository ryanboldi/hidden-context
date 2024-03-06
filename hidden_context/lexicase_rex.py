import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
# coding: utf-8

# Take as input a compressed pretrained network or run T_REX before hand
# Then run MCMC and save posterior chain
import time
import random

from .lexicase import select_from_scores

from tqdm import trange

#numpy version 
def calc_lexicase_scores(last_layer, features0, features1, prefs):
    '''use (i,j) indices and precomputed feature counts to do faster pairwise ranking loss'''

    popsize = last_layer.shape[0]

    #print(f"last layer shape: {last_layer.shape}")
    #print(f"popsize {popsize}")

    weights = last_layer #append bias and weights from last fc layer together

    #print(f"weights shape: {weights.shape}")
    #print(f"demo_cnts shape: {demo_cnts.shape}")

    returns0 = weights @ features0.T 
    returns1 = weights @ features1.T

    #print(f"returns0 shape: {returns0.shape}")

    #print(f"demo_returns shape: {demo_returns.shape}")
        
    #outputs = np.zeros((popsize, len(prefs),2)) #each row is a new pair of returns
    #outputs[:,:,0] = returns0
    #outputs[:,:,1] = returns1

    #print(f"outputs shape: {outputs.shape}")
    #print(f"prefs shape: {prefs.shape}")

    #find where pref is higher in the index of prefs
    #preds = np.argmax(outputs, axis=2)
    #scores = np.zeros((popsize, len(prefs)))

    #scores are where returns0 > returns1
    outputs = np.array((returns0 > returns1), dtype=np.float32)
    scores = np.array(np.abs(prefs - outputs))

    return scores


def lexicase_search(pairwise_prefs, features0, features1, popsize, num_steps, step_stdev, normalize, downsample_level=1, elitism=False, selection="lex", dstype="generation"):
    downsampling = dstype #event for new ds per selection event, generation for generation
    features0 = features0.cpu().detach().numpy()
    features1 = features1.cpu().detach().numpy()

    downsample_level = int(downsample_level)
    size_of_ds = len(pairwise_prefs) // downsample_level

    # create random numbers between -1 and 1 of shape: ((popsize, len(demo_cnts[0]))
    population = np.random.uniform(-1, 1, (popsize, len(features0[0])))
    #print(f"population shape: {population.shape}")

    #normalize the weight vector to have unit 2-norm
    if normalize:
        population = population / np.linalg.norm(population, axis=1)[:, None]

    ds_indices = np.arange(len(pairwise_prefs))
    np.random.shuffle(ds_indices)
    ds_indices = ds_indices[:size_of_ds]
    
    scores = calc_lexicase_scores(population, features0[ds_indices], features1[ds_indices], pairwise_prefs[ds_indices])

    #sum_scores = np.sum(scores, axis=1)

    #print(f"score of perfect individual: {sum_scores[0]}")

    #print(f"sum scores shape {sum_scores.shape}")
    #print(f"sum scores: {sum_scores}")

    #best_scores = [np.max(np.sum(scores, axis=1))]
    #print(f"shape of scores: {scores.shape}")

    for i in trange(num_steps, desc="Lex Steps"):
        if downsampling == "event":
            selected = []
            for i in range(len(scores)):
                # downsample
                scores = calc_lexicase_scores(population, features0, features1, pairwise_prefs)
                selected.append(select_from_scores(scores, selection=selection, elitism=False, epsilon=False, num_to_select=1))
            selected = np.array(selected)
        else:
            #take a proposal step
            selected = select_from_scores(scores, selection=selection, elitism=elitism, epsilon=False)


        new_pop = population[selected[int(elitism):]]

        # mutate pop with gaussian noise
        #pick indices to mutate the individuals for each individual
        indices = np.random.randint(0, new_pop.shape[1], size = new_pop.shape[0])

        #mask out everything but the index
        masked_pop = np.zeros_like(new_pop)

        for i in range(len(masked_pop)):
            masked_pop[i, indices[i]] = 1
            
        #print(f"First in new pop: {new_pop[0]}")
        new_pop = new_pop + np.random.normal(0, step_stdev, new_pop.shape) * masked_pop
        #print(f"First in new pop after: {new_pop[0]}")

        if elitism:
            new_pop = np.concatenate((np.expand_dims(population[selected[0]], axis=0), new_pop), axis=0) #elitism

        population = new_pop
        ds_indices = np.arange(len(pairwise_prefs))
        np.random.shuffle(ds_indices)
        ds_indices = ds_indices[:size_of_ds]

        scores = calc_lexicase_scores(population, features0[ds_indices], features1[ds_indices], pairwise_prefs[ds_indices])

        #best_scores.append(np.max(np.sum(scores, axis=1)))

        #project
        if normalize:
            population = population / np.linalg.norm(population, axis=1)[:, None]

    #one more selection
    scores = calc_lexicase_scores(population, features0, features1, pairwise_prefs)
    selected = select_from_scores(scores, selection=selection, elitism=elitism, epsilon=False)
    population = population[selected[int(elitism):]]

    return population, scores, None #, best_scores

def select_one(population, features0, features1, pairwise_prefs):
    scores = calc_lexicase_scores(population, features0, features1, pairwise_prefs)
    selected = select_from_scores(scores, selection="lex", elitism=False, epsilon=False)
    return population[selected[0]], scores[selected[0]]

def generate_feature_counts_from_model(demos, reward_net, n, device):
    feature_cnts = torch.zeros(len(demos), n) #no bias
    for i in range(len(demos)):
        traj = np.array(demos[i])
        #print(traj)
        traj = torch.from_numpy(traj).float().to(device)
        #print(len(trajectory))
        feature_cnts[i,:] = reward_net(traj)

    return feature_cnts.to(device)

def generate_feature_counts_from_state(demos, n, device):
    feature_cnts = torch.zeros(len(demos), n) #no bias
    
    for i in range(len(demos)):
        traj = np.array(demos[i])
        #print(traj)
        traj = torch.from_numpy(traj).float().to(device)
        #print(len(trajectory))
        feature_cnts[i] = torch.sum(traj, axis=0)

    print(f"feautre counts: {feature_cnts}")
    print(f"feature counts shape: {feature_cnts.shape}")
    return feature_cnts.to(device)