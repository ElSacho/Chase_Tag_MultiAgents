#!/usr/bin/env python3
from collections import namedtuple
import numpy as np
from catState import CatState
from gameEnv import GameEnv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        x = self.net(x)
        F.softmax(x, dim=1)
        return F.softmax(x, dim=1)
    
class CatAgent(CatState):
    def __init__(self, hidden_size, vision = 2):
        super().__init__(vision = vision)
        self.net = Net(self.observation_space,  hidden_size, self.action_space)
        self.reward = 0
        
    
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    for i in range(env.number_of_cats):
        batch.append([])
    episode_reward = [0.0]*env.number_of_cats
    episode_steps = []
    for i in range(env.number_of_cats):
        episode_steps.append([])
    obs = env.reset_cat()
    sm = nn.Softmax(dim=1)
    while True:
        next_obs = []
        done = []
        for i in range(env.number_of_cats):
            obs_v = torch.FloatTensor([obs[i]])
            act_probs_v = sm(net[i](obs_v))
            act_probs = act_probs_v.data.numpy()[0]
            action = np.random.choice(len(act_probs), p=act_probs)
            # action = np.argmax(act_probs)
            next_single_obs, reward, is_done, _ = env.cat_step(action, i)
            episode_reward[i] += reward
            next_obs.append(next_single_obs)
            done.append(is_done)
            
            step = EpisodeStep(observation=obs[i], action=action)
            episode_steps[i].append(step)
        env.draw()
        if True in done:
            for i in range(env.number_of_cats):
                e = Episode(reward=episode_reward[i], steps=episode_steps[i])
                batch[i].append(e)
                # print(Episode(reward=episode_reward[i], steps=episode_steps[i]))
            episode_reward = [0.0]*env.number_of_cats
            episode_steps = []
            for i in range(env.number_of_cats):
                episode_steps.append([])
            next_obs = env.reset_cat()
            if len(batch[1]) == batch_size:
                yield batch
                batch = []
                for i in range(env.number_of_cats):
                    batch.append([])
        obs = next_obs

def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))
    
    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        # test = np.array(train_obs)
        # print(test.shape)
        # l = math.sqrt(-1)
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = GameEnv(5,5, number_of_cats = 3, vision = 0)
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    cat_obs_size = env.cat_observation_space
    cat_n_actions = env.cat_action_space

    cat_net = [Net(cat_obs_size, HIDDEN_SIZE, cat_n_actions) for i in range(env.number_of_cats)]
    
    save_path = "cat_model.pt"
    # cat_net.load_state_dict(torch.load(save_path))
    
    objective = nn.CrossEntropyLoss()
    optimizer = [optim.Adam(params=cat_net[i].parameters(), lr=0.01) for i in range(env.number_of_cats)]

    for iter_no, batch in enumerate(iterate_batches(env, cat_net, BATCH_SIZE)):
        for i in range(env.number_of_cats):
            obs_v, acts_v, reward_b, reward_m = filter_batch(batch[i], PERCENTILE)
            optimizer[i].zero_grad()
            action_scores_v = cat_net[i](obs_v)
            loss_v = objective(action_scores_v, acts_v)
            loss_v.backward()
            optimizer[i].step()
            print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
                iter_no, loss_v.item(), reward_m, reward_b))
        # Sauvegardez le modèle
        torch.save(cat_net[0].state_dict(), save_path)
        