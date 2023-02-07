#!/usr/bin/env python3
from collections import namedtuple
import numpy as np
from catState import CatState
from gameEnv import GameEnv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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


def iterate_batches(env, cat_net, mouse_net, batch_size):
    batch_cat = []
    batch_mouse = []
    episode_reward_cat = 0.0
    episode_reward_mouse = 0.0
    cat_episode_steps = []
    mouse_episode_steps = []
    obs_cat = env.reset_cat()
    obs_mouse = env.reset_mouse()
    sm = nn.Softmax(dim=1)
    while True:
        # On bouge le chat
        obs_v_cat = torch.FloatTensor([obs_cat])
        act_probs_v_cat = sm(cat_net(obs_v_cat))
        act_probs_cat = act_probs_v_cat.data.numpy()[0]
        action_cat = np.random.choice(len(act_probs_cat), p=act_probs_cat)
        next_obs_cat, reward_cat, is_done_cat, _ = env.cat_step(action_cat)
        episode_reward_cat += reward_cat
        cat_step = EpisodeStep(observation=obs_cat, action=action_cat)
        cat_episode_steps.append(cat_step)
        
        # On mouge la souris
        obs_v_mouse = torch.FloatTensor([obs_mouse])
        act_probs_v_mouse = sm(mouse_net(obs_v_mouse))
        act_probs_mouse = act_probs_v_mouse.data.numpy()[0]
        action_mouse = np.random.choice(len(act_probs_mouse), p=act_probs_mouse)
        next_obs_mouse, reward_mouse, is_done_mouse, _ = env.mouse_step(action_mouse)
        episode_reward_mouse += reward_mouse
        mouse_step = EpisodeStep(observation=obs_mouse, action=action_mouse)
        mouse_episode_steps.append(mouse_step)
        
        env.draw()
        
        if is_done_cat or is_done_mouse:
            if episode_reward_cat == 0:
                episode_reward_cat += 100/env.cat.step
            e_cat = Episode(reward=episode_reward_cat, steps=cat_episode_steps)
            e_mouse = Episode(reward=episode_reward_mouse, steps=mouse_episode_steps)
            batch_cat.append(e_cat)
            batch_mouse.append(e_mouse)
            episode_reward_mouse = 0.0
            episode_reward_cat = 0.0
            cat_episode_steps = []
            mouse_episode_steps = []
            next_obs_cat = env.reset_cat()
            next_obs_mouse = env.reset_mouse()
            if len(batch_mouse) == batch_size:
                yield batch_cat, batch_mouse
                batch_cat = []
                batch_mouse = []
        obs_cat = next_obs_cat
        obs_mouse = next_obs_mouse


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
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = GameEnv(5,5, vision = 2)
    
    save_path = "cat_model.pt"
    
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    cat_obs_size = env.cat_observation_space
    cat_n_actions = env.cat_action_space

    cat_net = Net(cat_obs_size, HIDDEN_SIZE, cat_n_actions)
    mouse_net = Net(cat_obs_size, HIDDEN_SIZE, cat_n_actions)
    # save_path = "model.pt"
    # cat_net.load_state_dict(torch.load(save_path))
    
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=cat_net.parameters(), lr=0.01)
    
    objective_mouse = nn.CrossEntropyLoss()
    optimizer_mouse = optim.Adam(params=cat_net.parameters(), lr=0.01)

    for iter_no, batches in enumerate(iterate_batches(env, cat_net, mouse_net, BATCH_SIZE)):
        batch_cat , batch_mouse = batches
        
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch_cat, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = cat_net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss_cat=%.3f, reward_mean_cat=%.1f, rw_bound_cat=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        
        obs_v_mouse, acts_v_mouse, reward_b_mouse, reward_m_mouse = filter_batch(batch_mouse, PERCENTILE)
        optimizer_mouse.zero_grad()
        action_scores_v_mouse = mouse_net(obs_v_mouse)
        loss_v_mouse = objective_mouse(action_scores_v_mouse, acts_v_mouse)
        loss_v_mouse.backward()
        optimizer_mouse.step()
        print("%d: loss_mouse=%.3f, reward_mean_mouse=%.1f, rw_bound_mouse=%.1f" % (
            iter_no, loss_v_mouse.item(), reward_m_mouse, reward_b_mouse))
        torch.save(cat_net.state_dict(), save_path)
        