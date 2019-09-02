import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class Cross_Entropy_Agent(object):
    Episode = namedtuple('Episode', field_names=['reward', 'steps'])
    EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

    def __init__(self, environment=None, network=None, *args, **kwargs):
        '''
        A Cross-Entropy method implementation for Reinforcement Learning.

        Arguments:
        ----------
        environment: The external environment for agent to interact with. Class expects a Gym Environment.
        network: Pytorch neural network. Class expects a nn.Module object.
        '''
        super().__init__(*args, **kwargs)
        self.env = environment
        self.net = network
        
        self.objective = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=0.01)
        self.soft_max = nn.Softmax(dim=1)

        self.writer = SummaryWriter(comment="-cartpole")

        self.max_actions = self.env.action_space.n
        self.obs_size = self.env.observation_space.shape[0]

    def run_episode(self):
        '''
        A routine to perform a single episode.

        '''
        # reset the environment before start of episode.
        obs = self.env.reset()

        # total reward for the episode.
        ep_total_reward = 0.0

        # terminal state flag.
        done = False

        # List to capture episode steps.
        episode_steps = []

        while not done:
            obs_v = torch.FloatTensor([obs])

            # perform a feed-forward step to get probaility distribution over action space.
            act_prob_v = self.soft_max(self.net(obs_v))

            act_prob = act_prob_v.data.numpy()[0]

            # clip values to ensure no deterministic distribution.
            act_prob = np.clip(act_prob, 0.01, 0.99)

            action = np.random.choice(len(act_prob), p=act_prob)

            if action >= self.max_actions:
                raise Exception("sampled action number greater than max_action.")

            # perform action.
            next_obs, reward, done, _ = self.env.step(action)
            
            ep_total_reward += reward
            
            episode_steps.append(self.EpisodeStep(observation=obs, action=action))

            obs = next_obs

        return episode_steps, ep_total_reward


    def iterate_batch(self, batch_size):
        '''
        A routine to create a generator for epsiode batches.

        Arguments:
        ----------
        batch_size: max number of episodes to collect before training.
        '''
        batch = []

        while True:

            episode_steps, episode_reward = self.run_episode()

            batch.append(self.Episode(reward=episode_reward, steps=episode_steps))

            if len(batch) == batch_size:
                yield batch
                batch.clear()


    def filter_batch(self, batch, percentile):
        '''
        A routine to filter out episodes that are below set cut-off.
        
        Arguments:
        ----------
        batch: list of episodes.
        percentile: value to calculate reward cut-off.
        '''

        rewards = list(map(lambda s: s.reward, batch))
        
        reward_bound = np.percentile(rewards, percentile)
        
        reward_mean = float(np.mean(rewards))

        train_obs = []
        
        train_act = []
        
        for example in batch:
            if example.reward < reward_bound:
                continue
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))

        train_obs_v = torch.FloatTensor(train_obs)
        train_act_v = torch.LongTensor(train_act)

        return train_obs_v, train_act_v, reward_bound, reward_mean


    def run(self, batch_size, percentile):
        
        for iter_no, batch in enumerate(self.iterate_batch(batch_size)):
            obs_v, acts_v, reward_b, reward_m = self.filter_batch(batch, percentile)
            
            # clear the grads before start of backprop.
            self.optimizer.zero_grad()
            
            # Predict.
            action_scores_v = self.net(obs_v)
            
            # Calculate Loss.
            loss_v = self.objective(action_scores_v, acts_v)

            # Back-Prop.
            loss_v.backward()
            self.optimizer.step()
            
            # Log performance in console.
            print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
            
            # Logs for tensorboard.
            self.writer.add_scalar("loss", loss_v.item(), iter_no)
            self.writer.add_scalar("reward_bound", reward_b, iter_no)
            self.writer.add_scalar("reward_mean", reward_m, iter_no)
            
            if reward_m > 199:
                print("Solved!")
                break
        self.writer.close()






















