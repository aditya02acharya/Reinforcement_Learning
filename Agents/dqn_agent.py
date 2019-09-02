import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from .memory.replay_buffer import ReplayBuffer

from torch.utils.tensorboard import SummaryWriter

class DQN(object):

    Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

    def __init__(self, 
                 env=None, 
                 network_policy=None, 
                 network_target=None,
                 memory_size=10000, 
                 discount_rate=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=10**5,
                 learning_rate=0.01,
                 target_perf=100,
                 replay_warmup=1000,
                 target_update=100,
                 callback_dict=None):
        '''
        A Class for DQN implementation for Reinforcement Learning.

        Arguments:
        ----------
        environment: The external environment for agent to interact with. Class expects a Gym Environment.
        network: Pytorch neural network. Class expects a nn.Module object.
        '''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

        self.env = env
        self.net = network_policy.to(self.device).train()
        self.tgt_net = network_target.to(self.device).eval()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        self.gamma = discount_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_perf = target_perf
        self.target_update = target_update

        self.exp_buffer = ReplayBuffer(memory_size)
        self._reset()
        self.id = self.env.unwrapped.spec.id
        self.replay_warmup = replay_warmup

        self.callbacks = callback_dict
        self.writer = SummaryWriter(comment="-" + self.id)

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0):
        done_reward = None

        # Choose action using epsilon-greedy.
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(self.device)
            q_vals_v = self.net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)

        self.total_reward += reward

        exp = self.Experience(self.state, action, reward, is_done, new_state)

        # Add experience to replay memory.
        self.exp_buffer.add(exp)

        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

    def calc_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        # convert to tensor.
        states_v = torch.tensor(states).to(self.device)
        next_states_v = torch.tensor(next_states).to(self.device)
        actions_v = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device)
        done_mask = torch.ByteTensor(dones).to(self.device)

        
        # Feed-Forward data to get predicted value.
        state_action_values = self.net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

        # Feed-Forward data to get target value
        
        next_state_values = self.tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards_v

        return self.criterion(state_action_values, expected_state_action_values)

    def run(self, batch_size):
        total_rewards = collections.deque([0],maxlen=100)
        frame_idx = 0
        games_done = 0
        ts_frame = 0
        ts = time.time()
        best_mean_reward = None
        
        while True:
            frame_idx += 1
            
            epsilon = max(self.epsilon_end, self.epsilon_start - frame_idx / self.epsilon_decay)
            
            reward = self.play_step(epsilon)
            
            # Steup Stats.
            if reward is not None:
                games_done += 1

                total_rewards.append(reward)

                speed = (frame_idx - ts_frame) / (time.time() - ts)

                ts_frame = frame_idx

                ts = time.time()

                mean_reward = np.mean(total_rewards)

                print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, games_done, mean_reward, epsilon, speed))

                self.writer.add_scalar("epsilon", epsilon, frame_idx)
                self.writer.add_scalar("speed", speed, frame_idx)
                self.writer.add_scalar("reward_100", mean_reward, frame_idx)
                self.writer.add_scalar("reward", reward, frame_idx)

                if best_mean_reward is None or best_mean_reward < mean_reward:
                    torch.save(self.net.state_dict(), self.id + "-best.dat")
                    if best_mean_reward is not None:
                        print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))

                    best_mean_reward = mean_reward

                if mean_reward > self.target_perf:
                    print("Solved in %d frames!" % frame_idx)
                    break
            if len(self.exp_buffer) < self.replay_warmup:
                continue

            if frame_idx % self.target_update == 0:
                self.tgt_net.load_state_dict(self.net.state_dict())
            
            self.optimizer.zero_grad()
            batch = self.exp_buffer.sample(batch_size)
            loss_t = self.calc_loss(batch)
            loss_t.backward()
            self.optimizer.step()

        self.writer.close()
