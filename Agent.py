from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


class Replaybatch:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, log_prob, state_value, reward, entropy):
        item = (log_prob, state_value, reward, entropy)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        if len(self.buffer) == 0:
            raise ValueError("Replay batch is empty.")

        batch_size = min(batch_size, len(self.buffer))

        if batch_size == 1:
            batch = [self.buffer[-1]]
        else:
            candidate_count = max(len(self.buffer) - 1, 1)
            priorities = np.arange(1, candidate_count + 1, dtype=float)
            probabilities = priorities / priorities.sum()

            random_indices = np.random.choice(
                candidate_count,
                size=batch_size - 1,
                replace=candidate_count < (batch_size - 1),
                p=probabilities,
            )

            batch = [self.buffer[i] for i in random_indices]
            batch.insert(0, self.buffer[-1])

        log_probs, state_values, rewards, entropies = zip(*batch)

        log_probs = [item for sublist in log_probs for item in sublist]
        state_values = [item for item in state_values]
        rewards = torch.tensor(rewards, dtype=torch.float32)
        entropies = [item for item in entropies]

        return log_probs, state_values, rewards, entropies

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config.lstm_actor_num_layers
        self.hidden_size = config.lstm_actor_hidden_size
        self.lr = config.lr
        self.weight_decay = config.weight_decay

        self.n_inputs = config.state_size
        self.num_players = config.num_mobile
        self.num_actions = config.action_size
        self.num_mobile_location = config.num_mobile_location
        self.seq = config.seq

        self.lr1_actor_hidden_size = config.lr1_actor_hidden_size
        self.lr2_actor_hidden_size = config.lr2_actor_hidden_size
        self.lr3_actor_hidden_size = config.lr3_actor_hidden_size

        self.use_lstm = config.use_lstm
        self.bidirectional = config.bidirectional
        self.use_agent_as_batch = config.use_agent_as_batch
        self.add_drop_out = config.add_drop_out

        if self.use_lstm:
            if self.bidirectional:
                if self.use_agent_as_batch:
                    lstm_input_size = self.n_inputs
                else:
                    lstm_input_size = self.n_inputs * self.num_players

                self.lstm = nn.LSTM(
                    lstm_input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                    bidirectional=True,
                )
                self.affine = nn.Linear(self.hidden_size * 2, self.lr1_actor_hidden_size)
            else:
                if self.use_agent_as_batch:
                    lstm_input_size = self.n_inputs
                else:
                    lstm_input_size = self.n_inputs * self.num_players

                self.lstm = nn.LSTM(
                    lstm_input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                    bidirectional=False,
                )
                self.affine = nn.Linear(self.hidden_size, self.lr1_actor_hidden_size)
        else:
            if self.use_agent_as_batch:
                self.affine = nn.Linear(self.seq * self.n_inputs, self.lr1_actor_hidden_size)
            else:
                self.affine = nn.Linear(
                    self.seq * self.n_inputs * self.num_players,
                    self.lr1_actor_hidden_size,
                )

        self.affine2 = nn.Linear(self.lr1_actor_hidden_size, self.lr2_actor_hidden_size)

        if self.use_agent_as_batch:
            self.action_layer = nn.Linear(self.lr2_actor_hidden_size, self.num_actions)
        else:
            self.action_layer = nn.Linear(
                self.lr2_actor_hidden_size,
                self.num_actions * self.num_players,
            )

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, states_batch, batch=False):
        if not batch:
            if self.use_lstm:
                if self.use_agent_as_batch:
                    seq_len, n_agents, _ = states_batch.shape
                    state = torch.from_numpy(states_batch).float()
                    state = state.permute(1, 0, 2)
                else:
                    seq_len, n_agents, _ = states_batch.shape
                    states_batch = states_batch.reshape((states_batch.shape[0], -1))
                    state = torch.from_numpy(states_batch).float()
                    state = state.unsqueeze(0)
            else:
                if self.use_agent_as_batch:
                    seq_len, n_agents, _ = states_batch.shape
                    state = torch.from_numpy(states_batch).float()
                    state = state.permute(1, 0, 2)
                    state = state.reshape((state.shape[0], -1))
                    self.affine = nn.Linear(seq_len * self.n_inputs, self.lr1_actor_hidden_size)
                else:
                    seq_len, n_agents, _ = states_batch.shape
                    states_batch = states_batch.flatten()
                    state = torch.from_numpy(states_batch).float()
                    self.affine = nn.Linear(
                        seq_len * self.n_inputs * self.num_players,
                        self.lr1_actor_hidden_size,
                    )
        else:
            if self.use_lstm:
                if self.use_agent_as_batch:
                    bs, seq_len, n_agents, n_feature = states_batch.shape
                    state = states_batch.permute(0, 2, 1, 3)
                    state = state.reshape(bs * n_agents, seq_len, n_feature)
                else:
                    bs, seq_len, n_agents, n_feature = states_batch.shape
                    state = states_batch.reshape(bs, seq_len, n_agents * n_feature)
            else:
                if self.use_agent_as_batch:
                    bs, seq_len, n_agents, n_feature = states_batch.shape
                    state = states_batch.permute(0, 2, 1, 3)
                    state = state.reshape(bs * n_agents, seq_len * n_feature)
                    self.affine = nn.Linear(seq_len * self.n_inputs, self.lr1_actor_hidden_size)
                else:
                    bs, seq_len, n_agents, n_feature = states_batch.shape
                    state = states_batch.reshape(bs, seq_len * n_agents * n_feature)
                    self.affine = nn.Linear(
                        seq_len * self.n_inputs * self.num_players,
                        self.lr1_actor_hidden_size,
                    )

        batch_size = state.shape[0]
        state = Variable(state)

        if self.use_lstm:
            num_directions = 2 if self.bidirectional else 1
            h_0 = Variable(torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size))
            output, (hn, cn) = self.lstm(state, (h_0, c_0))
            x = output[:, -1, :]

            if self.add_drop_out:
                x = self.dropout(self.relu(self.affine(x)))
                x = self.dropout(self.relu(self.affine2(x)))
            else:
                x = self.relu(self.affine(x))
                x = self.relu(self.affine2(x))
        else:
            if self.add_drop_out:
                x = self.dropout(self.relu(self.affine(state)))
                x = self.dropout(self.relu(self.affine2(x)))
            else:
                x = self.relu(self.affine(state))
                x = self.relu(self.affine2(x))

        y = self.action_layer(x)

        if not batch:
            y = y.reshape(self.num_players, self.num_actions)
        else:
            y = y.reshape(bs, self.num_players, self.num_actions)

        probabilities = F.softmax(y, dim=-1)
        return probabilities


class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config.lstm_critic_num_layers
        self.hidden_size = config.lstm_critic_hidden_size
        self.lr = config.lr
        self.weight_decay = config.weight_decay

        self.n_inputs = config.state_size
        self.num_players = config.num_mobile
        self.num_actions = config.action_size
        self.num_mobile_location = config.num_mobile_location
        self.seq = config.seq

        self.lr1_critic_hidden_size = config.lr1_critic_hidden_size
        self.lr2_critic_hidden_size = config.lr2_critic_hidden_size
        self.lr3_critic_hidden_size = config.lr3_critic_hidden_size

        self.use_lstm = config.use_lstm
        self.bidirectional = config.bidirectional
        self.use_agent_as_batch = config.use_agent_as_batch
        self.add_drop_out = config.add_drop_out

        self.seed = config.seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.backends.cudnn.enabled:
                torch.cuda.manual_seed(self.seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

        if self.use_lstm:
            if self.bidirectional:
                self.lstm = nn.LSTM(
                    self.n_inputs * self.num_players,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                    bidirectional=True,
                )
                self.affine = nn.Linear(self.hidden_size * 2, self.lr1_critic_hidden_size)
            else:
                self.lstm = nn.LSTM(
                    self.n_inputs * self.num_players,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                    bidirectional=False,
                )
                self.affine = nn.Linear(self.hidden_size, self.lr1_critic_hidden_size)
        else:
            self.affine = nn.Linear(
                self.seq * self.n_inputs * self.num_players,
                self.lr1_critic_hidden_size,
            )

        self.affine2 = nn.Linear(self.lr1_critic_hidden_size, self.lr2_critic_hidden_size)
        self.value_layer = nn.Linear(self.lr2_critic_hidden_size, 1)
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, states_batch, batch=False):
        if not batch:
            if self.use_lstm:
                reshaped_states = states_batch.reshape((states_batch.shape[0], -1))
                state = torch.from_numpy(reshaped_states).float()
                state = state.unsqueeze(0)
            else:
                seq_len, n_agents, n_feature = states_batch.shape
                state = torch.from_numpy(states_batch.flatten()).float()
                self.affine = nn.Linear(
                    seq_len * self.n_inputs * self.num_players,
                    self.lr1_critic_hidden_size,
                )
        else:
            if self.use_lstm:
                shape_4d = states_batch.shape
                state = states_batch.reshape(shape_4d[0], shape_4d[1], -1)
            else:
                bs, seq_len, n_agents, n_feature = states_batch.shape
                state = states_batch.reshape(bs, seq_len * n_agents * n_feature)
                self.affine = nn.Linear(
                    seq_len * self.n_inputs * self.num_players,
                    self.lr1_critic_hidden_size,
                )

        batch_size = state.shape[0]
        state = Variable(state)

        if self.use_lstm:
            num_directions = 2 if self.bidirectional else 1
            h_0 = Variable(torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size))
            output, (hn, cn) = self.lstm(state, (h_0, c_0))
            hn = hn.view(-1, self.hidden_size)
            x = self.relu(hn)
        else:
            x = state

        if not batch:
            if self.add_drop_out:
                x = self.dropout(self.relu(self.affine(torch.flatten(x))))
                x = self.dropout(self.relu(self.affine2(torch.flatten(x))))
            else:
                x = self.relu(self.affine(torch.flatten(x)))
                x = self.relu(self.affine2(x))
        else:
            if self.add_drop_out:
                x = self.dropout(self.relu(self.affine(x)))
                x = self.dropout(self.relu(self.affine2(x)))
            else:
                x = self.relu(self.affine(x))
                x = self.relu(self.affine2(x))

        state_value = self.value_layer(x)
        return state_value


class ActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_inputs = config.state_size
        self.num_players = config.num_mobile
        self.num_actions = config.action_size
        self.num_mobile_location = config.num_mobile_location
        self.config = config

        self.seed = config.seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.backends.cudnn.enabled:
                torch.cuda.manual_seed(self.seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

        self.actor = Actor(config)
        self.critic = Critic(config)
        self.replay_batch = Replaybatch(50000)

        # self.temperature = 2
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.entropies = []
        self.advantages = []

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

        self.entropy_coefficient = config.entropy_coefficient

        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, states_batch, batch=False):
        state_value = self.critic(states_batch, batch)
        probabilities = self.actor(states_batch, batch)
        return state_value, probabilities

    def select_action(self, state_value, probabilities, exploration=False, eval=False, deterministic=False):
        entropy = -(probabilities * torch.log(torch.clamp(probabilities, 1e-8, 1.0))).sum(dim=-1).mean()

        probabilities = torch.clamp(probabilities, 0.0001, 0.9999)
        action_probs = probabilities + (probabilities == 0.0).float() * 1e-8

        action_distribution = Categorical(action_probs)

        if exploration and random.random() < self.epsilon:
            action = torch.zeros((self.num_players,), dtype=torch.long)
            for acc in range(len(action)):
                action[acc] = random.randint(0, self.num_mobile_location)
        else:
            if not deterministic:
                action = action_distribution.sample()
            else:
                action = np.argmax(action_probs.detach().cpu().numpy(), axis=-1)
                action = torch.tensor(action, dtype=torch.long)

        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        self.entropies.append(entropy)

        return action

    def batch_push(self):
        self.replay_batch.push(self.logprobs, self.state_values, self.rewards, self.entropies)

    def calculateLoss(self, gamma=0.99):
        rewards = []
        dis_reward = 0.0
        for reward in self.rewards[::-1]:
            dis_reward = float(reward) + gamma * dis_reward
            rewards.insert(0, dis_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        if rewards.numel() > 1 and float(rewards.std()) > 1e-8:
            rewards = (rewards - rewards.mean()) / rewards.std()
        else:
            rewards = rewards - rewards.mean()

        loss = 0.0
        loss2 = 0.0

        for logprob, value, reward, entropy in zip(self.logprobs, self.state_values, rewards, self.entropies):
            value_scalar = value.squeeze()
            reward_tensor = reward.to(value_scalar.dtype)

            advantage = reward_tensor.item() - value_scalar.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value_scalar, reward_tensor)

            loss = loss + action_loss.mean() - self.entropy_coefficient * entropy
            loss2 = loss2 + value_loss

        return loss, loss2

    def calculateLoss_batch(self, batch_size, gamma=0.99):
        logprob_batch, value_batch, rewards_batch, entropies_batch = self.replay_batch.sample(batch_size)

        rewards = []
        dis_reward = 0.0
        for t in reversed(range(len(rewards_batch))):
            dis_reward = rewards_batch[t] + gamma * dis_reward
            rewards.insert(0, dis_reward)

        rewards = torch.stack(rewards).float()
        if rewards.numel() > 1 and float(rewards.std()) > 1e-8:
            rewards = (rewards - rewards.mean()) / rewards.std()
        else:
            rewards = rewards - rewards.mean()

        value_batch = torch.stack([v.squeeze() for v in value_batch]).view(len(rewards), -1)
        entropies_batch = torch.stack(entropies_batch).view(len(rewards), -1)
        logprob_batch = torch.stack(logprob_batch).view(len(rewards), -1, self.num_players)

        loss = 0.0
        loss2 = 0.0

        for logprob, value, reward, entropy in zip(logprob_batch, value_batch, rewards, entropies_batch):
            advantage = reward - value
            action_loss = -(advantage.unsqueeze(1)) * logprob
            value_loss = F.smooth_l1_loss(value, reward.expand_as(value))
            entropy_loss = (self.entropy_coefficient * entropy).mean()

            loss = loss + action_loss.mean() - entropy_loss
            loss2 = loss2 + value_loss.mean()

        return loss, loss2

    def learn(self, loss, loss2):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        loss.backward()
        loss2.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
        self.clearMemory()

    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
        del self.advantages[:]
        del self.entropies[:]

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

