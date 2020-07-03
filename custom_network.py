import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import cv2
import utils
import hydra


# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)


def split_obs(obs):
    if len(obs.shape) == 4:
        img = obs[:, :3, :, :]
        robot_state_shape = int(obs[0, 3, 0, 0].cpu().numpy())
        robot_state = obs[:, 3, 0, 1: robot_state_shape + 1]
    elif len(obs.shape) == 3:
        img = obs[:3, :, :]
        robot_state_shape = int(obs[3, 0, 0].cpu().numpy())
        robot_state = obs[3, 0, 1: robot_state_shape + 1]
    else:
        raise ValueError
    return img, robot_state


# def nature_cnn(init_, input_shape, output_size, activation):
#     return nn.Sequential(
#             init_(nn.Conv2d(input_shape, 32, 8, stride=4)),
#             activation(),
#             init_(nn.Conv2d(32, 64, 4, stride=2)),
#             activation(),
#             init_(nn.Conv2d(64, 32, 3, stride=1)),
#             activation(),
#             Flatten(),
#             init_(nn.Linear(32 * 7 * 7, output_size)),
#             activation()
#     )


class EncoderDictObs(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        # self.num_layers = 4
        # self.num_filters = 32
        # self.output_dim = 35
        # self.output_logits = False
        # self.feature_dim = feature_dim
        #
        # self.convs = nn.ModuleList([
        #     nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
        #     nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
        #     nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
        #     nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
        # ])
        #
        # self.head = nn.Sequential(
        #     nn.Linear(self.num_filters * 35 * 35, self.feature_dim),
        #     nn.LayerNorm(self.feature_dim))

        self.num_layers = 3
        self.num_filters = 32
        self.output_dim = 7
        self.output_logits = False
        self.feature_dim = feature_dim

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[0] - 1, self.num_filters, 8, stride=4),
            nn.Conv2d(self.num_filters, self.num_filters * 2, 4, stride=2),
            nn.Conv2d(self.num_filters * 2, self.num_filters, 3, stride=1)
        ])

        self.head = nn.Sequential(
            nn.Linear(self.num_filters * self.output_dim * self.output_dim, self.feature_dim))

        self.outputs = dict()

    def forward_conv(self, obs):
        # obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv
        conv = conv.contiguous()
        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.head(h)
        if not self.output_logits:
            out = torch.tanh(out)

        self.outputs['out'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)
            if len(v.shape) > 2:
                logger.log_image(f'train_encoder/{k}_img', v[0], step)

        for i in range(self.num_layers):
            logger.log_param(f'train_encoder/conv{i + 1}', self.convs[i], step)


class ActorDictObs(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, encoder_cfg, action_shape, state_input_shape, hidden_dim, hidden_depth, cat_fc_size,
                 log_std_bounds):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder_cfg)

        self.log_std_bounds = log_std_bounds

        self.state_input = utils.mlp(state_input_shape, hidden_dim, hidden_dim, 1, nn.Tanh(), nn.Tanh())

        self.trunk = utils.mlp(self.encoder.feature_dim + hidden_dim, cat_fc_size,
                               2 * action_shape[0], 1, activation=nn.Tanh())

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, detach_encoder=False):
        img, robot_state = split_obs(obs)

        conv_features = self.encoder(img, detach=detach_encoder)
        robot_state_out = self.state_input(robot_state)
        h1 = torch.cat((conv_features, robot_state_out), 1)
        mu, log_std = self.trunk(h1).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.state_input):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor_state/fc{i}', m, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor_trunk/fc{i}', m, step)


class CriticDictObs(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, encoder_cfg, action_shape, state_input_shape, hidden_dim, hidden_depth, cat_fc_size):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder_cfg)

        self.Q1_state = utils.mlp(state_input_shape + action_shape[0], hidden_dim, hidden_dim, 1, nn.Tanh(), nn.Tanh())
        self.Q2_state = utils.mlp(state_input_shape + action_shape[0], hidden_dim, hidden_dim, 1, nn.Tanh(), nn.Tanh())

        self.Q1 = utils.mlp(self.encoder.feature_dim + hidden_dim,
                            cat_fc_size, 1, 1, activation=nn.Tanh())
        self.Q2 = utils.mlp(self.encoder.feature_dim + hidden_dim,
                            cat_fc_size, 1, 1, activation=nn.Tanh())

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action, detach_encoder=False):
        assert obs.size(0) == action.size(0)
        img, robot_state = split_obs(obs)
        conv_features = self.encoder(img, detach=detach_encoder)
        q1_state = self.Q1_state(torch.cat([robot_state, action], dim=-1))
        q2_state = self.Q1_state(torch.cat([robot_state, action], dim=-1))
        q1 = self.Q1(torch.cat([conv_features, q1_state], dim=-1))
        q2 = self.Q2(torch.cat([conv_features, q2_state], dim=-1))

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        self.encoder.log(logger, step)

        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1_state) == len(self.Q2_state)
        for i, (m1, m2) in enumerate(zip(self.Q1_state, self.Q2_state)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc1{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc1{i}', m2, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc2{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc2{i}', m2, step)