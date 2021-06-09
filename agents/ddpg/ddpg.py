import copy
import os
import time
from dataclasses import dataclass

import gym
import torch
from agents.ddpg.networks import (DDPGActor, DDPGCritic, TargetActor,
                                  TargetCritic)
from agents.utils import (ExperienceFirstLast, HyperParameters, NStepTracer,
                          OrnsteinUhlenbeckNoise, generate_gif)
from torch.nn import functional as F
from torch.optim import Adam


@dataclass
class DDPGHP(HyperParameters):
    AGENT: str = "ddpg_async"
    NOISE_SIGMA_INITIAL: float = None  # Initial action noise sigma
    NOISE_THETA: float = None
    NOISE_SIGMA_DECAY: float = None  # Action noise sigma decay
    NOISE_SIGMA_MIN: float = None
    NOISE_SIGMA_GRAD_STEPS: float = None  # Decay action noise every _ grad steps


def data_func(
    pi,
    device,
    queue_m,
    finish_event_m,
    sigma_m,
    gif_req_m,
    hp
):
    env = gym.make(hp.ENV_NAME)
    if hp.MULTI_AGENT:
        tracer = [NStepTracer(n=hp.REWARD_STEPS, gamma=hp.GAMMA)]*hp.N_AGENTS
    else:
        tracer = NStepTracer(n=hp.REWARD_STEPS, gamma=hp.GAMMA)
    noise = OrnsteinUhlenbeckNoise(
        sigma=sigma_m.value,
        theta=hp.NOISE_THETA,
        min_value=env.action_space.low,
        max_value=env.action_space.high
    )

    with torch.no_grad():
        while not finish_event_m.is_set():
            # Check for generate gif request
            gif_idx = -1
            with gif_req_m.get_lock():
                if gif_req_m.value != -1:
                    gif_idx = gif_req_m.value
                    gif_req_m.value = -1
            if gif_idx != -1:
                path = os.path.join(hp.GIF_PATH, f"{gif_idx:09d}.gif")
                generate_gif(env=env, filepath=path,
                             pi=copy.deepcopy(pi), hp=hp)

            done = False
            s = env.reset()
            noise.reset()
            if hp.MULTI_AGENT:
                [tracer[i].reset() for i in range(hp.N_AGENTS)]
            else:
                tracer.reset()
            noise.sigma = sigma_m.value
            info = {}
            ep_steps = 0
            if hp.MULTI_AGENT:
                ep_rw = [0]*hp.N_AGENTS
            else:
                ep_rw = 0
            st_time = time.perf_counter()
            for i in range(hp.MAX_EPISODE_STEPS):
                # Step the environment
                s_v = torch.Tensor(s).to(device)
                a_v = pi(s_v)
                a = a_v.cpu().numpy()
                a = noise(a)
                s_next, r, done, info = env.step(a)
                ep_steps += 1
                if hp.MULTI_AGENT:
                    for i in range(hp.N_AGENTS):
                        ep_rw[i] += r[f'robot_{i}']
                else:
                    ep_rw += r

                # Trace NStep rewards and add to mp queue
                if hp.MULTI_AGENT:
                    exp = list()
                    for i in range(hp.N_AGENTS):
                        kwargs = {
                            'state': s[i],
                            'action': a[i],
                            'reward': r[f'robot_{i}'],
                            'last_state': s_next[i]
                        }
                        exp.append(ExperienceFirstLast(**kwargs))
                    queue_m.put(exp)
                else:
                    tracer.add(s, a, r, done)
                    while tracer:
                        queue_m.put(tracer.pop())

                if done:
                    break

                # Set state for next step
                s = s_next

            info['fps'] = ep_steps / (time.perf_counter() - st_time)
            info['noise'] = noise.sigma
            info['ep_steps'] = ep_steps
            info['ep_rw'] = ep_rw
            queue_m.put(info)


class DDPG:

    def __init__(self, hp):
        self.device = hp.DEVICE
        # Actor-Critic
        self.pi = DDPGActor(hp.N_OBS, hp.N_ACTS).to(self.device)
        self.Q = DDPGCritic(hp.N_OBS, hp.N_ACTS).to(self.device)
        # Training
        self.tgt_Q = TargetCritic(self.Q)
        self.tgt_pi = TargetActor(self.pi)
        self.pi_opt = Adam(self.pi.parameters(), lr=hp.LEARNING_RATE)
        self.Q_opt = Adam(self.Q.parameters(), lr=hp.LEARNING_RATE)

        self.gamma = hp.GAMMA**hp.REWARD_STEPS

    def get_action(self, observation):
        s_v = torch.Tensor(observation).to(self.device)
        return self.pi.get_action(s_v)

    def share_memory(self):
        self.pi.share_memory()
        self.Q.share_memory()

    def loss(self, batch):
        state_batch = batch.observations
        action_batch = batch.actions
        reward_batch = batch.rewards
        mask_batch = batch.dones.bool()
        next_state_batch = batch.next_observations

        next_state_action = self.tgt_pi(next_state_batch)
        qf_next_target = self.tgt_Q(next_state_batch, next_state_action)
        qf_next_target[mask_batch] = 0.0
        next_q_value = reward_batch + self.gamma * qf_next_target
        qf = self.Q(state_batch, action_batch)
        Q_loss = F.mse_loss(qf, next_q_value.detach())

        pi = self.pi(state_batch)
        pi_loss = self.Q(state_batch, pi)
        pi_loss = -pi_loss.mean()

        return pi_loss, Q_loss

    def update(self, batch):
        pi_loss, Q_loss = self.loss(batch)

        # train critic
        Q_loss = Q_loss
        self.Q_opt.zero_grad()
        Q_loss.backward()
        self.Q_opt.step()

        # train actor - Maximize Q value received over every S
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()

        pi_loss = pi_loss.cpu().detach().numpy()
        Q_loss = Q_loss.cpu().detach().numpy()

        # Sync target networks
        self.tgt_Q.sync(alpha=1 - 1e-3)
        reward_mean = torch.mean(batch.rewards).cpu().numpy()
        return pi_loss, Q_loss, reward_mean
