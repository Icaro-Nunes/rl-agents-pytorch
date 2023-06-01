import torch
import collections
import time
import gym
import copy
import numpy as np
from agents.utils import NStepTracer, OrnsteinUhlenbeckNoise, generate_gif, HyperParameters, ExperienceFirstLast, extract_np_array_from_queue
import os
from dataclasses import dataclass


@dataclass
class RDPGHP(HyperParameters):
    AGENT: str = "rdpg_async"
    HISTORY_SIZE: int = 10
    HIDDEN_STATE_FACTOR: int = 5
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
    hp: RDPGHP
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
            history_queue = collections.deque(maxlen=hp.HISTORY_SIZE)
            last_action = np.array([-1.0 for val in range(hp.N_ACTS)])
            for i in range(hp.MAX_EPISODE_STEPS):
                # Step the environment
                history_queue.append(np.concatenate((last_action,s)))
                s_v = torch.Tensor(
                    np.expand_dims(
                        extract_np_array_from_queue(history_queue, hp.HISTORY_SIZE),
                        axis=0
                    )
                ).to(device)
                a_v = pi(s_v)
                a = np.squeeze(a_v.cpu().numpy(), axis=0)
                a = noise(a)
                last_action = a
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
                    tracer.add(
                        extract_np_array_from_queue(history_queue, hp.HISTORY_SIZE),
                        a,
                        r,
                        done
                    )
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