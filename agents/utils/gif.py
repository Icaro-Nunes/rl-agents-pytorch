from collections import deque
import torch
import numpy as np
import PIL
import os
from agents.utils.experience import extract_np_array_from_queue

def generate_gif(
    env, 
    filepath, 
    pi, 
    hp, 
    max_episode_steps=1200, 
    resize_to=None, 
    duration=25
):
    """
    Store a gif from the episode frames.

    Parameters
    ----------
    env : gym environment
    filepath : str
    pi : nn.Module
    max_episode_steps : int
    resize_to : tuple of ints, optional
    duration : float, optional
    """
    
    # collect frames
    frames = []
    s = env.reset()
    last_action = np.array([-1.0 for val in range(hp.N_ACTS)])
    history_queue = deque(maxlen=hp.HISTORY_SIZE)
    for t in range(max_episode_steps):
        if hp.AGENT == "rdpg_async":
            history_queue.append(
                np.concatenate((last_action,s))
            )
            s_v = torch.Tensor(
                np.expand_dims(
                    extract_np_array_from_queue(history_queue, hp.HISTORY_SIZE),
                    axis=0
                )
            ).to(hp.DEVICE)
            a = pi.get_action(s_v)
            s_next, r, done, info = env.step(a)
        elif hp.AGENT != "maddpg_async":
            s_v = torch.Tensor(s).to(hp.DEVICE)
            a = pi.get_action(s_v)
            s_next, r, done, info = env.step(a)
        else:
            a = [agent.action(obs) for agent, obs in zip(pi, s)]
            s_next, r, done, info = env.step(a)
        # store frame
        frame = env.render(mode='rgb_array')
        frame = PIL.Image.fromarray(frame)
        frame = frame.convert('P', palette=PIL.Image.ADAPTIVE)
        if resize_to is not None:
            if not (isinstance(resize_to, tuple) and len(resize_to) == 2):
                raise TypeError(
                    "expected a tuple of size 2, resize_to=(w, h)")
            frame = frame.resize(resize_to)

        frames.append(frame)

        if done:
            break

        s = s_next

    # store last frame
    frame = env.render(mode='rgb_array')
    frame = PIL.Image.fromarray(frame)
    frame = frame.convert('P', palette=PIL.Image.ADAPTIVE)
    if resize_to is not None:
        frame = frame.resize(resize_to)
    frames.append(frame)

    # generate gif
    frames[0].save(
        fp=filepath, 
        format='GIF', 
        append_images=frames[1:], 
        save_all=True,
        duration=duration, 
        loop=0
    )
