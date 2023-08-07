from agents.rdpg import RDPGActor
from agents.utils import extract_np_array_from_queue
import rsoccer_gym
import gym
import torch
from collections import deque
import numpy as np

WIN_SIZE = 1

env = gym.make('VSS-v0')

obs = env.reset()

# print(env.action_space.shape)
model = RDPGActor(obs.shape[0], env.action_space.shape[0], 42*5, torch.device("cpu"))
model.load_state_dict(torch.load('checkpoint.pth', map_location=torch.device('cpu'))['pi_state_dict'])

queue = deque(maxlen=WIN_SIZE)

queue.append(
    np.concatenate((np.array([-1.0, -1.0]), obs))
)

inp = np.expand_dims(
    extract_np_array_from_queue(queue, WIN_SIZE),
    axis=0
)

inp = torch.FloatTensor(inp)

traced_script_module = torch.jit.trace(model, inp)

torch.jit.save(traced_script_module, 'net.pt')
