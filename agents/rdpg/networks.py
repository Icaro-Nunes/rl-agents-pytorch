import torch
import torch.nn as nn
import copy
import numpy as np

class RDPGActor(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size, device):
        super(RDPGActor, self).__init__()

        self.device = device
        self.num_layers_lstm = 2
        self.hidden_size = hidden_size
        self.input_size = act_size+obs_size

        self.lstm = nn.LSTM(
            input_size=act_size+obs_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers_lstm,
            batch_first=True
        )

        self.net = nn.Sequential(
            nn.Linear(hidden_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        h_0 = torch.autograd.Variable(torch.zeros(self.num_layers_lstm, x.size(0), self.hidden_size)).to(self.device) #hidden state
        c_0 = torch.autograd.Variable(torch.zeros(self.num_layers_lstm, x.size(0), self.hidden_size)).to(self.device) #internal state

        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.net(hn)
        return out
    
    def get_action(self, x):
        return np.squeeze(self.forward(x).detach().cpu().numpy(), axis=0)


class RDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size, device):
        super(RDPGCritic, self).__init__()

        self.device = device
        self.num_layers_lstm = 2
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=act_size+obs_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers_lstm,
            batch_first=True
        )

        self.obs_net = nn.Sequential(
            nn.Linear(hidden_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x, a):
        h_0 = torch.autograd.Variable(torch.zeros(self.num_layers_lstm, x.size(0), self.hidden_size)).to(self.device) #hidden state
        c_0 = torch.autograd.Variable(torch.zeros(self.num_layers_lstm, x.size(0), self.hidden_size)).to(self.device) #internal state

        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next

        obs = self.obs_net(hn)
        return self.out_net(torch.cat([obs, a], dim=1))


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """

    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


class TargetActor(TargetNet):
    def __call__(self, S):
        return self.target_model(S)


class TargetCritic(TargetNet):
    def __call__(self, S, A):
        return self.target_model(S, A)
