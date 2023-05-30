import torch
import random
import matplotlib.pyplot as plt
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, out_size, input_size, hidden_size, num_layers, seq_length):
        super(Net, self).__init__()
        self.out_size = out_size #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, 32), #fully connected 1
            nn.ReLU(),
            nn.Linear(32, out_size) #fully connected last layer
        )
    
    def forward(self,x):
        h_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.net(hn)
        return out



STEPS = 10_000
BATCH_SIZE = 5
WINDOW_SIZE = 5
DOMAIN_LOW = 0
DOMAIN_TOP = 30
DOMAIN = range(DOMAIN_LOW, DOMAIN_TOP)
DOMAIN_ACCESS = range(DOMAIN_LOW, DOMAIN_TOP - WINDOW_SIZE)


f = lambda x: x

net = Net(1, 1, 2, 1, 10)
optim = torch.optim.Adam(net.parameters())
loss_fn = nn.MSELoss()

# training
for i in range(STEPS):

    input_batch = [
        [[j] for j in range(i, i+WINDOW_SIZE)] for i in random.sample(DOMAIN_ACCESS, BATCH_SIZE)
    ]

    correct_output_batch = [
        [[f(j)] for j in range(i, i+WINDOW_SIZE)] for i in random.sample(DOMAIN_ACCESS, BATCH_SIZE)
    ]

    input_batch = torch.tensor(input_batch).float()
    correct_output_batch = torch.tensor(correct_output_batch).float()

    output_batch: torch.Tensor = net(
        input_batch
    )

    optim.zero_grad()
    loss = loss_fn.forward(output_batch, correct_output_batch)
    loss.backward()
    optim.step()

fig, ax = plt.subplots()

ls = [i for i in DOMAIN]
ax.plot(ls, [f(i) for i in ls], label="Function")
ax.plot(ls, [net(torch.tensor([[j] for j in ls[:(i+1)]]).unsqueeze(0).float()).squeeze().item() for i in ls], label="Approx")
ax.legend()
ax.set_title('F x Approx')


plt.show()
print()