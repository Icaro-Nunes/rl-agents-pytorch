import torch
import random
import matplotlib.pyplot as plt
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, out_size, input_size, hidden_size, num_layers, seq_length, device):
        super(Net, self).__init__()
        self.out_size = out_size #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True).to(device) #lstm
        
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, 32), #fully connected 1
            nn.ReLU(),
            nn.Linear(32, out_size) #fully connected last layer
        ).to(device)
    
    def forward(self,x):
        h_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state
        c_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.net(hn)
        return out



STEPS = 100_000
BATCH_SIZE = 5
WINDOW_SIZE = 5
DOMAIN_LOW = 0
DOMAIN_TOP = 30
DOMAIN = range(DOMAIN_LOW, DOMAIN_TOP)
DOMAIN_ACCESS = range(DOMAIN_LOW, DOMAIN_TOP - WINDOW_SIZE)


f = lambda x: x

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

net = Net(1, 1, 2, 1, 10, device)
optim = torch.optim.Adam(net.parameters())
loss_fn = nn.MSELoss()

net.to(device)
optim
loss_fn.to(device)

# training
for _ in range(STEPS):

    input_batch = [
        [[j] for j in range(i, i+WINDOW_SIZE)] for i in random.sample(DOMAIN_ACCESS, BATCH_SIZE)
    ]

    correct_output_batch = [
        [ls[WINDOW_SIZE-1][0]+1] for ls in input_batch
    ]

    input_batch = torch.tensor(input_batch).float().to(device)

    correct_output_batch = torch.tensor(correct_output_batch).float().to(device)

    output_batch: torch.Tensor = net(
        input_batch
    )
    output_batch.to(device)

    optim.zero_grad()
    loss = loss_fn.forward(output_batch, correct_output_batch)
    loss.backward()
    optim.step()

fig, ax = plt.subplots()




ls = [i for i in DOMAIN]
ax.plot(ls, [f(i) for i in ls], label="Function")
ax.plot(ls, [net(torch.tensor([[j] for j in range(i - WINDOW_SIZE, i)]).unsqueeze(0).float().to(device)).squeeze().item() for i in ls], label="Approx")
ax.legend()
ax.set_title('F x Approx')


plt.show()
print()