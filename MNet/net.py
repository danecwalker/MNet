import torch.nn as nn

class MNet(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(MNet, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    y = self.l1(x)
    y = self.relu(y)
    y = self.l2(y)
    return y