from torch import nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        #out = self.final_activation(out)
        return out
