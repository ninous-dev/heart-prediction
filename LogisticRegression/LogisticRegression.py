from torch import nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, final_activation=None):
        super(LogisticRegression, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.final_activation = final_activation

    def forward(self, x):
        out = self.linear(x)

        if self.final_activation != None:
            out = self.final_activation(out)

        return out
