import torch
from torch.nn import Module
from torch.nn import Parameter
import pdb
import torch.jit as jit

class GRUCell(Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(hidden_size))
        self.bias_hh = Parameter(torch.randn(3 * hidden_size))

    # @jit.script_method
    def forward(self, input, hidden):
        input = input.squeeze()
        hidden = hidden.squeeze()
        # pdb.set_trace()
        i_n = torch.mm(input, self.weight_ih.t()) + self.bias_ih
        gh = torch.mm(hidden, self.weight_hh.t()) + self.bias_hh
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(h_r)
        inputgate = torch.sigmoid(h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        # pdb.set_trace()
        return None, hy.unsqueeze(0)