import torch

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

scripted_gate = torch.jit.script(MyDecisionGate())

class MyRNNLoop(torch.nn.Module):
    def __init__(self):
        super(MyRNNLoop, self).__init__()
        x, h = torch.rand(3, 4), torch.rand(3, 4)
        self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))

    def forward(self, xs):
        h, y = torch.zeros(3, 4), torch.zeros(3, 4)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y, h

rnn_loop = torch.jit.script(MyRNNLoop())
print("=== scripted_gate ===")
print(scripted_gate.code)
print("=== rnn_loop.cell ===")
print(rnn_loop.cell.code)
print("=== rnn_loop ===")
print(rnn_loop.code)

class WrapRNN(torch.nn.Module):
    def __init__(self):
        super(WrapRNN, self).__init__()
        self.loop = torch.jit.script(MyRNNLoop())

    def forward(self, xs):
        y, h = self.loop(xs)
        return torch.relu(y)

warprnn = WrapRNN()
traced = torch.jit.trace(warprnn, (torch.rand(10, 3, 4)))
print("=== WrapRNN ===")
print(traced.code)

traced.save('wrapped_rnn.pt')
loaded = torch.jit.load('wrapped_rnn.pt')

print(loaded)
print(loaded.code)

xs = torch.rand(3, 3, 4)
print(loaded(xs))
print(warprnn(xs))
