from engine import Scalar
import random
import math

class Init:

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0.0

    def parameters(self):
        return []

class Neuron(Init):

    def __init__(self, noinp):
        self.w = [Scalar(random.uniform(1, -1)) for _ in range(noinp)]
        self.b = Scalar(random.uniform(1, -1))

    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Init):

    def __init__(self, noinp, noout, **kwargs):
        self.neurons = [Neuron(noinp, **kwargs) for _ in range(noout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Init):
    
    def __init__(self, noinp, noouts):
        size = [noinp] + noouts
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(noouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"