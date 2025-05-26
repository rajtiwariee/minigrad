import random
from minigrad.engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1))
        self.nonlin = nonlin
    
    def __call__(self, x):
        out = sum(w * x_i for w, x_i in zip(self.weights, x)) + self.bias
        return out.relu() if self.nonlin else out
    
    def parameters(self):
        return self.weights + [self.bias]
    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.weights)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
        
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nout, **kwargs):
        # Fixed: properly handle both list and single output
        if isinstance(nout, list):
            sz = [nin] + nout
        else:
            sz = [nin, nout]
        
        # Create layers with proper activation handling
        self.layers = []
        for i in range(len(sz)-1):
            # Last layer typically doesn't use activation for regression
            nonlin = kwargs.get('nonlin', True) and (i < len(sz)-2)
            self.layers.append(Layer(sz[i], sz[i+1], nonlin=nonlin))
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

