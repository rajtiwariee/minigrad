# import random
# from minigrad.engine import Value


# class Module:

#     def zero_grad(self):
#         for p in self.parameters():
#             p.grad = 0

#     def parameters(self):
#         return []
# class Neuron(Module):
#     def __init__(self,nin,nonlin = True):
#         self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
#         self.bias = Value(random.uniform(-1, 1))
#         self.nonlin = nonlin
#     def __call__(self,x):
#         out = sum(w * x_i for w, x_i in zip(self.weights, x)) + self.bias
#         return out.relu() if self.nonlin else out
    
#     def parameters(self):
#         return self.weights + [self.bias]
    
#     def __repr__(self):
#         return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.weights)})"

    
# class Layer(Module): #single linear layer
#     def __init__(self, nin,nout, **kwargs):
#         self.neurons = [Neuron(nin,**kwargs) for _ in range(nout)]
        
#     def __call__(self, x):
#         out = [n(x) for n in self.neurons]
#         return out[0] if len(out) == 1 else out

#     def parameters(self):
#         return [p for n in self.neurons for p in n.parameters()]

#     def __repr__(self):
#         return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
# class MLP(Module):
#     def __init__(self,nin,nout,nhidden=1,**kwargs):
#         sz = [nin] + nout if isinstance(nout, list) else nout
#         self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(sz)-1)]
        
#     def __call__(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
    
#     def parameters(self):
#         return [p for layer in self.layers for p in layer.parameters()]
    
#     def __repr__(self):
#         return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
        

# if __name__ == "__main__":
#     # Example usage
#     # neuron = Neuron(3, 1)
#     # layer = Layer(3, 2, nonlin=True)
#     # mlp = MLP(3, [4,2,1], nonlin=True)
#     # x = [Value(0.5), Value(-0.2), Value(0.1)]
#     # # output = neuron(x)
#     # # output = layer(x)
#     # output = mlp(x)
#     # print("Output:", output)
    
#     # # Print parameters
#     # for param in mlp.parameters():
#     #     print("Parameter:", param.data)
    
    
#     #Lets train a model
#     import random
#     random.seed(42)  # for reproducibility
#     xs = [
#     [2.0, 3.0, 1.0],
#     [3.0, 1.0, 0.5],
#     [0.5, 1.0, 3.0],
#     [3.0, 1.0, 0.05],
#     ]
#     xs = [[Value(v) for v in x] for x in xs]
#     ys = [1.0, 0, 1.0, 1.0] # desired targets
#     model = MLP(3, [4, 2, 1], nonlin=True)
    
#     epoch = 3
        
#     for steps in range(epoch):
        
#         for x, ygt in zip(xs, ys):
#             #zero the gradients
#             model.zero_grad()
#             # forward pass
#             ypred = model(x)
#             loss = (ypred - ygt) ** 2
#             # backward pass
#             loss.backward()
#             # update parameters
#             for p in model.parameters():
#                 p.data += -0.1 * p.grad

            
#         print(f"Epoch {steps+1}/{epoch}, Loss: {loss.data},")
            
    
#     print(model(xs[1]))





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

