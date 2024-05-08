"""
Neuron class using micrograd.py
"""
import random
import microEngine as me
from dacPlot import draw_dot

class Module:
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
        
    def parameters(self):
        return []

class Neuron(Module):
    ''' Neuron for microEngine.py neural network '''
    
    def __init__(self, nInputs, nonlin='ReLU'):
        self.w = [me.Value(random.uniform(-1,1)) for _ in range(0,nInputs)]
        #self.w = [me.Value(random.gauss(0, 1/8)) for _ in range(0,nInputs)]
        #self.b = me.Value(random.uniform(-1,1))
        self.b = me.Value(0)
        self.nonlin = nonlin     

    def __call__(self, x):
        raw_act = sum((w_i*x_i for w_i, x_i in zip(self.w, x)), 0) + self.b
        if self.nonlin == 'tanh':
            return raw_act.tanh()
        elif self.nonlin == 'ReLU':
            return raw_act.relu()
        elif self.nonlin == 'sigmoid':
            return raw_act.sigmoid()
        else: # linear
            return raw_act
    
    def __repr__(self):
        return f"{self.nonlin}-Neuron({len(self.w)})"
    
    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    ''' Creates fully connected layer '''
    
    def __init__(self, nInputs, nLayer, nonlin='ReLU'):
        self.neurons = [Neuron(nInputs, nonlin) for _ in range(nLayer)]
        
    def __call__(self, x):        
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def __repr__(self):
        return f"Layer of [{', '.join(repr(n) for n in self.neurons)}]"
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

        
class Multilayer_Perceptron(Module):
    
    def __init__(self, nIns, nOuts, nonlin='ReLU'):
        sz = [nIns] + nOuts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=nonlin) for i in range(len(nOuts))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

''' 
Sample fully connected network consisting of:
- Input layer: 3 nodes
- Hidden layer: 4 nodes
- Hidden layer: 4 nodes
- Output layer: 1 node
'''
if __name__ == "__main__":
    
    ###########################################################################    
    ######################### EXAMPLE 1: BASIC USE ############################
    ########################################################################### 
    '''
    x = [2.0, 3.0, -1.0]
    n = Multilayer_Perceptron(3, [4,4,1])
    print(n(x))
    n(x).backward()
    draw_dot(n(x))
    '''
    ###########################################################################    
    ##################### EXAMPLE 2: BINARY CLASSIFIER ########################
    ########################################################################### 
    
    n = Multilayer_Perceptron(3, [4,4,1], 'tanh')     
    # Inputs
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    
    # Desired targets (of each input)
    ys = [1.0, -1.0, -1.0, 1.0] 
    
    # Play around with number of iterations and update learning rate 'lRate'
    # to explore convergence properties of the neural network
    lRate = 0.2
    numIter = 200     
    for k in range(numIter):
        
        # Forward pass with MSE for loss
        yPred = [n(x) for x in xs]
        loss = sum([(yOut - yTrue)**2 for yTrue, yOut in zip(ys, yPred)])
    
        # Backward pass
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()
    
        # Update neurons
        for p in n.parameters():
            p.data += -lRate * p.grad
            
        print(k, loss.data)
    print('-----------')
    print('Final Values:')
    print('-----------')
    print(yPred)
    