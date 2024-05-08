import math
from dacPlot import draw_dot

class Value:
    
    # _children=() sets it as empty tuple
    # _op='' sets it as empty string
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    # this formats the output of operations so that they are human readable
    def __repr__(self):
        return f'Value (data={self.data})'
    
    # note: Python interprets a+b as a.__add__(b)
    def __add__(self, other):
        # if other is not a Value class, assume a number and wrap it in class
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other): # other + self
        '''
        if isinstance(other, Value):
            return self.__add__(other)
        else:
            return self.__add__(Value(other))
        '''
        return self + other
    
    def __mul__(self, other):
        # if other is not a Value class, assume a number and wrap it in class
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other): # other * self
        return self * other
    
    def __neg__(self): # -self
        return self * (-1)
    
    def __sub__(self, other): # self - other
        return self + (-other)
    
    def __pow__(self, other):
        assert isinstance(other, (int,float)), 'only supporting int or float'
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += other*self.data**(other-1) * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other): # self / other
        return self * other**(-1)
    
    def __rtruediv__(self, other):
        return other * self**(-1)
    
    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad})'
    
    # activation functions
    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        n = self.data
        t = (math.exp(2*n)-1) / (math.exp(2*n)+1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad 
        out._backward = _backward
        return out
    
    def sigmoid(self):
        n = self.data
        sig = 1 / (1 + math.exp(-n))
        out = Value(sig, (self,), 'sigmoid')
        def _backward():
            self.grad += sig*(1-sig) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    # backpropagation
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        
if __name__ == "__main__":
    
    ###########################################################################    
    ################## MANUAL EXAMPLE 1: RANDOM NETWORK #######################
    ###########################################################################
    '''
    # example: e = a*b + c --> (a.__mul__(b)).__add__(c)
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e+c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d*f; L.label = 'L'
    '''
    ###########################################################################
    
    
    ###########################################################################    
    ###################### MANUAL EXAMPLE 2: NEURON ###########################
    ###########################################################################
    
    # Forward pass begin
    
    # Values and weights input to neuron
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    
    # Neuron bias
    b = Value(6.8813735870195432, label='b')
    
    # Setup expression: x1*w1 + x2*w2 + b
    x1w1 = x1*w1; x1w1.label='x1*w1'
    x2w2 = x2*w2; x2w2.label='x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label='n'
    
    # Implement activation with tanh
    #o = n.tanh(); o.label='o'
    
    # Implement activation of tanh with exponentials and division
    e = (2*n).exp(); e.label = 'e'
    o = (e-1)/(e+1); o.label = 'o'
        
    # Forward pass end
    
    # Enter draw_dot(Value) to generate a graph of this pass, where Value is
    # the final "root" node (i.e., loss function) in the network 
    draw_dot(o)
    
    # Perform backpropagation: 
    # Start at end of "network" and calculate gradient for child nodes, where
    # gradient is the derivative of the final node (or loss function) with 
    # respect to the value in each node; i.e., a recursive application of the 
    # chain rule moving backwards through the graph 
    
    # Manual backpropagation (node-by-node):
    '''
    o.grad = 1.0 # initialize final output gradient
    o._backward()
    n._backward()
    x1w1x2w2._backward()
    x1w1._backward()
    x2w2._backward()
    draw_dot(o)
    o.backward()
    draw_dot(o)
    '''
    
    # Automatic backpropagation
    o.backward()
    draw_dot(o)

    ###########################################################################
