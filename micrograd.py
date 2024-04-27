import math
import numpy as np
import matplotlib as plt
from graphviz import Digraph
from IPython.display import display, SVG # for optional display in plot pane


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
        return self + other    
        '''
        if isinstance(other, Value):
            return self.__add__(other)
        else:
            return self.__add__(Value(other))
        '''
    
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
    
    def __neg__(self, other): # -self
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
    
    # activation function
    def tanh(self):
        n = self.data
        t = (math.exp(2*n)-1) / (math.exp(2*n)+1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad 
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
        
'''
For visualization purposes: create a graph network of nodes and operations:
    def trace(root):
    def draw_dot(root):
'''
def trace(root):
    # build set of all nodes and edges in graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir':'LR'})
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        
        # create rectangular 'record' node for any Value in graph
        dot.node(name=uid, label='{ %s | data %.4f | grad %.4f }' % (n.label, n.data, n.grad), shape='record')
        
        # if node resulted from an operation, create operation '_op' node
        # then connect operation node to current node
        if n._op:
            dot.node(name=(uid + n._op), label=n._op)
            dot.edge(uid+n._op, uid)
    
    # connect n1 to the operation node of n2
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)    
    
    # for optional display in Spyder IDE plot pane
    # use IPython's display function to render the SVG
    display(SVG(dot.pipe(format='svg')))
    
    return dot

# MANUAL EXAMPLE 1: RANDOM NETWORK ############################################
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
###############################################################################



# MANUAL EXAMPLE 2: NEURON ####################################################

# values and weights input to neuron
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# neuron bias
b = Value(6.8813735870195432, label='b')

# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label='x1*w1'
x2w2 = x2*w2; x2w2.label='x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label='n'

# implement activation with tanh
#o = n.tanh(); o.label='o'

# implement activation with exponentials and division
e = (2*n).exp(); e.label = 'e'
o = (e-1)/(e+1); o.label = 'o'
o.backward()
draw_dot(o)
###############################################################################

'''
This is the end of the forward pass
Enter draw_dot(L) to generate a graph of this pass
Now we will being backpropagation (below)

Backpropagation: 
Start at end and calculate a gradient at all preceding value nodes
Gradient is the derivative of L with respect to the variable in each node
I.e., a recursive application of the chain rule moving backward through graph 
'''

# MANUAL BACKPROPAGATION EXAMPLE 2: NEURON ####################################
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
###############################################################################

'''
topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)
build_topo(o)
print(topo)
'''


