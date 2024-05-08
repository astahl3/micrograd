'''
For visualization purposes: create a graph network of nodes and operations:
    def trace(root):
    def draw_dot(root):
        
Intended to use with micrograd engine
'''
from graphviz import Digraph
from IPython.display import display, SVG # for optional display in plot pane

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
        dot.node(name=uid, label='{ %s | data %.4f | grad %.4f }' % 
                 (n.label, n.data, n.grad), shape='record')
        
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