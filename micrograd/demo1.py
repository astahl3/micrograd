'''
Simple 'for-fun' demonstration of the micrograd engine and network
    - micrograd/microEngine.py
    - micrograd/net.py
    
Set up for +1 / -1 moon classification using tanh() for neuron activation
layer. However, relu() and sigmoid() are also available for neurons for more
traditional binary classifier structures.
'''
#import math
#import random
import numpy as np
import matplotlib.pyplot as plt
import microEngine as me
import net

def make_moons(n_samples=100, noise=0.1):
    ''' Generate a half moon classification dataset '''
    
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Outer circle
    theta = np.linspace(0, np.pi, n_samples_out)
    xo = np.cos(theta) + np.random.normal(scale=noise, size=n_samples_out)
    yo = np.sin(theta) + np.random.normal(scale=noise, size=n_samples_out)

    # Inner circle
    theta = np.linspace(0, np.pi, n_samples_in)
    xi = 1 - np.cos(theta) + np.random.normal(scale=noise, size=n_samples_in)
    yi = 0.5 - np.sin(theta) + np.random.normal(scale=noise, size=n_samples_in)

    X = np.vstack([(np.append(xo, xi)), (np.append(yo, yi))]).T
    y = np.array([0] * n_samples_out + [1] * n_samples_in)

    return X, y

def loss(X, y, model, batch_size=None):
    ''' Loss function with optional stochastic batching '''
    
    if batch_size is None:
        Xb, yb = X, y  
    else:
        ri = np.random.permutation(len(X))[:batch_size]
        Xb, yb = X[ri], y[ri]

    # Make the inputs Value class objects 
    inputs = [list(map(me.Value, xrow)) for xrow in Xb]
    
    # Forward pass 
    scores = list(map(model, inputs))
    
    # Custom loss:
    # 0 if correct classification, 1 + ypred^2 incorrect classification
    losses = [(1 + score_k**2 if y_k * score_k.data <= 0 else 0) for y_k, score_k in zip(yb, scores)]
    
    # SVM max-margin loss
    #losses = [(1 + -y_k * score_k).tanh() for y_k, score_k in zip(yb, scores)]
    
    # Sigmoid loss
    #losses = [- (y_k * me.Value(math.log(score_k.data + 1e-4)) + (1 - y_k) * me.Value(math.log(1 - score_k.data + 1e-4)))
    #          for y_k, score_k in zip(yb, scores)]
    
    data_loss = sum(losses) / len(losses)  # Average loss over the batch
    
    # L2 regularization
    alpha = 1e-6
    reg_loss = alpha * sum((p * p for p in model.parameters()))
    total_loss = data_loss + reg_loss
    
    # Calculate accuracy as the percentage correctly classified
    accuracy = [(y_k > 0) == (score_k.data > 0.0) 
                for y_k, score_k in zip(yb, scores)]
    
    return total_loss, sum(accuracy) / len(accuracy)


# Generate moon data
X, y = make_moons(n_samples=100, noise=0.1)
y = y * 2 - 1  # Transform y to be -1 or 1 for tanh()

# Visualize the training data
'''
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr')
plt.title('Simple Two-Moons Dataset')
plt.show()
'''
# Build network model
model = net.Multilayer_Perceptron(2, [16,16,16,1], 'tanh') # 2 layer network
print(model)
N = 100
lRate0 = 1.0
for k in range(0,N):
    
    # Forward pass
    total_loss, acc = loss(X, y, model)
    if acc == 1:
        break
    
    # Backward pass
    model.zero_grad()
    total_loss.backward()
    
    # Update learning rate for fine-tuning result
    if acc >= 0.9:
        lRate = lRate0 / 2
    elif acc >= 0.96:
        lRate = lRate0 / 5
    else:
        lRate = lRate0
    #lRate = lRate0 - 0.9*lRate0*k/N
    
    # Update neurons (SGD)
    for p in model.parameters():
        p.data += -lRate * p.grad
    
    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")

# Visualize the boundary layer
h = 0.25
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()]
inputs = [list(map(me.Value, xrow)) for xrow in Xmesh]
scores = list(map(model, inputs))
Z = np.array([s.data > 0 for s in scores])
Z = Z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())