from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def predict(X, w):
    z = np.dot(X, w)
    probs = 1 / (1 + np.exp(-z))
    y_dach = ((probs >= 0.5) * 2) - 1
    return  y_dach

iris = datasets.load_iris()
X = iris.data[:100, :2] # extract two features
y = iris.target[:100]
y = (y * 2) - 1 # convert to -1 and 1

np.random.seed(1)
# shuffle data
indices = np.arange(y.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# add column of ones (for intercept)
X = np.c_[X, np.ones(X.shape[0])]

# init
w = np.zeros(X.shape[1])
# w = np.random.random(X.shape[1])
X_rows = X.shape[0]
y_rows = y.shape[0]

import time
t0 = time.time()

# f   = sum(log(exp(-y.*(X*w)) + vector(1)))

# derivative of f with respect to w --> g
# f'  = -X'*(exp(-y.*(X*w)).*y./(vector(1)+exp(-y.*(X*w))))

# derivative of f'' with respect to w --> H
# f'' =  -((diag(exp(-y.*(X*w)).*y.*exp(-y.*(X*w)).*y./((vector(1)+exp(-y.*(X*w))).*(vector(1)+exp(-y.*(X*w)))))*X)'*X-(diag(y.*exp(-y.*(X*w)).*y./(vector(1)+exp(-y.*(X*w))))*X)'*X)

# "simple" newton method
for i in range(3):
    g =  -(X.T).dot(((np.exp(-(y * (X).dot(w))) * y) / (np.ones(y_rows) + np.exp(-(y * (X).dot(w))))))
    H = -(((((((np.exp(-(y * (X).dot(w))) * y) * np.exp(-(y * (X).dot(w)))) * y) / ((np.ones(y_rows) + np.exp(-(y * (X).dot(w)))) * (np.ones(y_rows) + np.exp(-(y * (X).dot(w))))))[:, np.newaxis] * X).T).dot(X) - (((((y * np.exp(-(y * (X).dot(w)))) * y) / (np.ones(y_rows) + np.exp(-(y * (X).dot(w)))))[:, np.newaxis] * X).T).dot(X))
    # w = w - np.linalg.inv(H).dot(g)
    w = w - np.linalg.solve(H, g)

t1 = time.time()
total = t1-t0
print("Seconds: ", "{0:0.2f}".format(total))

predicted = predict(X, w)

print("Accuracy: ", sum(abs(predicted == y)) / X_rows)

# draw plot
color=['red' if label == -1 else 'blue' for label in y]
plt.scatter(X[:,0], X[:,1], color=color)
plt.plot([4.0, 7.0], [(0.0 - w[2] - w[0] * 4.0) / w[1], (0.0 - w[2] - w[0] * 7.0) / w[1]]) # line [x1, x2], [y1, y2]
plt.show()
