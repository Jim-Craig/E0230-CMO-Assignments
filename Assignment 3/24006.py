import numpy as np
import matplotlib.pyplot as plt

"""# CMO Assignment -3

## 1 Systems of Linear Equations

#### 3. Use the KKT conditions to solve ConvProb, and arrive at an expression for x∗,and show the intermediate steps. Write code to evaluate the expression and report x∗.
"""

A = np.array([[2,-4,2,-14],[-1,2,-2,11],[-1,2,-1,7]])
b = np.array([10,-6,-5])

AAT = A @ A.T

AAT.shape

EV, EVectors = np.linalg.eig(AAT)

"""AAT has a 0 eigen value, hence invertible. Thus we need to use Psuedo inverse to ccompute the expression$$x = A^T(AA^T)^{-1}b$$"""

pinv_AAT = np.linalg.pinv(AAT)

x = A.T @ pinv_AAT @ b

np.linalg.norm(x)

np.allclose(np.zeros(3),(A @ x - b))

"""So the value of X calculated from the formula actually solves the constraint provided

#### Use the derived projection operator and implement projected gradient descent to solve ConvProb.<br> The equation for the projection operation on a z $\in \rm I\!R^4$ is given by $$P_c(z) = z[I - A^T(AA^T)^{-1}A] -A^T(AA^T)^{-1}b$$
We'd like to implement the gradient descent into using this equation to respect the constratints enforced. This our iteration step would be:<br> for a feasible point $x^k$ , $$x^{k+1} =P_c(x^k + \alpha \nabla f(x^k))$$
"""

constantTerm1 = A.T @ pinv_AAT @ A
constantTerm2 = (A.T @ pinv_AAT @ b).reshape((4,1))
I = np.eye(4,4)
x = x.reshape((4,1))

def projectPoint(x):
  projectedPoint  = (I - constantTerm1) @ x + constantTerm2
  return projectedPoint

def projectedGradientDescent(alpha = 0.1,initialX = np.ones((4,1)), maxIteration = 1000):
  gradFx = initialX
  XUpdated = projectPoint(initialX - alpha * gradFx)
  tolerance = 1e-10
  iteration = 0
  XUpdatedInitial = initialX
  norm = np.linalg.norm(XUpdated - x)
  normDiff = []
  normDiff.append(norm)
  while( np.linalg.norm(XUpdated - XUpdatedInitial) > tolerance and iteration < maxIteration):
    XUpdatedInitial = XUpdated
    gradFx = XUpdatedInitial
    XUpdated = projectPoint(XUpdatedInitial - alpha * gradFx)
    norm = np.linalg.norm(XUpdated - x)
    normDiff.append(norm)
    # print(f"Iteration {iteration}, Norm of update: {np.linalg.norm(XUpdated - XUpdatedInitial)}")
    iteration += 1
  print(f"Iteration {iteration}, Norm of update: {np.linalg.norm(XUpdated - XUpdatedInitial)}")
  return XUpdated, normDiff

i = 0.1
normDiffList =[]
alphaList = []
while(i < 1):
  XUpdated, normDiff = projectedGradientDescent(alpha = i, initialX = np.ones((4,1)))
  normDiffList.append(normDiff)
  alphaList.append(i)
  i += 0.05

np.allclose(XUpdated,x)

alphaList[3]

import matplotlib.pyplot as plt
for normDiff in normDiffList:
  txt = "Alpha is {xpoint}".format(xpoint = round(alphaList[normDiffList.index(normDiff)], 2))
  plt.plot(normDiff, label=txt)
  plt.legend(loc='upper right')

plt.show()

projectPoint(np.zeros((4,1)))

"""## 2 Support Vector Machines"""

data = np.loadtxt("/content/sample_data/Data.csv", delimiter=",")
labels = np.loadtxt("/content/sample_data/Labels.csv", delimiter=",")
labels = labels.reshape((len(labels),1))

"""#### We have the data in the form of $(x_i, y_i)$ where the $x \in \rm I\!R^2 $ and the $y \in \{-1,1\}$. The objective is to find a function $$f(x) = \text{sign}(w^Tx + b)$$. <br> We need to minimse the given function $$\text{f(x)} = \frac{1}{2}||w||^2_2$$
Additoinally we're given the following constraint:
$$y_i(w^Tx_i + b) \ge 1 \forall i\in\{1,2...,n\}$$
"""

data.shape, labels.shape

import cvxpy as cp
w = cp.Variable((2,1))
b = cp.Variable()
lambd = cp.Parameter(nonneg=True)
objective  = 0.5 * cp.norm2(w)**2
# lagrangian = 0.5 * cp.norm(w, 'fro')**2 - cp.sum(lambd * (labels.T * (data @ w + b) - 1))
constraint1 = [labels[i] * (w.T @ data[i] + b) >= 1 for i in range(len(labels))]
prob = cp.Problem(cp.Minimize(objective), constraint1)

prob.solve()

W_calculated = w.value
b_calcualted = b.value

for i in range(len(labels)):
  Y_calculated = W_calculated.T @ data[i] + b_calcualted
  print("label value is : {} and the calculated value is {}".format(labels[i], Y_calculated))

0.5 * np.linalg.norm(W_calculated)**2

W_calculated

"""###### 2.3 Value of the Lambdas"""

sum1 = 0
sum2 = 0
for i in range(len(labels)):
  if(labels[i] == 1):
    sum1 += constraint1[i].dual_value
  if(labels[i] == -1):
    sum2 += constraint1[i].dual_value
  print(constraint1[i].dual_value)

sum1, sum2

for i in range(len(labels)):
  print(constraint1[i].dual_value)

"""4.Write a problem to solve the dual problem"""

# Define parameters
N, d = 10, 2  # Example: N data points, d features
X = data  # N x d feature matrix
y = labels  # N labels, either -1 or 1

# Compute the matrix A where A_ij = y_i * y_j * x_i^T * x_j
A = np.zeros((N, N))  # N x N matrix
for i in range(N):
    for j in range(N):
        A[i, j] = y[i] * y[j] * np.dot(X[i], X[j]) + 1e-3

np.linalg.eig(A)[0]

def projectPoint(lambda_):
    return np.maximum(lambda_, 0)

# Define the dual objective function: g(lambda) = 1^T * lambda - 1/2 * lambda^T * A * lambda
def objectionFunction(lambda_):
  obj = np.sum(lambda_) - 0.5 * lambda_.T @ A @ lambda_
  return -obj

def gradObjectiveFunction(lambda_):
  grad = np.expand_dims(np.ones(len(lambda_)),1) - A @ lambda_
  return -grad

def gradientProjectorOperator(lambda_):
  numerator = lambda_.T @ labels
  denominator = labels.T @ labels
  projectedLambda = lambda_ - (numerator / denominator) * labels
  projectedLambda = projectPoint(projectedLambda)
  return projectedLambda

def projectedGradientDescent(initialX, alpha=0.01) :
  gradFx = gradObjectiveFunction(initialX)
  XUpdated = gradientProjectorOperator(initialX - alpha * gradFx)
  tolerance = 1e-10
  iteration = 0
  gradFx = gradObjectiveFunction(XUpdated)
  normGradFx = np.linalg.norm(gradFx)
  maxIteration = 1000
  while( normGradFx > tolerance and iteration < maxIteration):
    XUpdatedInitial = XUpdated
    gradFx = gradientProjectorOperator(XUpdatedInitial)
    XUpdated = projectPoint(XUpdatedInitial - alpha * gradFx)
    gradFx = gradObjectiveFunction(XUpdated)
    normGradFx = np.linalg.norm(gradFx)
    # print(f"Iteration {iteration}, Norm of update: {np.linalg.norm(XUpdated - XUpdatedInitial)}")
    iteration += 1
  print(f"Iteration {iteration}, Norm of update: {normGradFx}")
  return XUpdated

lambda_ = np.expand_dims(np.ones(len(labels)),1)
finalLambda = projectedGradientDescent(lambda_, 0.01)

finalLambda

