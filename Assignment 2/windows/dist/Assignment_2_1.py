from oracles import f1, f2, f3
import numpy as np
SR_No= 24006
#Question 1.2
#Find the minimizer for 0.5 x^T .A . x + b^T . x + c

def functionValue_1(x, A, b):
    return  0.5 * np.dot(x.T, np.dot(A, x)) - np.dot(b.T, x)

def gradientFx_1(x, A, b):
    return A @ x -b

def calculateX_i(x_i, u_i, A, b):
    gradFx = gradientFx_1(x_i, A, b)  # Calculate the gradient
    denominator = u_i.T @ A@ u_i#u_i^T*A*u_i
    numerator = -(np.dot(gradFx, u_i))
    alpha = numerator / denominator  # Step size
    return x_i + alpha * u_i  # Update the solution

def calculateU_i(gradFx, u_i, A):
    denominator = u_i.T @ A@ u_i#u_i^T*A*u_i
    numerator = gradFx.T@ A@ u_i

    gamma = numerator/ denominator # g^i * u_i/denominator
    return -gradFx + gamma*u_i 


def conjugateGradient(initialX = np.array([1,1,1,0,1])) :
    results = f1(SR_No, True)
    A = np.array(results[0])
    b = np.array(results[1]).flatten()
    gradfx = gradientFx_1(initialX, A, b)
    normGrafFx = np.linalg.norm(gradfx)
    epsilon = 1e-10
    initialu_i = -gradfx
    iterations = 0
    while(normGrafFx > epsilon ):
        updatedX = calculateX_i(initialX, initialu_i,A)
        initialX = updatedX
        gradfx = gradientFx_1(initialX)
        updatedU = calculateU_i(gradfx, initialu_i,A)
        initialu_i = updatedU
        normGrafFx = np.linalg.norm(gradfx)
        iterations = iterations + 1 
    print(functionValue_1(initialX, A, b))
    print("The minimiser reached at : ", initialX)
    print("The number of iterations : ", iterations)

# conjugateGradient(np.array([0,0,0,0,1]))

#Question 1.2
#Find the minimizer for 0.5 x^T.A.A^T.x + 2b^T.A.x + b^Tb 
def functionValue_2(x, A, b):
    return  x.T @ A.T @ A @ x - 2*b.T @ A @ x + b.T @ b

def gradientFx_2(x, A, b):
    return 2* A.T @ A @ x - 2* A.T @ b

def calculateX_i2(x_i, u_i, A, b):
    gradFx = gradientFx_2(x_i, A, b)  # Calculate the gradient
    denominator = 2* u_i.T @ A.T @ A @ u_i#2* u_i^T*A^T*A*u_i
    numerator = -(np.dot(gradFx, u_i))
    alpha = numerator / denominator  # Step size
    return x_i + alpha * u_i  # Update the solution

def calculateU_i2(gradFx, u_i, A):
    denominator = 2* u_i.T @ A.T @ A @ u_i#2* u_i^T*A^T*A*u_i
    numerator = 2* gradFx.T@ A.T @ A @ u_i

    gamma = numerator/ denominator # g^i * u_i/denominator
    return -gradFx + gamma*u_i 

def conjugateGradient_2(initialX = np.array([1,1,1,0,1])) :
    results = f1(SR_No, False)
    A = np.array(results[0])
    b = np.array(results[1]).flatten()
    gradfx = gradientFx_2(initialX, A, b)
    normGrafFx = np.linalg.norm(gradfx)
    epsilon = 1e-10
    initialu_i = -gradfx
    iterations = 0
    while(normGrafFx > epsilon ):
        updatedX = calculateX_i2(initialX, initialu_i,A, b)
        initialX = updatedX
        gradfx = gradientFx_2(initialX, A, b)
        updatedU = calculateU_i2(gradfx, initialu_i,A)
        initialu_i = updatedU
        normGrafFx = np.linalg.norm(gradfx)
        iterations = iterations + 1 
        print(normGrafFx)
    print("Function value at minima is : ",functionValue_2(initialX, A, b))
    print("The minimiser reached at : ", initialX)
    print("The number of iterations : ", iterations)

# conjugateGradient_2()

results = f1(SR_No, False)
A = np.array(results[0])
b = np.array(results[1]).flatten()
print(np.linalg.eig(A.T@A))