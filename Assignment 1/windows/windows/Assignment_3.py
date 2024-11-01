from CMO_A1 import f1, f2, f3, f4
import numpy as np
import math as mth
import matplotlib.pyplot as plt

SR_NO = 24006
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0], int)

def plotExpectedFunctionGraph(ListFxList):
    mean_list = np.mean(ListFxList, axis=0)
    deff_mean = np.array([[mean_list[i + 1] - mean_list[i] for i in range(len(mean_list) - 1)]])
    plot(np.arange(len(deff_mean)), deff_mean, "Difference in consequitive Expected function value")
    plt.show()

def contourPlots():
    # Create a grid of points (x, y)
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Calculate Z values for each (x, y) point
    Z = A3_Func(X, Y)

    # Create the contour plot
    plt.figure(figsize=(6, 6))
    contour = plt.contour(X, Y, Z, levels=20, cmap='viridis')

    # Add labels to the contour lines
    plt.clabel(contour, inline=True, fontsize=8)

    # Add labels and title
    plt.title('Contour Plot of f(x, y) = e^(xy)')
    plt.xlabel('x')
    plt.ylabel('y')

    # Show the plot
    plt.colorbar(contour)


def plotGraphs(xList, gradFxList, fxList):
    plot(np.arange(len(xList)), xList, "Norm of GradFx")

def plot(xPoints, yPoints, ylabel):
    plt.plot(xPoints, yPoints,color = 'r')
    plt.ylabel(ylabel)
    plt.xlabel("Iterations")

def A3_Func(x, y): 
    return np.exp(x * y)

def grad_Fx(x, y):
    fx = A3_Func(x, y)
    return np.array([y * fx, x * fx])

def hessian_Fx(x, y): 
    fx = A3_Func(x, y)
    return np.array([
        [y**2 * fx, (1 + x*y) * fx],  # F_xx, F_xy
        [(1 + x*y) * fx, x**2 * fx]   # F_yx, F_yy
    ])

def gaussian_noise():
    mean = [0, 0]
    cov = [[0.01, 0], [0, 0.01]] 
    noise = np.random.multivariate_normal(mean, cov, 1)[0]
    return noise
 
def ConstantGradientDescent_3(alpha):
    x = np.random.uniform(-1, 1, 1)[0]  # Extract scalar
    y = x
    initialx = np.array([x, y])
    fx = A3_Func(x, y)

    gradFx = grad_Fx(initialx[0], initialx[1])
    discentDirection = - gradFx
    iteration = 0
    normGradfx = np.linalg.norm(gradFx)
    gradFxList = [normGradfx]
    ListFxList = []
    fxList = [fx]
    xList = [x]
    yList = [y]
    tolerance  = 1e-3
    for i in range(50):
        while((normGradfx - 0.0) > tolerance and iteration < 10000):
            noise = gaussian_noise() 
            updatedX = initialx + alpha*(discentDirection + noise)
            initialx = updatedX
            x, y = initialx[0], initialx[1]

            fx = A3_Func(x, y)
            gradFx = grad_Fx(initialx[0], initialx[1])
            discentDirection = - gradFx
            normGradfx = np.linalg.norm(gradFx)

            xList.append(x)
            yList.append(y)
            gradFxList.append(normGradfx)
            fxList.append(fx)
            iteration +=1
            print(normGradfx)
        ListFxList.append(fxList)
    plotExpectedFunctionGraph(ListFxList)
    # contourPlots()
    # plt.quiver(xList[:-1], yList[:-1], 
    #            np.diff(xList), np.diff(yList), 
    #            scale_units='xy', angles='xy', scale=1, color='red', label='Descent Direction')

    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()

#5. gradient descent with a decreasing step-sizes
def DiminishingGradientDescent_3(alpha):
    x = np.random.uniform(-1, 1, 1)[0]  # Extract scalar
    y = x
    initialx = np.array([x, y])
    fx = A3_Func(x, y)

    gradFx = grad_Fx(initialx[0], initialx[1])
    discentDirection = - gradFx
    iteration = 0
    normGradfx = np.linalg.norm(gradFx)
    gradFxList = [normGradfx]
    fxList = [fx]
    xList = [x]
    yList = [y]
    ListFxList = []
    tolerance  = 1e-8
    min_alpha = 1e-3
    
    for i in range(50):
        while((normGradfx - 0.0) > tolerance and iteration <10000):
            updatedAlpha = max(alpha / (iteration + 1), min_alpha)
            noise = gaussian_noise()
            updatedNoise = noise/ (iteration + 1)
            updatedX = initialx + updatedAlpha*(discentDirection + noise)
            initialx = updatedX
            alpha = updatedAlpha
            x, y = initialx[0], initialx[1]

            fx = A3_Func(x, y)
            gradFx = grad_Fx( x, y)
            discentDirection = - gradFx
            normGradfx = np.linalg.norm(gradFx)

            xList.append(x)
            yList.append(y)
            gradFxList.append(normGradfx)
            fxList.append(fx)
            iteration +=1
            print(normGradfx)
        ListFxList.append(fxList)
    plotExpectedFunctionGraph(ListFxList)
    # contourPlots()
    # plt.quiver(xList[:-1], yList[:-1], 
    #            np.diff(xList), np.diff(yList), 
    #            scale_units='xy', angles='xy', scale=1, color='red', label='Descent Direction')

    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()

DiminishingGradientDescent(0.1)
