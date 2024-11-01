from oracles_updated import f1, f2, f3
import numpy as np
import math as mth
import matplotlib.pyplot as plt

SR_No = 24006
x0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

def plotFunctionGradientDescent(listFunctionList1, listFunctionList2, alpha, ylabel):
    
    list = listFunctionList1
    txt = "Alpha is {xpoint}".format(xpoint = alpha)
    plt.plot(range(len(list)), list, color="b", marker="o",label=txt)
    offset = len(list)
    # [x+10 for x in range(10)]
    plt.plot(np.array([x+offset for x in range(len(listFunctionList2))]), listFunctionList2, color="r", marker="x")
    plt.ylabel(ylabel)
    plt.xlabel("Iterations")
    plt.legend()
    plt.show()

def ConstantGradientDescent(alpha, iterations, initialx = x0):
    discentDirection = - f3(initialx,SR_No, 1)
    normGradfx = np.linalg.norm(f3(initialx,SR_No, 1))
    gradFxList = [normGradfx]
    fxList = [f3(initialx,SR_No, 0)]
    xList = [initialx]
    print("Starting the Gradient Descent")
    cost = 0
    for i in range(iterations):
        updatedX = initialx + alpha*discentDirection
        initialx = updatedX
        normGradfx = np.linalg.norm(f3(initialx,SR_No, 1))
        discentDirection = - f3(initialx,SR_No, 1)
        xList.append(initialx)
        gradFxList.append(normGradfx)
        fxList.append(f3(initialx,SR_No, 0))
        cost = cost + 1
    # print("Final point for alpha = {alphaf} is {point}".format(alphaf = alpha, point = initialx))
    # print("Best function value observed over all the 100 iterations", min(fxList))
    print("Norm of the function after Gradient Descent {norm} and cost is {cost1}".format(norm = normGradfx, cost1 = cost))
    return initialx, cost, fxList

def updateX_i(x_i):
    return x_i - f3(x_i, SR_No, 2)

def newtonMethod(iterations, x = x0):
    initialX = x
    cost = 0
    print("Starting the Newton Method")
    normGradfx = np.linalg.norm(f3(initialX,SR_No, 1))
    tolerance = 1e-5
    fxList = []
    while(normGradfx > tolerance):
        updatedX = updateX_i(initialX)
        initialX = updatedX
        normGradfx = np.linalg.norm(f3(initialX,SR_No, 1))
        cost = cost + 25
        fxList.append(f3(initialX,SR_No, 0))
    # print("xPoint value at iteration 100 is : ",initialX)
    print("Norm of the function after Newton Method {norm} and cost is {cost1}".format(norm = np.linalg.norm(f3(initialX, SR_No, 1)), cost1 = cost))
    return initialX, cost, fxList

fxList = ConstantGradientDescent(0.1, 100)
txt = "Alpha is {xpoint}".format(xpoint = 0.1)
plt.plot(range(len(fxList[2])), fxList[2],label=txt)
plt.ylabel(txt)
plt.xlabel("Iterations")
plt.legend()
plt.show()

k_step = range(1, 100, 10)


def partialNewton(alpha, iterations, initialX = x0):
    # 0.0565 , 23
    cost = 0
    print("Experiment with {alpha1} and {iterations1}".format(alpha1 = alpha, iterations1 = iterations))
    updatedX = ConstantGradientDescent(alpha, iterations, initialX)
    initialX = updatedX[0]
    cost += updatedX[1]
    GDFxList = updatedX[2]
    updatedX = newtonMethod(100-iterations, initialX)
    cost += updatedX[1]
    NFxList = updatedX[2]
    print(cost)
    print("X point value: ", updatedX[0])
    print("Function value: ", f3(updatedX[0],SR_No, 0))
    print(NFxList)
    plotFunctionGradientDescent(GDFxList, NFxList, alpha, "FunctionValue")
    
# partialNewton(0.0565 , 23)
# print([x+10 for x in range(10)])
# partialNewton(0.0565, 23)