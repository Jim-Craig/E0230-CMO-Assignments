from oracles import f1, f2, f3
import numpy as np
import matplotlib.pyplot as plt

SR_No = 24006
x0 = np.array([0,0,0,0,0])

def plotFunctionValueNewton(listFunctionList, intialPointList, ylabel):
    for i  in range(len(listFunctionList)):
        list = listFunctionList[i]
        txt = "Alpha is {xpoint}".format(xpoint = intialPointList[i])
        plt.plot(range(len(list)), list, label=txt)
    plt.ylabel(ylabel)
    plt.xlabel("Iterations")
    plt.legend()
    plt.show()

def plot(xPoints, yPoints, ylabel):
    plt.plot(yPoints)
    plt.ylabel(ylabel)
    plt.xlabel("Iterations")
    plt.show()

def updateX_i(x_i):
    return x_i - f2(x_i, SR_No, 2)

def ConstantGradientDescent(alpha, initialx = x0):
    discentDirection = - f2(initialx,SR_No, 1)
    normGradfx = np.linalg.norm(f2(initialx,SR_No, 1))
    gradFxList = [normGradfx]
    fxList = [f2(initialx,SR_No, 0)]
    xList = [initialx]
    for i in range(100):
        updatedX = initialx + alpha*discentDirection
        normGradfx = np.linalg.norm(f2(initialx,SR_No, 1))
        discentDirection = - f2(initialx,SR_No, 1)
        initialx = updatedX
        xList.append(initialx)
        gradFxList.append(normGradfx)
        fxList.append(f2(initialx,SR_No, 0))
    print("Final point for alpha = {alphaf} is {point}".format(alphaf = alpha, point = initialx))
    return fxList
    
def newtonMethod(x = x0):
    initialX = x
    gradFx = f2(initialX, SR_No, 1)
    normGradFx = np.linalg.norm(gradFx)
    functionValueList = [f2(initialX, SR_No, 0)]
    xPointsList = [x0]
    iteration = 0
    for i in range(100):
        updatedX = updateX_i(initialX)
        initialX = updatedX
        functionValueList.append(f2(initialX, SR_No, 0))
        xPointsList.append(initialX)
        iteration = iteration + 1
        gradFx = f2(initialX, SR_No, 1)
        normGradFx = np.linalg.norm(gradFx)

    gradFx = f2(initialX, SR_No, 1)
    normGradFx = np.linalg.norm(gradFx)
    print("final Norm of the Newton is :", normGradFx)
    print("final fucnction value of the Newton is :", functionValueList[-1])
    print("final xPoint of the Newton is :", xPointsList[-1])
    print("Number of iteration :", iteration)
    return functionValueList

listOfFunctionValueList = []
listAplha = []
for i in range(5):
    alpha = np.round(np.random.uniform(1e-3, 1e-2), 3)
    listAplha.append(alpha)
    listOfFunctionValueList.append(ConstantGradientDescent(alpha))
plotFunctionValueNewton(listOfFunctionValueList, listAplha, "Function Value")

newtonMethod(np.array([1.0, 1.0, 1.0, 1.0, 1.0]).T)