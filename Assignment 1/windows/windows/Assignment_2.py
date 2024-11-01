from CMO_A1 import f1, f2, f3, f4
import numpy as np
import math as mth
import matplotlib.pyplot as plt

SR_NO = 24006
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0], int)

def plotGraphs(xList, gradFxList, fxList):
    plot(np.arange(len(xList)), gradFxList, "Norm of GradFx")
    
    Fxerror = []
    for x in fxList:
        norm = np.linalg.norm(x - fxList[len(fxList) -1])
        Fxerror.append(norm) 
    plot(np.arange(len(Fxerror)), Fxerror, "F(xk) - F(x*)")
    ratios = []
    for i in range(len(Fxerror) - 2):
        ratio = Fxerror[i] / Fxerror[i + 1]
        ratios.append(ratio)
    plot(np.arange(len(ratios)), ratios, "Ratio of Fxk - FxT difference values*" )

    Xerror = []
    for x in xList:
        norm = np.linalg.norm(x - xList[len(xList) -1])
        Xerror.append(mth.pow(norm, 2)) 
    plot(np.arange(len(Xerror)), Xerror, "Square of Norm of Xk - X*" )

    ratios = []
    for i in range(len(Xerror) - 2):
        ratio = Xerror[i] / Xerror[i + 1]
        ratios.append(ratio)
    plot(np.arange(len(ratios)), ratios, "Ratio of xk-xT difference values*" )
    

def plot(xPoints, yPoints, ylabel):
    plt.scatter(xPoints, yPoints)
    plt.ylabel(ylabel)
    plt.xlabel("Iterations")
    plt.show()


#2.a Gradient Descent
def ConstantGradientDescent(alpha, initialx):
    oracleOPInitial = f4(SR_NO, initialx)
    discentDirection = - oracleOPInitial[1]
    iteration = 0
    normGradfx = np.linalg.norm(oracleOPInitial[1])
    gradFxList = [normGradfx]
    fxList = [oracleOPInitial[0]]
    xList = [initialx]
    tolerance  = 1e-8
    while((normGradfx - 0.0) > tolerance):
        updatedX = initialx + alpha*discentDirection
        oracleOP = f4(SR_NO, updatedX)
        normGradfx = np.linalg.norm(oracleOP[1])
        discentDirection = - oracleOP[1]
        initialx = updatedX
        xList.append(initialx)
        gradFxList.append(normGradfx)
        fxList.append(oracleOP[0])
        iteration +=1
    plotGraphs(xList, gradFxList, fxList)


#2.b Gradient Descent with Diminishing step-size
def DiminishingGradientDescent(alpha, initialx):
    oracleOPInitial = f4(SR_NO, initialx)
    discentDirection = - oracleOPInitial[1]
    iteration = 0
    normGradfx = np.linalg.norm(oracleOPInitial[1])
    gradFxList = [normGradfx]
    fxList = [oracleOPInitial[0]]
    xList = [initialx]
    for i in range(10000):
        updatedAlpha = alpha/(i+1)
        updatedX = initialx + updatedAlpha*discentDirection
        oracleOP = f4(SR_NO, updatedX)
        normGradfx = np.linalg.norm(oracleOP[1])
        discentDirection = - oracleOP[1]
        initialx = updatedX
        xList.append(initialx)
        gradFxList.append(normGradfx)
        fxList.append(oracleOP[0])
        iteration +=1
        error = []
    plotGraphs(xList, fxList, gradFxList)

#2.c : Update alpha using Wolfe Goldstein conditions
def InExactLineSearch(c1, gamma):
    c2 = 1 - c1
    initialX = x0
    oracleOPInitial = f4(SR_NO, initialX)  # f(x0), grad(f(x0))
    discentDirection = -oracleOPInitial[1]  # -grad(f(x0))
    iteration = 0
    normGradfx = np.linalg.norm(oracleOPInitial[1])
    gradFxList = [normGradfx]
    fxList = [oracleOPInitial[0]]
    xList = [initialX]
    tolerance = 1e-7
    alpha = updateAlphaByWolfe(oracleOPInitial[0], oracleOPInitial[1], initialX, discentDirection, c1, c2, gamma)
    
    while normGradfx > tolerance:
        updatedX = initialX + alpha * discentDirection
        oracleOP = f4(SR_NO, updatedX)  # f(xk+1), grad(f(xk+1))
        normGradfx = np.linalg.norm(oracleOP[1])
        discentDirection = -oracleOP[1]
        initialX = updatedX
        
        # Update alpha using Wolfe conditions
        alpha = updateAlphaByWolfe(oracleOP[0], oracleOP[1], initialX, discentDirection, c1, c2, gamma)
        
        xList.append(initialX)
        gradFxList.append(normGradfx)
        fxList.append(oracleOP[0])
        iteration += 1
        print(normGradfx)

    print(f"Final X: {xList[-1]}")
    print(f"Final Fx: {fxList[-1]}")
    print(f"Total iterations: {iteration}")
    plotGraphs(xList, gradFxList, fxList)


def updateAlphaByWolfe(fx, gradFx, initialX, descentDirection, c1, c2, gamma):
    alpha = 1
    max_iterations = 1000  # To avoid infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        updatedX = initialX + alpha * descentDirection
        oracleOP = f4(SR_NO, updatedX)  # f(xk+1), grad(f(xk+1))

        # Wolfe condition checks
        check1 = oracleOP[0] <= fx + c1 * alpha * np.dot(descentDirection, gradFx)
        check2 = -np.dot(descentDirection, oracleOP[1]) <= -c2 * np.dot(descentDirection, gradFx)

        if check1 and check2:
            break  # Wolfe conditions satisfied

        # Update alpha
        alpha *= gamma
        iteration += 1

    # if iteration >= max_iterations:
    #     print("Warning: Wolfe conditions not satisfied within max iterations.")

    return alpha
    

#2.d Exact Line Search:

def ExactLineSearch():
    initialX = x0
    oracleOPInitial = f4(SR_NO, initialX)
    discentDirection = - oracleOPInitial[1]
    iteration = 0
    normGradfx = np.linalg.norm(oracleOPInitial[1])
    gradFxList = [normGradfx]
    fxList = [oracleOPInitial[0]]
    xList = [initialX]
    tolerance  = 1e-8
    alpha = upadteAlphaByELS(oracleOPInitial[1], discentDirection)  
    
    while(normGradfx > tolerance):
        updatedX = initialX + alpha*discentDirection #xk+1
        oracleOP = f4(SR_NO, updatedX)#fx(xk+1), gradFx(xk+1)
        normGradfx = np.linalg.norm(oracleOP[1])
        discentDirection = - oracleOP[1]#- gradFx(xk)
        initialX = updatedX
        updatedAlpha = upadteAlphaByELS(oracleOP[1], discentDirection)#we pass grad Fx(xk) and -gradFx(xK)
        alpha = updatedAlpha
        xList.append(initialX)
        gradFxList.append(normGradfx)
        fxList.append(oracleOP[0])
        iteration +=1
        print(normGradfx)
    
    print(xList[len(xList) -1])
    print(fxList[len(fxList) -1])
    print(iteration)
    plotGraphs(xList, gradFxList, fxList)

def findAofFx(fx, gradFx, descentDirection):
    result = 2*(fx - np.dot(gradFx, descentDirection))
    return result

def upadteAlphaByELS(gradFx, descentDirection):#grad Fx(xk) and -gradFx(xK)
    fxDescentDirection = f4(SR_NO, descentDirection)[0]#f(descentDirection)
    gradFxZero = f4(SR_NO, x0)[1]#gradFx(0)
    numerator = -(np.dot(gradFx, descentDirection))
    demonminator = findAofFx(fxDescentDirection, gradFxZero, descentDirection)#pkTApk
    return numerator/demonminator


ExactLineSearch()