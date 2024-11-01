from oracles import f1, f2, f3
import numpy as np
import math as mth
import matplotlib.pyplot as plt

SR_No = 24006
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

def plotFunctionGradientDescent(listFunctionList1, alpha, qnwlist, ylabel):
    
    for i in range(5):
        txt = "Alpha is {xpoint}".format(xpoint = alpha[i])
        list = listFunctionList1[i]
        plt.plot(range(len(list)), list, marker="o",label=txt)
    plt.plot(range(len(qnwlist)), qnwlist, marker="o",label=txt)
    plt.ylabel(ylabel)
    plt.xlabel("Iterations")
    plt.legend()
    plt.show()

def ConstantGradientDescent(alpha, iterations, initialx = x0):
    discentDirection = - f2(initialx,SR_No, 1)
    normGradfx = np.linalg.norm(f2(initialx,SR_No, 1))
    gradFxList = [normGradfx]
    fxList = []
    print("Starting the Gradient Descent")
    cost = 0
    for i in range(100):
        updatedX = initialx + alpha*discentDirection
        initialx = updatedX
        normGradfx = np.linalg.norm(f2(initialx,SR_No, 1))
        discentDirection = - f2(initialx,SR_No, 1)
        gradFxList.append(normGradfx)
        fxList.append(f2(initialx,SR_No, 0))
        cost = cost + 1
    # print("Final point for alpha = {alphaf} is {point}".format(alphaf = alpha, point = initialx))
    # print("Best function value observed over all the 100 iterations", min(fxList))
    print("Norm of the function after Gradient Descent {norm} and cost is {cost1}".format(norm = normGradfx, cost1 = cost))
    print("Final X for alpha : {alpha1} is {x}".format(alpha1 = alpha, x = np.round(initialx,2)))
    return fxList

def upadteAlphaByELS(gradFx, descentDirection):#grad Fx(xk) and -gradFx(xK)
    numerator = - gradFx.T @ descentDirection
    fx_descentDirection = f2(descentDirection, SR_No, 0)
    denominator = 2*(fx_descentDirection - f2(x0, SR_No, 1).T @ descentDirection - f2(x0, SR_No, 0))
    alpha = numerator / denominator
    return alpha

def updateX_i(x_i, B_i, gradFx):
    descentDirection = DescentDirection(B_i, gradFx) #u_k
    alpha = upadteAlphaByELS(gradFx, descentDirection)
    return x_i + alpha*descentDirection #x_k+1 = x^k + alpha*u_k

def DescentDirection(B_i, gradFx):
    return -(B_i @ gradFx) #-B_k * gradFx_k

def updateBk(B_k, delta_k, gamma_k):
    differenceTerm = delta_k - B_k @ gamma_k #delta_k - b_k*gamma_k
    numerator = np.expand_dims(differenceTerm, -1) @ np.expand_dims(differenceTerm, -1).T
    denominator = differenceTerm.T @ gamma_k
    return B_k + (numerator/denominator)  

def quasiNewton(initialX = x0):
    gradFx = f2(initialX, SR_No, 1)#gradFx_k
    B_k = np.identity(5) #I
    normGradfx = np.linalg.norm(gradFx) #normGradFx
    tolerance = 1e-10
    fxList = []
    while(normGradfx > tolerance):
        updatedX = updateX_i(initialX, B_k, gradFx)
        delta_k = updatedX -initialX #delta_k = x_k+1 - x_k
        gradFxUpdated = f2(updatedX, SR_No, 1) #gradFx_k+1
        gamma_k = gradFxUpdated - gradFx #gradFx_k+1 - gradFx_k
        gradFx = gradFxUpdated
        B_kupdated = updateBk(B_k, delta_k, gamma_k)
        B_k = B_kupdated
        normGradfx = np.linalg.norm(gradFx)
        initialX = updatedX
        fxList.append(f2(initialX, SR_No, 0))
    print("Final X at Quasi Newton: ", np.round(initialX,2))

def partialNewton(initialX = x0):
    fxList1 = ConstantGradientDescent(0.05, 100, x0)
    # txt = "Alpha is {xpoint}".format(xpoint = 0.05)
    # plt.plot(range(len(fxList1)), fxList1, marker="o",label=txt)
    
    fxList2 = ConstantGradientDescent(0.001, 100, x0)
    # txt = "Alpha is {xpoint}".format(xpoint = 0.001)
    # plt.plot(range(len(fxList2)), fxList2, marker="o",label=txt)
    
    fxList3 = ConstantGradientDescent(0.051, 100, x0)
    # txt = "Alpha is {xpoint}".format(xpoint = 0.051)
    # plt.plot(range(len(fxList3)), fxList3, marker="o",label=txt)
    
    fxList4 = ConstantGradientDescent(0.025, 100, x0)
    # txt = "Alpha is {xpoint}".format(xpoint = 0.025)
    # plt.plot(range(len(fxList4)), fxList4, marker="o",label=txt)
    
    fxList5 = ConstantGradientDescent(0.003, 100, x0)
    # txt = "Alpha is {xpoint}".format(xpoint = 0.003)
    # plt.plot(range(len(fxList5)), fxList5, marker="o",label=txt)

    qnwfxlist = quasiNewton(initialX)
    # plt.plot(range(len(qnwfxlist)), qnwfxlist, marker="o",label="Quasi-Newton")

    # plt.ylabel("Function Value")
    # plt.xlabel("Iterations")
    # plt.legend()
    # plt.show()

partialNewton()