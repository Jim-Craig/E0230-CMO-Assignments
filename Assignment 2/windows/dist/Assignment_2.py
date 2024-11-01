from oracles_updated import f1, f2, f3
import numpy as np
import matplotlib.pyplot as plt
import math as mth

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

###########################################################################################
#Question 2 : Newton's Method
x0 = np.array([0,0,0,0,0])

def plotFunctionValueNewton_2(listFunctionList, intialPointList, ylabel):
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

def updateX_i_2(x_i):
    return x_i - f2(x_i, SR_No, 2)

def ConstantGradientDescent_2(alpha, initialx = x0):
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
    
def newtonMethod_2(x = x0):
    initialX = x
    gradFx = f2(initialX, SR_No, 1)
    normGradFx = np.linalg.norm(gradFx)
    functionValueList = [f2(initialX, SR_No, 0)]
    xPointsList = [x0]
    iteration = 0
    for i in range(100):
        updatedX = updateX_i_2(initialX)
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
    listOfFunctionValueList.append(ConstantGradientDescent_2(alpha))
plotFunctionValueNewton_2(listOfFunctionValueList, listAplha, "Function Value")

newtonMethod_2(np.array([1.0, 1.0, 1.0, 1.0, 1.0]).T)
###################################################################################
#3. Newton’s Method continued:

x0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

def plotFunctionGradientDescent_3(listFunctionList1, listFunctionList2, alpha, ylabel):
    
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

def ConstantGradientDescent_3(alpha, iterations, initialx = x0):
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

def newtonMethod_3(iterations, x = x0):
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

fxList = ConstantGradientDescent_3(0.1, 100)
txt = "Alpha is {xpoint}".format(xpoint = 0.1)
plt.plot(range(len(fxList[2])), fxList[2],label=txt)
plt.ylabel(txt)
plt.xlabel("Iterations")
plt.legend()
plt.show()


def partialNewton_3(alpha, iterations, initialX = x0):
    # 0.0565 , 23
    cost = 0
    print("Experiment with {alpha1} and {iterations1}".format(alpha1 = alpha, iterations1 = iterations))
    updatedX = ConstantGradientDescent_3(alpha, iterations, initialX)
    initialX = updatedX[0]
    cost += updatedX[1]
    GDFxList = updatedX[2]
    updatedX = newtonMethod_3(100-iterations, initialX)
    cost += updatedX[1]
    NFxList = updatedX[2]
    print(cost)
    print("X point value: ", updatedX[0])
    print("Function value: ", f3(updatedX[0],SR_No, 0))
    print(NFxList)
    plotFunctionGradientDescent_3(GDFxList, NFxList, alpha, "FunctionValue")
    
# partialNewton(0.0565 , 23)

#################################################################################################
#4. Quasi-Newton Methods

x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

def plotFunctionGradientDescent_4(listFunctionList1, alpha, qnwlist, ylabel):
    
    for i in range(5):
        txt = "Alpha is {xpoint}".format(xpoint = alpha[i])
        list = listFunctionList1[i]
        plt.plot(range(len(list)), list, marker="o",label=txt)
    plt.plot(range(len(qnwlist)), qnwlist, marker="o",label=txt)
    plt.ylabel(ylabel)
    plt.xlabel("Iterations")
    plt.legend()
    plt.show()

def ConstantGradientDescent_4(alpha, iterations, initialx = x0):
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

def upadteAlphaByELS_4(gradFx, descentDirection):#grad Fx(xk) and -gradFx(xK)
    numerator = - gradFx.T @ descentDirection
    fx_descentDirection = f2(descentDirection, SR_No, 0)
    denominator = 2*(fx_descentDirection - f2(x0, SR_No, 1).T @ descentDirection - f2(x0, SR_No, 0))
    alpha = numerator / denominator
    return alpha

def updateX_i(x_i, B_i, gradFx):
    descentDirection = DescentDirection_4(B_i, gradFx) #u_k
    alpha = upadteAlphaByELS_4(gradFx, descentDirection)
    return x_i + alpha*descentDirection #x_k+1 = x^k + alpha*u_k

def DescentDirection_4(B_i, gradFx):
    return -(B_i @ gradFx) #-B_k * gradFx_k

def updateBk_4(B_k, delta_k, gamma_k):
    differenceTerm = delta_k - B_k @ gamma_k #delta_k - b_k*gamma_k
    numerator = np.expand_dims(differenceTerm, -1) @ np.expand_dims(differenceTerm, -1).T
    denominator = differenceTerm.T @ gamma_k
    return B_k + (numerator/denominator)  

def quasiNewton_4(initialX = x0):
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
        B_kupdated = updateBk_4(B_k, delta_k, gamma_k)
        B_k = B_kupdated
        normGradfx = np.linalg.norm(gradFx)
        initialX = updatedX
        fxList.append(f2(initialX, SR_No, 0))
    print("Final X at Quasi Newton: ", np.round(initialX,2))

def partialNewton_4(initialX = x0):
    fxList1 = ConstantGradientDescent_4(0.05, 100, x0)
    # txt = "Alpha is {xpoint}".format(xpoint = 0.05)
    # plt.plot(range(len(fxList1)), fxList1, marker="o",label=txt)
    
    fxList2 = ConstantGradientDescent_4(0.001, 100, x0)
    # txt = "Alpha is {xpoint}".format(xpoint = 0.001)
    # plt.plot(range(len(fxList2)), fxList2, marker="o",label=txt)
    
    fxList3 = ConstantGradientDescent_4(0.051, 100, x0)
    # txt = "Alpha is {xpoint}".format(xpoint = 0.051)
    # plt.plot(range(len(fxList3)), fxList3, marker="o",label=txt)
    
    fxList4 = ConstantGradientDescent_4(0.025, 100, x0)
    # txt = "Alpha is {xpoint}".format(xpoint = 0.025)
    # plt.plot(range(len(fxList4)), fxList4, marker="o",label=txt)
    
    fxList5 = ConstantGradientDescent_4(0.003, 100, x0)
    # txt = "Alpha is {xpoint}".format(xpoint = 0.003)
    # plt.plot(range(len(fxList5)), fxList5, marker="o",label=txt)

    qnwfxlist = quasiNewton_4(initialX)
    # plt.plot(range(len(qnwfxlist)), qnwfxlist, marker="o",label="Quasi-Newton")

    # plt.ylabel("Function Value")
    # plt.xlabel("Iterations")
    # plt.legend()
    # plt.show()

partialNewton_4()