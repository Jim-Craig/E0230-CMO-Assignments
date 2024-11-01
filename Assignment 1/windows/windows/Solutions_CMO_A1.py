from CMO_A1 import f1, f2, f3, f4
import numpy as np
import math as mth
import matplotlib.pyplot as plt

SR_NO = 24006
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0], int)
#####################################################################################################################
#Question 1
def plot(function, begin, end, num):
    xPoints = np.random.uniform(begin, end, num)
    yPoints = []
    for pt in xPoints:
        yPoints.append(function(SR_NO, pt))
    plt.scatter(xPoints, yPoints)
    plt.show()


#1: Implement the main function as requested in the Assignment for checking if the function is convex
def isConvex(function, interval) :
    flag = True
    for i in range(1,1000) :
        randomValues = np.random.uniform(interval[0], interval[1], 2)
        result = checkConvex(function, randomValues[0], randomValues[1])
        if(not result):
            flag = False
            break
    if(flag):
        print("Function is convex")
    else:
        print("Function not convex")

def isStrictlyConvex(function, interval) :
    flag = True
    for i in range(1,1000) :
        randomValues = np.random.uniform(interval[0], interval[1], 2)
        result = checkStrictlyConvex(function, randomValues[0], randomValues[1])
        if(not result):
            flag = False
            break
    if(flag):
        print("Function is strictly convex \n")
    else:
        print("Function not strictly convex \n")
    
# Function that contains the logic to check if the function is convex or not. 
# we check the convexity condition for different values of lambda, ld. We compare the difference between a tolerance of 1e-5
#  Return true or false depending upon the result
def checkConvex(function, x, y):    
    lambdaPoints = np.random.uniform(0, 1, 100)
    for ld in lambdaPoints :
        intPoints = ld*x + (1-ld)*y
        difference = function(SR_NO,intPoints) - (ld*function(SR_NO,x) + (1-ld)*function(SR_NO,y))
        tolerance = 1e-5
        if(difference <= tolerance):
            continue 
        else:
            return False
    return True

# Function that contains the logic to check if the function is Strictly convex or not. 
# we check the convexity condition for different values of lambda, ld. Return true or false depending upon the result
def checkStrictlyConvex(function, x, y):  
    flag = True  
    lambdaPoints = np.random.uniform(0, 1, 100)
    for ld in lambdaPoints :
        if(ld != 0 and ld != 1): 
            intPoints = ld*x + (1-ld)*y
            difference = function(SR_NO,intPoints) - (ld*function(SR_NO,x) + (1-ld)*function(SR_NO,y))
            tolerance = 1e-20
            if(difference <= tolerance):
                continue 
            else:
                return False

    return True

#TODO: Write a program to find x*
def findMinimum(func, a, b): 
    xPoints = np.linspace(a, b, 1000)
    minimum = func(SR_NO, xPoints[0])
    xStar = xPoints[0]
    for i in range(1, len(xPoints)):
        fx = func(SR_NO, xPoints[i])
        if(minimum > fx):
            minimum = fx
            xStar = xPoints[i]
    print("X star is ", xStar)
    print("Minimum function value is ", minimum)
    

####################################################

#2. isCoercive returns whether the function is coercive or not
def isCoercive(func):
    pXInfPoint = findPointsOfExpansion(func, 0, 100 )
    nXInfPoint = findPointsOfExpansion(func, -100, 0 )
    pointOfInf = max(pXInfPoint, abs(nXInfPoint))
    
    isIncreasingSequence = doesFunctionGoInf(func, -pointOfInf, -100) and doesFunctionGoInf(func, pointOfInf, 100)

    if isIncreasingSequence:
        print("Function F3 is coercive and starts expanding at point : ", pointOfInf)
        return True
    else:
        print("Function F3  is not coercive")
        return False
    

def doesFunctionGoInf(func, begin, end):
    stepSize = 0.5
    x_values = np.arange(begin, end, stepSize)
    f_values = [func(SR_NO, x) for x in x_values]

    # Check if the function values are increasing and return the result
    return np.all(np.diff(f_values) > 0)
    
# Finds the point where the function starts to go towards the infinity
def findPointsOfExpansion(func, begin , end):
    xPoints = xPoints = np.linspace(begin, end, 10000)
    f_values = [func(SR_NO, x) for x in xPoints]

    diff_sequence = np.diff(f_values)
    tolerance = 1e-8
    item = len(diff_sequence)-1

    for i in range(len(diff_sequence) - 1, -1, -1):
        if -diff_sequence[i] > tolerance:
            return xPoints[i]# Return the point where the difference is 0 or positive
        
#Return the first point if no clear increasing trend is found
    return xPoints[0]
            


# TODO: write a function FindStationaryPoints that finds the Roots, Minima, LocalMaxima  
def questionOneCheck():
    functions = [f1, f2]
    print("Checking Convexiety")
    for func in functions:
        print("Checking for ",func)
        isConvex(func, [-2,2])
    print("Checking Strictly Convexiety")
    for func in functions:
        print("Checking for ",func)
        isStrictlyConvex(func, [-2,2])
    print("Checking Strictly Convexiety for f3")
    isCoercive(f3)


#####################################################################################################################
#Question : 2
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
def ConstantGradientDescent(alpha = 1e-5, initialx =x0):
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
    print("Finished the Constant Gradient Descent Algo.")
    print("Value of x* is ", xList[len(xList)-1])
    print("Value of f(x*) is ", fxList[len(fxList)-1])
    print("Number of iterations is ", iteration)

#2.b Gradient Descent with Diminishing step-size
def DiminishingGradientDescent(alpha=1e-3, initialx=x0, iterations = 10000):
    oracleOPInitial = f4(SR_NO, initialx)
    discentDirection = - oracleOPInitial[1]
    iteration = 0
    normGradfx = np.linalg.norm(oracleOPInitial[1])
    gradFxList = [normGradfx]
    fxList = [oracleOPInitial[0]]
    xList = [initialx]
    for i in range(iterations):
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
    print("Finished the Diminishing Gradient Descent Algo.")
    print("Value of x* is ", xList[len(xList)-1])
    print("Value of f(x*) is ", fxList[len(fxList)-1])
    print("Number of iterations is ", iteration)

#2.c : Update alpha using Wolfe Goldstein conditions
def InExactLineSearch(c1, gamma, initialX0 = x0):
    c2 = 1 - c1
    initialX = initialX0
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

    print("Finished the InExact Line Search Gradient Descent Algo.")
    print("Value of x* is ", xList[len(xList)-1])
    print("Value of f(x*) is ", fxList[len(fxList)-1])
    print("Number of iterations is ", iteration)

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
    
    print("Finished the Exact Line Search Gradient Descent Algo.")
    print("Value of x* is ", xList[len(xList)-1])
    print("Value of f(x*) is ", fxList[len(fxList)-1])
    print("Number of iterations is ", iteration)

def findAofFx(fx, gradFx, descentDirection):
    result = 2*(fx - np.dot(gradFx, descentDirection))
    return result

def upadteAlphaByELS(gradFx, descentDirection):#grad Fx(xk) and -gradFx(xK)
    fxDescentDirection = f4(SR_NO, descentDirection)[0]#f(descentDirection)
    gradFxZero = f4(SR_NO, x0)[1]#gradFx(0)
    numerator = -(np.dot(gradFx, descentDirection))
    demonminator = findAofFx(fxDescentDirection, gradFxZero, descentDirection)#pkTApk
    return numerator/demonminator

def questionTwoCheck():
    ConstantGradientDescent()
    DiminishingGradientDescent()
    InExactLineSearch(0.1, 0.5)
    ExactLineSearch()
#####################################################################################################################
#Question 3:
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
#####################################################################################################################
#Question 4

def func(x):
    return x*(x-1)*(x-3)*(x+2)

def plotGraphs(aList, bList=[]):
    # Compute the differences for arrows
    a_diff = [aList[i + 1] - aList[i] for i in range(len(aList) - 1)]
    # b_diff = [bList[i + 1] - bList[i] for i in range(len(bList) - 1)]

    # Scatter plot of the points
    plt.scatter(range(len(aList) - 1), aList[:-1], color='red', label='Interval ratios')
    # plt.scatter(range(len(bList) - 1), bList[:-1], color='blue', label='F(b)')

    # Plot arrows to show direction
    # plt.quiver(range(len(aList) - 1), aList[:-1], range(1, len(aList)), a_diff, angles='xy', scale_units='xy', scale=1, color='red', alpha=0.5)
    # plt.quiver(range(len(bList) - 1), bList[:-1], range(1, len(bList)), b_diff, angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.5)

    
    # for i in range(len(aList) - 1):
    #     plt.annotate('', xy=(aList[i+1], i+1), xytext=(aList[i], i),
    #              arrowprops=dict(facecolor='red', shrink=0.01))

    # for i in range(len(bList) - 1):
    #     plt.annotate('', xy=(bList[i+1], i+1), xytext=(bList[i], i),
    #              arrowprops=dict(facecolor='blue', shrink=0.01))


    plt.xlabel('Iterations')
    plt.ylabel('Interval ratios')
    plt.legend()
    plt.grid
    plt.show()

#4.1. Golden Section Search Implementation
def goldenSearch(a = 1,b = 3):
    phi = (1+mth.sqrt(5))/2
    x1 = b - (b - a) / phi
    x2 = a + (b - a) / phi
    difference = abs(a - b)
    f1 = func(x1)
    f2 = func(x2)
    tolerance = 1e-4
    tupple = [[a,b]]
    iteration = 0
    aList=[x1]
    bList=[x2]
    while(difference > tolerance ):
        if(f1 < f2):
            b= x2
            x2 = x1
            f2 = f1
            x1 = b - (b - a) / phi
            f1 = func(x1)
        else:
            a = x1  
            x1 = x2  
            f1 = f2  
            x2 = a + (b - a) / phi
            f2 = func(x2)
        aList.append(x1)
        bList.append(x2)
        tupple.append([a,b]) 
        difference = abs(a - b)
        iteration += 1
    print("Number of actual iterations : ", iteration)
    return tupple

def No_of_iterations(a, b, tolerance):
    phi = (1 + mth.sqrt(5)) / 2
    iterations = mth.ceil(mth.log((b - a) / tolerance) / mth.log(phi))
    return iterations



def findAllStationaryPoints():
    for i in range(10) :
        x, y = np.random.uniform(1, 3, 2).T
        print(goldenSearch(x, y))

def exportToExcel(values):
    df = pd.DataFrame({
    'Serial Number': range(1, len(values) + 1),
    'Intervals': values
    })

    df.to_excel('table_Fibonacci.xlsx', index=False)

def fibonacci(n):
    # Generate the first n Fibonacci numbers
    fibs = [0, 1]
    for i in range(2, n):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs

def fibonacciSearch(a=1, b=3, tolerance=1e-3):
    # Determine the number of Fibonacci numbers needed
    n = 25
    fibs = fibonacci(n)
    
    k = n - 1
    x1 = a + (fibs[k - 2] / fibs[k]) * (b - a)
    x2 = a + (fibs[k - 1] / fibs[k]) * (b - a)
    difference = abs(a - b)
    f1 = func(x1)
    f2 = func(x2)
    tolerance = 1e-4
    tupple = [[a,b]]
    iteration = 0
    aList=[f1]
    bList=[f2]
    while(difference > tolerance ):
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (fibs[k - 3] / fibs[k - 1]) * (b - a)
            f1 = func(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (fibs[k - 2] / fibs[k - 1]) * (b - a)
            f2 = func(x2)
        aList.append(f1)
        bList.append(f2)
        iteration+=1
        tupple.append([a,b])
        difference = abs(a - b)
        k -= 1
    difference = abs(a - b)
    iteration += 1
    print("Number of actual iterations : ", iteration)

def questionFourCheck():
    print("Start Golden Section search")
    goldenSearch()
    print("Start Fibonacci search")
    fibonacciSearch()



#####################################################################################################################
questionOneCheck()
questionTwoCheck()
questionFourCheck()