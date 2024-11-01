from CMO_A1 import f1, f2, f3, f4
import numpy as np
import math as mth
import matplotlib.pyplot as plt

SR_NO = 24006

def plot(function, begin, end, num):
    xPoints = np.random.uniform(begin, end, num)
    yPoints = []
    for pt in xPoints:
        yPoints.append(function(SR_NO, pt))
    plt.scatter(xPoints, yPoints)
    plt.show()

#Question 1
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

#isConvex(f2, [-2,2])
def check() :
    functions = [f1, f2]
    print("Checking Convexiety")
    for func in functions:
        isConvex(func, [-2,2])
    print("Checking Strictly Convexiety")
    for func in functions:
        isStrictlyConvex(func, [-2,2])

####################################################

#2. isCoercive returns whether the function is coercive or not
def isCoercive(func):
    pXInfPoint = findPointsOfExpansion(func, 0, 100 )
    nXInfPoint = findPointsOfExpansion(func, -100, 0 )
    pointOfInf = max(pXInfPoint, abs(nXInfPoint))
    
    isIncreasingSequence = doesFunctionGoInf(func, -pointOfInf, -100) and doesFunctionGoInf(func, pointOfInf, 100)

    if isIncreasingSequence:
        print("Function is coercive and starts expanding at point : ", pointOfInf)
        return True
    else:
        print("Function is not coercive")
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
            print(xPoints[i])
            return xPoints[i]# Return the point where the difference is 0 or positive
        
#Return the first point if no clear increasing trend is found
    return xPoints[0]
            


# TODO: write a function FindStationaryPoints that finds the Roots, Minima, LocalMaxima     
isCoercive(f3)