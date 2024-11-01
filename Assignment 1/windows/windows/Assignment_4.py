import numpy as np
import math as mth
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp

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
def goldenSearch(a,b):
    phi = (1+mth.sqrt(5))/2
    # x1 = phi*a + (1-phi)*b
    # x2 = (1-phi)*a + phi*b
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
    diff = np.array(bList) - np.array(aList) 
    ratios = []
    for i in range(len(diff) - 1):
        ratio = diff[i] / diff[i + 1]
        ratios.append(ratio)

    plotGraphs(ratios)
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

##4.2 Fibonacci sequence Search
def goldenSearch(a,b):

    phi = (1+mth.sqrt(5))/2
    # x1 = phi*a + (1-phi)*b
    # x2 = (1-phi)*a + phi*b
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
    diff = np.array(bList) - np.array(aList) 
    ratios = []
    for i in range(len(diff) - 1):
        ratio = diff[i] / diff[i + 1]
        ratios.append(ratio)

    plotGraphs(ratios)
    return tupple

def fibonacci(n):
    # Generate the first n Fibonacci numbers
    fibs = [0, 1]
    for i in range(2, n):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs

def fibonacciSearch(a, b, tolerance=1e-3):
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
        print("Difference : ", difference)
        k -= 1
    difference = abs(a - b)
    iteration += 1
    print("Number of actual iterations : ", iteration)
    diff = np.array(bList) - np.array(aList) 
    ratios = []
    for i in range(len(diff) - 1):
        ratio = diff[i] / diff[i + 1]
        ratios.append(ratio)

    plotGraphs(ratios)

fibonacciSearch(1,3,1e-4)
