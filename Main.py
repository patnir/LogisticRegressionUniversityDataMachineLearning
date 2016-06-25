# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 20:25:30 2016

@author: Rahul Patni
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy
import math
from scipy.optimize import fmin_bfgs  

# Data contains scores of 2 exams and also whether the individual with the score was admitted or not

def loadData(X, y, filename):
    fhand = open(filename)
    for line in fhand:
        line = line.split(',')
        toAdd = []
        for i in range(len(line) - 1):
            toAdd.append(float(line[i]))
        X.append(toAdd)
        y.append([float(line[-1])])
    return;

def extractColummnFromMatrix(matrix, i):
    return [row[i] for row in matrix]

def printArray(array):
    for i in array:
        print i
    
def plotData(X, y):
    exam1Data = extractColummnFromMatrix(X, 0)
    exam2Data = extractColummnFromMatrix(X, 1)
    accepted = []
    rejected = []
    for i in range(len(y)):
        if (y[i][0] == 1):
            accepted.append([exam1Data[i], exam2Data[i]])
        else:
            rejected.append([exam1Data[i], exam2Data[i]])
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend('accepted', 'rejected')
    red_data = mpatches.Patch(color='red', label='Accepted')
    blue_data = mpatches.Patch(color='blue', label='Rejected')
    plt.legend(handles=[red_data, blue_data])
    plt.title('Exam Scores of Students Applying to University')
    plt.plot(extractColummnFromMatrix(accepted, 0), extractColummnFromMatrix(accepted, 1), 'ro')
    plt.plot(extractColummnFromMatrix(rejected, 0), extractColummnFromMatrix(rejected, 1), 'b+')    
    plt.show()

def sigmoid(z):
    result = -1 * z
    result = [[1.0 / (1.0 + math.exp(x))] for x in result]
    return result
    

def computeCostGradient(theta, X, y):
    m = len(y)
    # Splitting up cost calculation to two parts A, B
    sig = sigmoid(numpy.dot(X, numpy.transpose(theta)))
    A = [[math.log(x[0])] for x in sig]
    A = -1 * numpy.dot(numpy.transpose(y), A)
    sig1 = [[1 - x[0]] for x in sig]
    y1 = [[1 - x[0]] for x in y]
    B = [[math.log(x[0])] for x in sig1]
    B = -1 * numpy.dot(numpy.transpose(y1), B)
    cost = (1.0 / m) * (A + B)
    
    # Calculation of gradient
    grad = [sig[x][0] - y[x][0] for x in range(len(y))]
    grad = numpy.transpose(numpy.dot(numpy.transpose(grad), X))
    grad = (1.00 / m) * grad
    return (cost, grad)
    
def computeCost(theta, X, y):
    m = len(y)
    # Splitting up cost calculation to two parts A, B
    sig = sigmoid(numpy.dot(X, numpy.transpose(theta)))
    A = [[math.log(x[0] + 1)] for x in sig]
    A = -1 * numpy.dot(numpy.transpose(y), A)
    sig1 = [[1 - x[0]] for x in sig]
    y1 = [[1 - x[0]] for x in y]
    B = [[math.log(x[0] + 1)] for x in sig1]
    B = -1 * numpy.dot(numpy.transpose(y1), B)
    cost = (1.0 / m) * (A + B)
    return cost
    
def ComputeCost(theta, X, y):
    m = float(len(X))
    predictions = numpy.dot(X, theta)
    errors_squared = numpy.subtract(predictions, y)
    #print errors_squared
    errors_squared = [round(math.pow(x, 2), 5) for x in errors_squared]
    J = (1.0 / (2.0 * m)) * sum(errors_squared)
    return J 
    
def computeGradient(theta, X, y):
    m = len(y)
    # Splitting up cost calculation to two parts A, B
    sig = sigmoid(numpy.dot(X, numpy.transpose(theta)))
    # Calculation of gradient
    grad = [sig[x][0] - y[x][0] for x in range(len(y))]
    grad = numpy.transpose(numpy.dot(numpy.transpose(grad), X))
    grad = (1.00 / m) * grad
    return grad


def optimization(theta, X, y):
    x0 = theta
    theta = fmin_bfgs(computeCost, x0, fprime=computeGradient, args=(X, y))
    
def main():
    X = []
    y = []
    loadData(X, y, 'data.txt')
    plotData(X, y)
    # Add intercept data to X
    X = [[1.0] + x for x in X]
    theta = numpy.zeros((1, 3))
    cost = computeCost(theta, X, y)
    print "cost"
    printArray(cost)
    grad = computeGradient(theta, X, y)
    print "new gradient"
    printArray(grad)
    print "optimizing"
    optimization(theta, X, y)
    return;
    
if __name__ == "__main__":
    main()