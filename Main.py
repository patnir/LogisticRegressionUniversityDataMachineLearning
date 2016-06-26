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
    result = -1.0 * z
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
    printArray(sig)
    # shifting values the the sigmoid function to ensure there is no log(0)
    A = [[math.log(x[0] + 1)] for x in sig]
    A = -1 * numpy.dot(numpy.transpose(y), A)
    sig1 = [[1.0 - x[0]] for x in sig]
    y1 = [[1.0 - x[0]] for x in y]
    B = [[math.log(x[0] + 1)] for x in sig1]
    B = -1 * numpy.dot(numpy.transpose(y1), B)
    cost = (1.0 / m) * (A + B)
    return cost
    
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
    return theta
    

def predict(theta, X):
    predictions = sigmoid(numpy.dot(X, numpy.transpose(theta)));
    for i in range(len(predictions)):
        if (predictions[i][0] >= 0.5):
            predictions[i][0] = 1
        else:
            predictions[i][0] = 0
    return predictions
    
def accuracy(theta, X, y):
    predictions = predict(theta, X)
    correct = 0
    for i in range(len(predictions)):
        if (predictions[i][0] == y[i][0]):
            correct += 1.0
    print "Training accuracy: ",correct / len(y) * 100
    
def plotDecisionBoundary(theta, X, y):
    maxExam1 = max(extractColummnFromMatrix(X, 1)) + 2
    minExam1 = min(extractColummnFromMatrix(X, 1)) - 2
    xData = [minExam1, maxExam1]
    yData = []
    for i in range(len(xData)):
        yData.append((-1 / theta[2]) * (numpy.prod(xData[i] * theta[1]) + theta[0]))
    exam1Data = extractColummnFromMatrix(X, 1)
    exam2Data = extractColummnFromMatrix(X, 2)
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
    plt.plot(xData, yData, 'g-') 
    plt.show()

def testing(X, y):
    theta = numpy.zeros((1, len(X[0])))
    

def main():
    X = []
    y = []
    loadData(X, y, 'data.txt')
    plotData(X, y)
    # Add intercept data to X
    X = [[1.0] + x for x in X]
    theta = numpy.zeros((1, len(X[0])))
    theta = optimization(theta, X, y) 
    printArray(theta)
    accuracy(theta, X, y)
    plotDecisionBoundary(theta, X, y)
    testing(X, y)
    return;
    
if __name__ == "__main__":
    main()