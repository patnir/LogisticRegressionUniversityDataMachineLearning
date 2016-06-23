# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 20:25:30 2016

@author: Rahul Patni
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math

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
    printArray(result)
    return result
    

def costFunction(X, y, theta):
    
    return;

def main():
    X = []
    y = []
    loadData(X, y, 'data.txt')
    plotData(X, y)
    # Add intercept data to X
    X = [[1.0] + x for x in X]
    theta = np.zeros((len(X[0]), 1))
    printArray(theta)
#    cost, grad = costFunction(X, y, theta)
#    printArray(cost)
#    printArray(grad)
    sigmoid(np.dot(X, theta))
    return;
    
if __name__ == "__main__":
    main()