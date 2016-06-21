# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 20:25:30 2016

@author: Rahul Patni
"""
import matplotlib.pyplot as plt

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
    column1 = extractColummnFromMatrix(X, 0)
    column2 = extractColummnFromMatrix(X, 1)
    

def main():
    X = []
    y = []
    loadData(X, y, 'data.txt')
    plotData(X, y)
    return;
    
if __name__ == "__main__":
    main()