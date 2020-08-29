# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 00:19:19 2020

@author: SiddarthaThentu, RekhaRani

"""
import os
# cmd = "pip install pandas"
# os.system(cmd)

import pandas as pd
import timeit
import time
import math

#Record the time when the program starts

def readFile(fileName,numOfItems):
    
    #read the data from the csv file into a pandas dataframe
    data = pd.read_csv(fileName) 
    #initialize an array to store respective weights of all items
    weightsOfItems=[]
    #initialize an array to store respective values of all items
    valuesOfItems = []
    #initialize an array to store respective value/weight ratio of all items
    valueWeightRatio = []
    #Load weights of items from dataframe
    weightsOfItems = data['weight'].tolist()
    #Load values of items from dataframe
    valuesOfItems = data['Value'].tolist()
    
    weightsOfItems = weightsOfItems[:numOfItems]
    valuesOfItems = valuesOfItems[:numOfItems]
    
    for index in range(numOfItems):
        valueWeightRatio.append(valuesOfItems[index]/weightsOfItems[index])
    #Initialize number of items    
    #return the properties of knapsack
    return valuesOfItems, weightsOfItems,valueWeightRatio


def dynamicKnapsack(numOfItems,knapsackMaxCapacity,values,weights):
    
    #create a 2-dimensional table of number of items and knapsack capacity
    #start = timeit.default_timer()
    start = time.time()
    #print(values)
    dynamicTable = [[0 for x in range(knapsackMaxCapacity+1)] for x in range(numOfItems+1)] 
    #For each item of the total items
    for index in range(numOfItems + 1):
      #for each knapsack capacity
      for eachWeight in range(knapsackMaxCapacity + 1):
         #initialize first column and row to zeros
         if index == 0 or eachWeight == 0:
            dynamicTable[index][eachWeight] = 0
         #There are two cases
         #Either pick the nth item or do not pick the nth item
         elif weights[index-1] <= eachWeight:
            dynamicTable[index][eachWeight] = max(values[index-1] + dynamicTable[index-1][eachWeight-weights[index-1]], dynamicTable[index-1][eachWeight])
         #if nth item is heavier than Capacity, the solution would be the solution of (n-1)
         else:
            dynamicTable[index][eachWeight] = dynamicTable[index-1][eachWeight]
    #The last element of the table will store the max value
    stop = time.time()
    timeTaken = stop-start
    #print(type(timeTaken))
    print("time taken for Dynamic = ",(stop-start)," ms")
    return dynamicTable[numOfItems][knapsackMaxCapacity]

from queue import Queue
import pandas as pd

class Node:
    def __init__(self):
        self.level = None
        self.value = None
        self.bound = None
        self.weight = None

    def __str__(self):
        return "Level: %s Profit: %s Bound: %s Weight: %s" % (self.level, self.profit, self.bound, self.weight)


def bound(node, n, W, items):
    if(node.weight >= W):
        return 0

    boundValue = int(node.profit)
    j = node.level + 1
    totweight = int(node.weight)

    while ((j < n) and (totweight + items[j].weight) <= W):
        totweight += items[j].weight
        boundValue += items[j].value
        j += 1

    if(j < n):
        boundValue += (W - totweight) * items[j].value / float(items[j].weight)

    return boundValue

Que = Queue()

def KnapSackBranchNBound(weight, items, total_items):
    items = sorted(items, key=lambda x: x.value/float(x.weight), reverse=True)

    curr = Node()
    temp = Node()

    curr.level = -1
    curr.value = 0
    curr.weight = 0

    Que.put(curr)
    maxProfit = 0;

    while not Que.empty():
        curr = Que.get()
        temp = Node()                                  # Added line
        if curr.level == -1:
            temp.level = 0

        if curr.level == total_items - 1:
            continue

        temp.level = curr.level + 1
        temp.weight = curr.weight + items[temp.level].weight
        temp.value = curr.value + items[temp.level].value
        if (temp.weight <= weight and temp.value > maxProfit):
            maxProfit = temp.value;

        temp.bound = bound(temp, total_items, weight, items)
        if (temp.bound > maxProfit):
            Que.put(temp)

        temp = Node()                                  # Added line
        temp.level = curr.level + 1                       # Added line
        temp.weight = curr.weight
        temp.value = curr.value
        temp.bound = bound(temp, total_items, weight, items)
        if (temp.bound > maxProfit):
            # print(items[v.level])
            Que.put(temp)

    return maxProfit

def fptasKanpsack(numOfItems,knapsackMaxCapacity,values,weights,epsilon):

    #create a 2-dimensional table of number of items and knapsack capacity
    #start = timeit.default_timer()
    maxVal = max(values)
    
    scalingFactor  = (maxVal * epsilon)/numOfItems
    for i in range(len(values)):
        values[i] = math.floor(values[i]/scalingFactor)
    start = time.time()
    
    dynamicTable = [[0 for x in range(knapsackMaxCapacity+1)] for x in range(numOfItems+1)] 
    #For each item of the total items
    for index in range(numOfItems + 1):
      #for each knapsack capacity
      for eachWeight in range(knapsackMaxCapacity + 1):
         #initialize first column and row to zeros
         if index == 0 or eachWeight == 0:
            dynamicTable[index][eachWeight] = 0
         #There are two cases
         #Either pick the nth item or do not pick the nth item
         elif weights[index-1] <= eachWeight:
            dynamicTable[index][eachWeight] = max(values[index-1] + dynamicTable[index-1][eachWeight-weights[index-1]], dynamicTable[index-1][eachWeight])
         #if nth item is heavier than Capacity, the solution would be the solution of (n-1)
         else:
            dynamicTable[index][eachWeight] = dynamicTable[index-1][eachWeight]
    #The last element of the table will store the max value
    stop = time.time()
    print("time taken for fptas = ",(stop-start)," ms")
    return scalingFactor*dynamicTable[numOfItems][knapsackMaxCapacity]
    
def bruteKnapsack(knapsackMaxCapacity,weights,values,numOfItems) -> int:

    # Base Case 
    if numOfItems == 0 or knapsackMaxCapacity == 0 : 
        return 0
  
    #if weight is more than capacity, cannot include
    if (weights[numOfItems-1] > knapsackMaxCapacity): 
          return bruteKnapsack(knapsackMaxCapacity , weights , values , numOfItems-1) 
    #maximum of two case
    else: 
        value =  max(values[numOfItems-1] + bruteKnapsack(knapsackMaxCapacity-weights[numOfItems-1] , weights , values , numOfItems-1), 
                    bruteKnapsack(knapsackMaxCapacity , weights , values , numOfItems-1))  

    return value

def greedyKnapsack(knapsackCapacity,valuesOfItems,weightsOfItems,valueWeightRatio):
    
    #record the method run time
    start = time.time()
    
    knapsack = []
    #inititliaze the current knapsack weights and values to zero
    knapsackWeight = 0
    knapsackValue = 0

    while(knapsackWeight <= knapsackCapacity):

        maxItem = max(valueWeightRatio)

        indexOfMaxItem = valueWeightRatio.index(maxItem)
        
        #keep adding the highest profitable valu/weight ratio element
        if weightsOfItems[indexOfMaxItem]+ knapsackWeight <= knapsackCapacity:
            knapsack.append(indexOfMaxItem+1)
            knapsackWeight += weightsOfItems[indexOfMaxItem]
            knapsackValue += valuesOfItems[indexOfMaxItem]
            valueWeightRatio[indexOfMaxItem] = -1
        else:
            break
        
    stop = time.time()
        
    print("time taken for greedy = ",(stop-start)," ms")
    
    return knapsackValue

    
def printValue(value):
    #print the maximum value of the items loaded in knapsack
    print("Value of items in the knapsack =",value)
    
def printAllValues(bruteValue,branchValue,dpValue,greedyValue,fptasValue):
    print("\n")
    #print all the values of the methods
    print("Value of items for Greedy =",greedyValue)
    print("Value of items for Bruteforce =",bruteValue)
    print("Value of items for Dynamic =",dpValue)
    print("Value of items for FPTAS =",fptasValue)
    print("Value of items for Branch & Bound =",branchValue)


#driver code
print("\nWelcome to the knapsack program")
print("Loading data")

fileName = 'csv.csv'
setValue = True
knapsackMaxCapacity = int(input("Please enter maximum kapacity of knapsack :"))

while(setValue):   
    
    numOfItems = int(input("Please enter the number of items (<=50):"))
    if numOfItems>50:
        print("Error, too big a dataset. You can select upto 50 items.")
        print("Note this contraint is because brute force and branch and bound algorithms take long to run on bigger datasets")
        print("If you are planning to run other approaches, you can include upto 50000.")
        continue
    #load parameters of knapsack
    valuesOfItems, weightsOfItems, valueWeightRatio = readFile(fileName,numOfItems)
    
    from collections import namedtuple
    Item = namedtuple("Item", ['index', 'value', 'weight'])
    items = []
    for i in range(0,numOfItems):
        items.append(Item(i-1, int(valuesOfItems[i]), float(weightsOfItems[i])))
    epsilon = 0.5

    print("Data loaded. Number of items in the dataset: ",numOfItems)
    print("Which approach do you want to choose?")    
    print("1: Brute Force 2: Branch & Bound 3: Greedy approach 4: Dynamic Programming 5: FPTAS 6: ALL")
    option = input("Please enter your option: ")

    if option == '4':
        print("\nRunning Dynamic Approach")
        dpValue = dynamicKnapsack(numOfItems,knapsackMaxCapacity,valuesOfItems,weightsOfItems)
        printValue(dpValue)
        
    #Brute Force
    if option == '1':
        print("\nRunning Brute Approach")
        start = timeit.default_timer()
        bruteValue = bruteKnapsack(knapsackMaxCapacity,valuesOfItems,weightsOfItems,numOfItems)
        stop = timeit.default_timer()
        print("time taken for brute = ",(stop-start)) 
        printValue(bruteValue)
        
    #BranchNBound
    if option == '2':
        print("\nRunning Branch and Bound Approach")
        start = timeit.default_timer()
        kbb = KnapSackBranchNBound(knapsackMaxCapacity,items, numOfItems)
        stop = timeit.default_timer()
        print("time taken for branch & bound = ",(stop-start)) 
        printValue(kbb)
        
    #Greedy    
    if option == '3':
        print("\nRunning Greedy Approach")
        greedyValue = greedyKnapsack(knapsackMaxCapacity,valuesOfItems,weightsOfItems,valueWeightRatio)
        printValue(greedyValue)
    
    #All approaches
    if option == '6':
        print("\nRunning All approaches\n")
        dpValue = dynamicKnapsack(numOfItems,knapsackMaxCapacity,valuesOfItems,weightsOfItems)
        start = time.time()
        bruteValue = bruteKnapsack(knapsackMaxCapacity,weightsOfItems,valuesOfItems,numOfItems)
        stop = time.time()
        print("time taken for brute force = ",(stop-start)," ms")
        start = time.time()
        branchValue = KnapSackBranchNBound(knapsackMaxCapacity,items, numOfItems)
        stop = time.time()
        print("time taken for branch and bound = ",(stop-start)," ms")
        greedyValue = greedyKnapsack(knapsackMaxCapacity,valuesOfItems,weightsOfItems,valueWeightRatio)
        fptasValue = fptasKanpsack(numOfItems,knapsackMaxCapacity,valuesOfItems,weightsOfItems,epsilon)
        printAllValues(bruteValue,branchValue,dpValue,greedyValue,fptasValue)
     
    #FPTAS
    if option == '5':
        print("\nRunning FPTAS")
        fptasValue = fptasKanpsack(numOfItems,knapsackMaxCapacity,valuesOfItems,weightsOfItems,epsilon)
        printValue(fptasValue)
        
    print("\nDo you want to try again with different number of items? ")
    
    while(True):
        option2 = input("Press 'y' or 'n' : ")
        if option2=='n':
            print("\nThankyou. Terminating")
            setValue = False
            break
        elif option2 == 'y':
            break
        else:
            print("Please enter a valid option")
    



    


