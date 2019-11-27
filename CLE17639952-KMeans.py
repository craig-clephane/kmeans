#Author: Craig Clephane
#Last Edited: 23/11/2019
#Python Version 3.7.4

#Imported Librarys to perform the required tasks 
import csv 
import matplotlib.pyplot as plt
import math
import numpy as numpy
import pandas as pd
import random 

#Global Variables which are used throughout the program 
centroids = []
previousCentroids = []
data = 'CMP3751M_CMP9772M_ML_Assignment_1_Task_2_Dataset.csv'
data = pd.read_csv(data)
df = numpy.array(data)
close = False
dataclusters = []
distance = []

#initialise centroids in a random position based on boundaries
def initialise_centroids(dataset, k): 
    global centroids
    if k is 2:
        centroids = [[random.uniform(4.5,5.5), random.uniform(1.5,2.5), random.uniform(1.5,5.5),random.uniform(0.2,4.5)], [random.uniform(6.5,8.5), random.uniform(4,6), random.uniform(1.5,5.5), random.uniform(0.2,4.5)]]
    if k is 3: 
        centroids = [[random.uniform(4.5,5.5), random.uniform(1.5,2.5), random.uniform(1.5,5.5),random.uniform(0.2,4.5)], [random.uniform(5.5,7.5), random.uniform(2.5,4), random.uniform(1.5,5.5),random.uniform(0.2,4.5)], [random.uniform(6.5,8.5), random.uniform(4,6), random.uniform(1.5,5.5),random.uniform(0.2,4.5)]]


#Returns the distance of the values from one vector of variables to another, 
#this is used for two purposes one to find the distance between a position 
# nd a centroid and another to handle a portion of the calculation for the objective function. 
def compute_euclidean_distance(vec_1, vec_2):
    lengthofvector = len(vec_1)
    total = 0
    for i in range(lengthofvector):
        total = total + (vec_1[i] - vec_2[i])**2
    distance = math.sqrt(total)
    distance = round(distance, 3)
    #print(" "+ str(vec_1) + " " + str(vec_2) + " : " + str(distance))      #PRINT VALUES
    return distance
    
#Kmeans function calls upon the compute Euclidean distance function and 
#calculates the closest centroid to the given data point. 
#By determining which centroid is the closest, the function 
#will return an index value which is the closest centroid. 
#Other returned values include the value in which is closest, 
#this is used for an objective function.
def kmeans(data, k):
    global distance
    distanceMatrix = []
    for i in range(k):
        dis = compute_euclidean_distance(data, centroids[i])
        distanceMatrix.append(dis)
    cluster_assigned = distanceMatrix.index(min(distanceMatrix))
    #print(" " + str(data) + " : " + str(cluster_assigned) + "\n")      #PRINT VALUES
    return cluster_assigned, distanceMatrix[cluster_assigned]

#A function which displays colour co-ordinated clusters, this is done 
#by iteratively cycling through each data point and determining the centroid 
#that it has been assigned to.  
def display_plot(k, y):
    plt.xlabel('Height')        #Change depending on which values you are displaying 
    plt.ylabel('Leg Length')
    for cluster in dataclusters:
        plt.scatter([item[0] for item in cluster], [item[y] for item in cluster])    
    cen = pd.DataFrame(centroids)
    plt.scatter(cen[0], cen[y], s = 100, alpha = 0.9, linewidths=1)
    plt.show()

#A termination function which measures the distance between the previous centroid 
#location and the new centroid location. If the centroid has moved little, end the 
#iteration. This is in place of the objective function.
def termination(i, j):
    global close
    k = i - j
    k = round(k, 2)
    k = abs(k)
    if (k < 0.03):
        close = True
    print("Change : " + str(k))

#A function which uses calculated information, such as which centroid is closest to the data point.
# A mean of the data points assigned to a centroid are calculated, this is then returned as an array of new centroid locations. . 
def newCentroid(clusterAssigned, y):
    global dataclusters
    global centroids 
    global previousCentroids
    centroidNumber = 0
    iteration = 0
    newCentroid = []
    dataclusterpoints = [[] for i in range(len (df[1:]))]

    #move through each centroid
    for row in centroids:
        numberOfInstances = 0 
        dataNumber = 0
        mean = [0.0,0.0,0.0,0.0]
        #For each row in the dataframe
        for row in df[1:]:
            clusterNo = clusterAssigned[dataNumber]
            dataNumber = dataNumber + 1
            #If the cluster number is equal to which the data has been assigned, append the data to the corrisponding data array and 
            #cappend on the mean variable
            if clusterNo is centroidNumber:
                iteration = 0
                dataclusterpoints[clusterNo].append( row )
                for i in mean:
                    mean[iteration] = mean[iteration] + row[iteration]
                    iteration = iteration + 1
                numberOfInstances = numberOfInstances + 1
        #Calculate the mean for each cluster
        for i in range(len(mean)):
            #print(" " + str(mean[i]) + " / " + str(numberOfInstances))     #PRINT MEAN
            mean[i] = mean[i] / numberOfInstances
        for i in range(len(mean)):
            mean[i] = round(mean[i], 3)
        newCentroid.append(mean)
        centroidNumber = centroidNumber + 1
    previousCentroids = centroids
    centroids = newCentroid
    dataclusters = dataclusterpoints
    termination(sum(sum(x) for x in previousCentroids), sum(sum(x) for x in centroids))
    #print("New centroids: " + str(newCentroid))        #PRINT CENTROID LOCATIONS
    return newCentroid

#The objective function is responsible for minimising the error between 
#the centroids and the assigned data points. This is done by determining 
#the SSE of each data point with the assigned centroid. This function 
#will return a value which is then plotted within another function. 
def objective_function(value, clusterAssigned):
    summation = 0
    loop = 0
    for item in df:
        cen = clusterAssigned[loop]
        value = (compute_euclidean_distance(item, centroids[cen]))
        summation = summation + value 
        loop = loop + 1
    return summation

#A function which displays a downward plot of evaluated errors between iterations. 
def objective_function_display(aggList):
    plt.xlabel('iteration Step')
    plt.ylabel('Objective Function')
    plt.plot(aggList)
    plt.show()

#The main function loops continuously until the termination function 
#returns True. This function is responsible for calling other functions 
#such as Kmeans and the objective function to display results to the user. 
def main():
    k = 2       #CHANGEABLE NUMBER OF CLUSTERS (TWO OR THREE)
    y = 2       #CHANGEABLE COLUMN VALUE, (TWO OR THREE)
    looped = 0
    aggList = []

    initialise_centroids(data, k)

    while close is False:
        valueList = []
        clusterAssigned = []
        for item in df:
            clust, value = kmeans(item, k)
            clusterAssigned.append(clust)
            valueList.append(value)
        print(clusterAssigned)
        newCentroid(clusterAssigned, y)
        sumed = objective_function(valueList, clusterAssigned)
        aggList.append(sumed)
        looped = looped + 1
    #print("Times kmeans looped : " + str(looped))      #PRINTABLE ITERATION VALUE
    display_plot(k, y)
    objective_function_display(aggList)
    print(distance)
    
main()