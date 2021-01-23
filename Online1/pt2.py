from math import sqrt, exp
import numpy as np

file = open("Train.txt")

numOfClass, numOfFeatures, datasetLength = 0,0,0

lines = file.readlines()

dataset = []

count = 0

for line in lines:
    if(count == 0):
        var = line.split()
        numOfFeatures = int(var[0])
        numOfClass = int(var[1])
        datasetLength = int(var[2])
    else:
        var = line.split()
        data = []
        for i in var:
            data.append(float(i))
        dataset.append(data)
    count = count + 1
    
print(numOfClass, numOfFeatures, datasetLength)
print(dataset)
print(count)

avg = []

for i in range(numOfFeatures):
    sum = 0
    for data in dataset:
        sum = sum + data[i] 
    avg.append(float(sum/datasetLength))
    
print(avg)

co_variance = []

for i in range(numOfFeatures):
    sum = 0
    for data in dataset:
        sum = sum + (data[i] - avg[i])**2
    co_variance.append(float(sum/datasetLength))

print(co_variance)

std_dev = []

for i in range(numOfFeatures):
    std_dev.append(sqrt(co_variance[i]))

print(std_dev)

co_variance_matrix = []

for i in range(numOfFeatures):
    row = []
    for j in range(numOfFeatures):
        sum = 0
        for data in dataset:
            sum = sum + (data[i] - avg[i]) * (data[j] - avg[j])
        row.append(float(sum/datasetLength))
    co_variance_matrix.append(row)

print(co_variance_matrix)

co_variance_matrix = np.matrix(co_variance_matrix)

print(co_variance_matrix)

inv_co_variance_matrix = np.linalg.inv(co_variance_matrix)

#print(inv_co_variance_matrix)

feature_vectors = []

for data in dataset:
    data.pop()
    feature_vector = np.matrix(data)
    feature_vector = np.transpose(feature_vector)
    feature_vectors.append(feature_vector)
    
#print(feature_vectors)
    
average = np.matrix(avg).transpose()

det = np.linalg.det(co_variance_matrix)

print(det)

normal_distribution = []

for data in feature_vectors:
    x = np.matrix(data - average)
    val = (1/(sqrt(((2*3.1416)**(float(numOfFeatures/2))))*sqrt(det)))*(exp(-0.5*(x.transpose()*(inv_co_variance_matrix*x))))
    normal_distribution.append(val)
    
print(normal_distribution)

    
    
    
    
    
    
    
    
    
    




