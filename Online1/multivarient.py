from math import sqrt, exp
import numpy as np

#file = open("instructions and data\during coding/Train.txt")

file = open("train.txt")

lines = file.readlines()

numFeatures, numberClass, datasetLen = 0, 0, 0

dataset = []

count = 0
for line in lines:
    if(count == 0):
        var = line.split()
        numFeatures = int(var[0])
        numClass = int(var[1])
        datasetLen = int(var[2])
    else:
        var = line.split()
        size = len(var)
        data = []
        index = 0
        for i in var:
            if(index == size-1):
                data.append(int(i))
            else:
                data.append(float(i))
            index += 1
        dataset.append(data)

    count += 1

#print(dataset)

class_wise_dataset = []

classes = set()

for i in dataset:
    classes.add(i[numFeatures])

prior_probability = []

for c in classes:
    d = []
    count = 0
    for data in dataset:
        if(data[numFeatures] == c):
            d.append(data)
            count += 1
    class_wise_dataset.append(d)
    prior_probability.append(float(count/datasetLen))


class_average = []

for c in classes:
    avg = []
    for i in range(numFeatures):
        sum = 0
        for c_w_d in class_wise_dataset[c-1]:
            sum += c_w_d[i]
        average = sum/len(class_wise_dataset[c-1])
        avg.append(average)
    class_average.append(avg)

classwise_std_dv = []

#print('Class Wise: ')
#print('Average')
#print(class_average)


co_variance_matrix = []

for i in range(numFeatures):
    row = []
    for j in range(numFeatures):
        sum = 0
        for data in dataset:
            sum += (data[i] - avg[i]) * (data[j] - avg[j])
        row.append(float(sum/datasetLen))
    co_variance_matrix.append(row)

#print('Covariance Matrix: ')
#print(co_variance_matrix)

co_variance_matrix = np.matrix(co_variance_matrix)

#print(co_variance_matrix)

inv_co_variance_matrix = np.linalg.inv(co_variance_matrix)

#print('Inverse Covariance Matrix: ')
#print(inv_co_variance_matrix)

det = np.linalg.det(co_variance_matrix)

#print('Determinent: ')
#print(det)

#file = open("instructions and data\during coding\Test.txt")

file = open("test.txt")

linesT = file.readlines()

datasetT = []

for line in linesT:
        var = line.split()
        size =  len(var)
        index = 0
        data = []
        for i in var:
            if(index == size-1):
                data.append(int(i))
            else:
                data.append(float(i))
                index += 1
        datasetT.append(data)

#print("Test dataset: ")
#print(datasetT)

feature_vectors = []

for data in datasetT:
    p = data.pop()
    feature_vector = np.matrix(data)
    feature_vector = np.transpose(feature_vector)
    feature_vectors.append(feature_vector)
    data.append(p)

#print(feature_vectors)

output = []


for test_data in feature_vectors:
    probabilty = []
    for c in range(numClass):
        xt = test_data - np.matrix(class_average[c]).transpose()
        xt = np.matrix(xt)
        #print(xt)
        n = float(1 /( ((2*3.1416)**float(numFeatures/2))*(det)**(.5)))
        p = n*exp((-.5)*xt.transpose()*inv_co_variance_matrix*xt)
        px = p*prior_probability[c]
        probabilty.append(px)
    output.append(probabilty)


print("Output: ")
#print(output)

accurate = 0

#feature_vectors = np.array(feature_vectors)

count = 0 
c = 0
for out in output:
    if datasetT[count][numFeatures] == out.index(max(out))+1:
        accurate += 1
        #print(datasetT[count][numFeatures])
    else:
        if(c == 0):
            print("SampleNo , FeatureValue , ActualClass , EstimatedClass")
            c = 1
        else:
            print(count+1, feature_vectors[count], datasetT[count][numFeatures], out.index(max(out))+1)
    count += 1

print('Accuracy : ',float((accurate/len(datasetT)))*100)