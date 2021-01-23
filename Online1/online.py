from math import sqrt

file = open("dataset.txt")

lines = file.readlines()
numFeatures,numClass,datasetLen = 0,0,0
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

feature_average = []

for i in range(numFeatures):
    sum = 0
    for data in dataset:
        sum += data[i]
    average = sum / datasetLen    
    feature_average.append(average)

co_vars = []

for i in range(numFeatures):
    sum = 0
    for data in dataset:
        sum += (data[i] - feature_average[i])**2
    co_var = sum / datasetLen
    co_vars.append(co_var)

feature_stddv = [sqrt(x) for x in co_vars]

print('Feature Wise: ')
print(feature_average)
print(feature_stddv)

class_wise_dataset = []

classes =  set()

for data in dataset:
    classes.add(data[numFeatures])

for c in classes:
    d = []
    for data in dataset:
        if(data[numFeatures] == c):
            d.append(data)
    class_wise_dataset.append(d)

class_average = []

for c in classes:
    avg = []
    for i in range(numFeatures):
        sum = 0
        for cwd in class_wise_dataset[c-1]:
            sum += cwd[i]
        average = sum / len(class_wise_dataset[c-1])
        avg.append(average)
    class_average.append(avg)

classwise_std_dv = []

for c in classes:
    co_vars = []
    for i in range(numFeatures):
        sum = 0
        for c_w_d in class_wise_dataset[c-1]:
            sum += (c_w_d[i]-class_average[c-1][i])**2
        average = sum/len(class_wise_dataset[c-1])
        co_vars.append(average)
    co_vars = [sqrt(x) for x in co_vars]
    classwise_std_dv.append(co_vars)

print('Class Wise: ')
print(class_average)
print(classwise_std_dv)

