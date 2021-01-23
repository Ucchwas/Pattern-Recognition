from math import sqrt, exp

file = open("Train.txt")

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

print(dataset)

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

print('Class Wise: ')
print('Average')
print(class_average)

for c in classes:
    co_vars = []
    for i in range(numFeatures):
        sum = 0
        for c_w_d in class_wise_dataset[c-1]:
            sum += (c_w_d[i]-class_average[c-1][i])**2
        average = sum/len(class_wise_dataset[c-1])
        co_vars.append(average)
    std_dv = [sqrt(x) for x in co_vars]
    classwise_std_dv.append(std_dv)
print('Std Dv: ')
print(classwise_std_dv)

print('Prior Probabilty')
print(prior_probability)

