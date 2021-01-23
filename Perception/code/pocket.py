import numpy as np

file = open("trainLinearlyNonSeparable.txt")

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

file = open("testLinearlyNonSeparable.txt")

linesT = file.readlines()

test_dataset = []

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
        test_dataset.append(data)
#print(test_dataset)

def test(dataset,w):
    count = 0
    for i in dataset:
        x = np.array(i)
        feature = x[numFeatures]
        x[numFeatures] = 1
        x = np.array(x)
        x = x.reshape(numFeatures+1,1)
        dot_product = np.dot(w,x)
        p = -1
        if(dot_product>=0):
            p = 1
        else:
            p = 2
        if(p == feature):
            count += 1
    return float(count/len(dataset))


learning_rate = 0.03
np.random.seed(41)
w = np.random.random_sample(((numFeatures+1),))
ws = w
missClass = test(dataset,w)*len(dataset)


for i in range(1000):
    Y = []
    delta = []
    count = 0
    for i in range(datasetLen):
        x = np.array(dataset[i])
        feature = x[numFeatures]
        x[numFeatures] = 1
        x = x.reshape(numFeatures+1,1)
        dot_product = np.dot(w,x)
        if(feature == 2 and dot_product>0):
            Y.append(x)
            delta.append(1)
            count += 1
        elif(feature == 1 and dot_product<0):
            Y.append(x)
            delta.append(-1)
            count += 1
    sum = np.zeros(numFeatures+1)
    for i in range(len(Y)):
        sum += delta[i]*Y[i].transpose()[0]

    w = w-learning_rate*sum

    if(count<missClass):
        missClass = count
        ws = w

    if(count == 0):
        break
#print(ws)

print('sample no', 'feature values' ,'actual class' ,'predicted class')

count = 0

for i in range(len(test_dataset)):
    x = np.array(test_dataset[i])
    feature = x[numFeatures]
    x[numFeatures] = 1
    x = np.array(x)
    x = x.reshape(numFeatures+1,1)
    dot_product = np.dot(w,x)
    p = -1
    if(dot_product>=0):
        p = 1
    else:
        p = 2
    if(p == feature):
        count += 1
    feature_values = np.array(test_dataset[i])
    print(i+1,feature_values,p)

#print(count)
print("Accuracy :",float((count/len(dataset))*100))


























