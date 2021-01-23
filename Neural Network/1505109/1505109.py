import numpy as np
from sklearn import preprocessing
from scipy.spatial import distance

def split(s, delim):
    words = []
    word = []
    for c in s:
        if c not in delim:
            word.append(c)
        else:
            if word:
                words.append(''.join(word))
                word = []
    if word:
        words.append(''.join(word))
    return words

def loadfile(filename):
    file = open(filename, "r")
    rows = list()
    for line in file:
        vals = split(line, [' ' ,'\t', '\n'])
        rows.append(vals)
    return rows

train=loadfile('trainNN.txt')

test=loadfile('testNN.txt')

#print(train)

min_max_scaler = preprocessing.StandardScaler()


train = np.array(train)
train = train.astype(np.float)
train_output_col = train[:,-1].copy()
#print(train_output_col)
train_output_col = train_output_col.astype(np.int)
train_output_col = np.array(train_output_col)
train_output_col = np.array(train_output_col).tolist()
#print(train_output_col)

train_Y = train[:,-1].copy()
train_Y = np.array(train_Y)
train_Y = np.eye(np.max(train_Y).astype(int))[train_Y.astype(int)-1]
#print(train_Y)

train = min_max_scaler.fit_transform(train)
train[:,-1] = np.ones((train.shape[0]))

#print(train[:,:-1])

test = np.array(test)
test = test.astype(np.float)
test_output_col= test[:,-1].copy()
#print(train_output_col)
test_output_col = test_output_col.astype(np.int)
test_output_col = np.array(test_output_col)
test_output_col = np.array(test_output_col).tolist()
#print(train_output_col)

test_Y= test[:,-1].copy()
test_Y= np.array(test_Y)
test_Y = np.eye(np.max(test_Y).astype(int))[test_Y.astype(int)-1]
#print(train_Y)

test = min_max_scaler.fit_transform(test)
test[:,-1]=np.ones((test.shape[0]))

#print(test)


def sigmoid(x):
    return 1 / (1 + np.exp(-1*x))

def sigmoid_der(x):
    return 1*sigmoid(x)*(1-sigmoid(x))

train = train[:,:-1]
test = test[:,:-1]

input_layer_neurons = train.shape[1]
output_layer_neurons = test.shape[1]
#print(input_layer_neurons)
#print(output_layer_neurons)
L = 4
k = [input_layer_neurons,5,5,5,output_layer_neurons]
learning_rate = 0.3
Max_epoch = 1000
N = len(train)
threshold = 10
min_error = np.inf
best_W = []
#print(k[1:])

W = []
for i in range(len(k)-1):
    w = np.random.uniform(0,1,(k[i+1],k[i]+1))
    #print(w)
    W.append(w)
#print(W)

for epoch in range(Max_epoch):
    y_m = []
    correct = 0
    for i in range(N):
        v = []
        y = []
        input_neuron = [1]
        var = np.array(train[i])
        input_neuron.extend(var)
        input_neuron = np.array(input_neuron)
        temp = input_neuron
        

        for r in range(len(k)-1):
            w = W[r]
            var = np.dot(w,input_neuron)
            v.append(var)
            var = sigmoid(var)
            out = [1]
            out.extend(var)
            input_neuron = np.array(out)
            y.append(input_neuron) 
            #print(y)

        y_m.append(y[L-1][1:])

        #print(np.argmax(y[L-1][1:]),train_output_col[i]-1,np.argmax(train_Y[i]))
        #print(y_m)

        lastY = len(y)-1
        error = 0.5* (y[lastY][1:]-train_Y[i])*(y[lastY][1:]-train_Y[i])
        if error.sum() < min_error:
            min_error = error.sum()
            best_W = W
        #print(y[lastY][1:])
        deltaL = 0
        ei = y[lastY][1:]-train_Y[i]
        #print(v)
        deltaL = np.multiply(ei,sigmoid_der(v[lastY]))
        #print(deltaL)
        delta = [] 
        
        for i in range(L-1):
            delta.append([0]*k[i+1]) 
        
        delta.append(deltaL)
        
        #print(delta)

        for r in reversed(range(1,L,1)):
            
            for j in range(0,k[r]):
                #print('j kr',j,k[r])
                ej = 0
                for n in range(0,k[r+1]):
                    ej += delta[r][n]*W[r][n][j+1]
                #print('ej:',ej)
                delta[r-1][j] = ej*sigmoid_der(v[r-1][j])
                #print('delta:',delta[r-1][j])   
        #print(delta)

        for r in range(0,L):
            for j in range(0,k[r+1]):
                del_w = []
                if r == 0:
                    del_w = np.multiply(-learning_rate*delta[r][j],temp)
                else:
                    del_w = np.multiply(-learning_rate*delta[r][j],y[r-1])
                W[r][j] = W[r][j] + np.array(del_w)
        #print(W)
    #cost_func
    # 
    #print(correct/N)   
    sum = 0
    for i in range(N):
        #print('y_m: ',y_m[i].shape[0])
        #print('train_Y: ',train_Y[i].shape[0])
        for j in range(train_Y[i].shape[0]):
            #print(train_Y[i])
            euclidean_dis = distance.euclidean(y_m[i],train_Y[i])
            sum += euclidean_dis
    print(sum)
    #res = epoch
    if(sum < threshold):
        break
print(epoch+1)
W = best_W
output = []
count1 =0 
for i in range(N):
        v = []
        y = []
        input_neuron = [1]
        var = np.array(train[i])
        input_neuron.extend(var)
        input_neuron = np.array(input_neuron)

        for r in range(len(k)-1):
            w = W[r]
            var = np.dot(w,input_neuron)
            v.append(var)
            var = sigmoid(var)
            out = [1]
            out.extend(var)
            input_neuron = np.array(out)
            y.append(input_neuron)
    
        if np.argmax(y[L-1][1:]) == np.argmax(train_Y[i]):
            count1 += 1



output = []
count = 0
for i in range(N):
        v = []
        y = []
        output_neuron = [1]
        var = np.array(test[i])
        output_neuron.extend(var)
        output_neuron = np.array(output_neuron)

        for r in range(len(k)-1):
            w = W[r]
            var = np.dot(w,output_neuron)
            v.append(var)
            var = sigmoid(var)
            out = [1]
            out.extend(var)
            output_neuron = np.array(out)
            y.append(output_neuron)

        print(i+1,test[i],np.argmax(test_Y[i]),np.argmax(y[L-1][1:]))

        if np.argmax(y[L-1][1:]) == np.argmax(test_Y[i]):
            count += 1


print("Accuracy on Train Data set:   "+str((count1/N)*100))

print("Accuracy is on Test Data Set:   "+str((count/N)*100))





















        















