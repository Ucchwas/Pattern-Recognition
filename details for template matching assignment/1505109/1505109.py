import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

count = 0

p = 50

inp = cv2.VideoCapture('input.mov')
ref = cv2.imread('reference.jpg')

ret, frame = inp.read()


def Exhaustive_search(frame,reference):
    refx = reference.shape[0]
    refy = reference.shape[1]
    #print(refx)
    #print(refy)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #print(frame.shape)
    reference = cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)
    #print(reference)
    c = np.zeros((frame.shape[0]-refx+1,frame.shape[1]-refy+1))
    #print(c)
    for i in range(frame.shape[0]-refx+1):
        for j in range(frame.shape[1]-refy+1):
            c[i,j] = np.sum(reference.astype(int) * frame[i:i+refx, j:j+refy].astype(int))/\
                (np.linalg.norm(reference) * np.linalg.norm(frame[i:i+refx, j:j+refy]))
            #print(c[i,j])
    #print(c.shape)
    # global count
    # count += 1
    
    temp =  np.unravel_index(np.argmax(c, axis=None), c.shape)
    
    ex = c.shape[0]*temp[0] + temp[1]
    global count
    count += ex
    #print(temp)
    
    return int(temp[0]+refx/2), int(temp[1]+refy/2) 

x, y = Exhaustive_search(frame, ref)

#count = 0

def twoDLogSearch(frame, reference):
    best = -math.inf
    refx = reference.shape[0]
    refy = reference.shape[1]

    framex = frame.shape[0]
    framey = frame.shape[1]

    l = int(framey/4)
    x, y = int(framex/2), int(framey/2)

    while(True):
        for i in range(-1, 2):
            for j in range(-1, 2):
                #print(i,j)
                #print(x+i*l, y+j*l)
                newx = x + i * l - int(refx / 2)
                newy = y + j * l - int(refy / 2)
                if newx >= 0 and newy >= 0 and newx+refx <= framex and newy+refy <= framey:
                    #print(newx,newy)
                    temp = np.sum(reference.astype(int) * frame[newx:newx+refx, newy:newy+refy].astype(int))/\
                        (np.linalg.norm(reference) * np.linalg.norm(frame[newx:newx+refx, newy:newy+refy]))
                    if temp > best:
                        best = temp
                        arg_best = i, j
                        #print(arg_best)
        global count
        count += 1
        #print(count)
        x, y = x + arg_best[0]*l, y + arg_best[1]*l
        #print(x,y)
        l = int(l/2)
        if l < 1: 
            break
    return x + arg_best[0]*l*2, y + arg_best[1]*l*2

#print(count)
res = cv2.VideoWriter('output.mov', cv2.VideoWriter_fourcc(*'XVID'), inp.get(cv2.CAP_PROP_FPS),\
                      (int(inp.get(cv2.CAP_PROP_FRAME_WIDTH)),int(inp.get(cv2.CAP_PROP_FRAME_HEIGHT))))

def run_algo(frame, ref, x, y, p, algo):
    threshold = lambda x : 0 if x < 0 else x
    newx,newy = algo(frame[threshold(x - p): threshold(x + p), threshold(y - p): threshold(y + p)], ref)
    return threshold(x - p) + newx, threshold(y - p) + newy


frame_count = 1

while inp.isOpened():

    ret, frame = inp.read()
    
    if ret == True:
        x, y = run_algo(frame, ref, x, y, p, Exhaustive_search)
        print(x,y)
        frame = cv2.rectangle(frame,(int(y  - ref.shape[1]/2), int(x - ref.shape[0]/2)), \
                          (int(y + ref.shape[1]/2), int(x + ref.shape[0]/2)), (0, 0, 255), 3)

        res.write(frame)
        frame_count += 1
    
    else: 
        break

#print(count)
#print(frame_count)


#print('Value', count/frame_count)

inp.release()
res.release()








