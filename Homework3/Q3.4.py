import threading
import time

MAX = 1000000
MIN = 2
lock = threading.Lock()
primeList = [True]*(MAX+1)

def isPrime(MAX):
    global primeList
    startNum = 2
    lock.acquire()
    while startNum*startNum <= MAX: 
        if primeList[startNum] == True:  
            mult = startNum * 2
            while mult <= MAX: 
                primeList[mult] = False
                mult += startNum 
        startNum += 1 
    lock.release()
    
def sumPrime(start,end,primeList):
    global total
    for i in range (start,end+1):
        if(primeList[i]==True):
            lock.acquire()
            total = total + i
            lock.release()

total = 0


## Threading Implementation ##
start = time.time()
t1 = threading.Thread(target=isPrime, args=(MAX,), name='Thread_1')
t1.start()
t1.join()
t2 = threading.Thread(target=sumPrime, args=(MIN,100000,primeList), name='Thread_2') 
t3 = threading.Thread(target=sumPrime, args=(100001,200000,primeList), name='Thread_3') 
t4 = threading.Thread(target=sumPrime, args=(200001,300000,primeList), name='Thread_4') 
t5 = threading.Thread(target=sumPrime, args=(300001,400000,primeList), name='Thread_5')
t6 = threading.Thread(target=sumPrime, args=(400001,500000,primeList), name='Thread_6')
t7 = threading.Thread(target=sumPrime, args=(500001,600000,primeList), name='Thread_7') 
t8 = threading.Thread(target=sumPrime, args=(600001,700000,primeList), name='Thread_8') 
t9 = threading.Thread(target=sumPrime, args=(700001,800000,primeList), name='Thread_9')
t10 = threading.Thread(target=sumPrime, args=(800001,900000,primeList), name='Thread_10') 
t11 = threading.Thread(target=sumPrime, args=(900001,MAX,primeList), name='Thread_11') 
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t7.start()
t8.start()
t9.start()
t10.start()
t11.start()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()
t7.join()
t8.join()
t9.join()
t10.join()
t11.join()

end = time.time()
print("Time: %.5f s"%(end-start))
print(total)


### Normal Implementation ###

primeList = [True]*(MAX+1)
startNum = 2
start = time.time()

while startNum*startNum <= MAX: 
    if primeList[startNum] == True:  
        mult = startNum * 2
        while mult <= MAX: 
            primeList[mult] = False
            mult += startNum 
    startNum += 1 
total = 0
for i in range (2, 1000000 + 1): 
    if(primeList[i]): 
        total += i 
end = time.time()
print("Time: %.5f s"%(end-start))
print(total)