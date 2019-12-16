import multiprocessing as mp
import threading
import time

### Original Implementation Target Function ###
def add():
    global var
    var = var + 1
    
    
## Original Implementation
print("\n### Original Implementation ###")
var = 0
start = time.time()
for i in range(50000000):
    add()
end = time.time()
print("Time: %.5f s"%(end-start))
print("Final var Value: %i "%var)


### Threading Implementation Target Function ###

lock = threading.Lock()
def addT(numThread):
    global varT
    lock.acquire()
    for i in range(int(50000000/numThread)):
        varT = varT+1  
    lock.release()
    print(varT)

## Threading Implementation
print("\n### Threading Implementation ###")
varT = 0
numThread = 4
start = time.time()
thread1 = threading.Thread(target = addT,args=(numThread,))
thread2 = threading.Thread(target = addT,args=(numThread,))
thread3 = threading.Thread(target = addT,args=(numThread,))
thread4 = threading.Thread(target = addT,args=(numThread,))
thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread1.join()
thread2.join()
thread3.join()
thread4.join()
end = time.time()
print("Time: %.5f s"%(end-start))
print("Final var Value: %i \n"%varT)
