import multiprocessing as mp
import threading
import time

if __name__ == '__main__':
    
### Original Implementation Target Function ###
    def add():
        global var
        var = var + 1

    def minus():
        global var
        var = var - 1

### Original Implementation ###
    print("\n### Original Implementation ###")
    var = 0
    start = time.time()
    for i in range(5000000):
        add()
        minus()
    end = time.time()
    print("Time: %.5f s"%(end-start))
    print("Final var Value: %i "%var)


### Threading Implementation Target Function ###
    lock = threading.Lock()
    def addT():
        global varT
        lock.acquire()
        for i in range(5000000):
            varT = varT+1  
        lock.release()
        #print(varT)

    def minusT():
        global varT
        lock.acquire()
        for i in range(5000000):
            varT = varT-1
        lock.release()
        #print(varT)

## Threading Implementation ###
    print("\n### Threading Implementation ###")
    varT = 0
    start = time.time()
    thread1 = threading.Thread(target = addT)
    thread2 = threading.Thread(target = minusT)
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    end = time.time()
    print("Time: %.5f s"%(end-start))
    print("Final var Value: %i \n"%varT)
