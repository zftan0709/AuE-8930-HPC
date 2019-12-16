import multiprocessing as mp
import time


if __name__ == '__main__':
    
### Original Implementation Target Function ###
    def add():
        global var
        var = var + 1 

### Original Implementation ###
    print("\n### Original Implementation ###")
    var = 0
    start = time.time()
    for i in range(50000000):
        add()
    end = time.time()
    print("Time: %.5f s"%(end-start))
    print("Final var Value: %i "%var)


### Multiprocessing Implementation Target Function ###
    def addM(q,numProcess):
        varM = q.get()
        for i in range(int(50000000/numProcess)):
            varM = varM+1
        q.put(varM)
        print(varM)

### Multiprocessing Implementation ###
    print("\n### Multiprocessing Implementation ###")
    varM = 0
    numProcess = 5
    q = mp.Queue()
    q.put(varM)
    start = time.time()
    p1 = mp.Process(target=addM,args=(q,numProcess))
    p2 = mp.Process(target=addM,args=(q,numProcess))
    p3 = mp.Process(target=addM,args=(q,numProcess))
    p4 = mp.Process(target=addM,args=(q,numProcess))
    p5 = mp.Process(target=addM,args=(q,numProcess))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    end = time.time()
    print("Time: %.5f s"%(end-start))
    varM = q.get()
    print(varM)
    print("Final var Value: %i "%varM)
