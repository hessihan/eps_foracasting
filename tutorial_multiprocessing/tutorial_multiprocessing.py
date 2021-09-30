import numpy as np
import time
from multiprocessing import Pool, cpu_count

def process(n):
    time.sleep(0.1)
    return 2 ** n

def single(n):        
    input_list = list(range(n))
    start = time.time() #処理開始時間

    #========計算処理========    
#     for i in range(n):
#         process(i)
    result = map(process, input_list)
    #     np.mean(np.array(list(result)))
    #=======================
    
    end = time.time() #処理終了時間
    delta = end - start #処理時間
    print('処理時間:{}s'.format(round(delta,3)))

def multi(n):
    input_list = list(range(n))
    start = time.time()

    #========計算処理========
    p = Pool(cpu_count() - 1)
    result = p.map(process, input_list)
    p.close
    #=======================

    end = time.time()
    delta = end - start
    print('処理時間:{}s'.format(round(delta,3)))  
    
if __name__ == "__main__":
    loop_n = 20000
    single(loop_n)
    multi(loop_n)