# https://non-dimension.com/python-multitask/

import numpy as np
import time

def task(_n):
    s = 0
    for i in range(1,_n+1):
        s+=i
        time.sleep(0.1)
    return s

ns = list(np.arange(1,11)) #1〜10までの数字のリストを作成

start = time.time() #処理開始時間

#========計算処理========
sms_single = []
for n in ns:
    sms_single.append(task(n))
#=======================
    
end = time.time() #処理終了時間
delta = end - start #処理時間
print('処理時間:{}s'.format(round(delta,3)))

import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

ns = list(np.arange(1,11)) 

start = time.time()

#========計算処理========
with ThreadPoolExecutor(6) as e:
    ret = e.map(task, ns)
sms_multi = [r for r in ret]
#=======================
  
end = time.time()
delta = end - start
print('処理時間:{}s'.format(round(delta,3)))  