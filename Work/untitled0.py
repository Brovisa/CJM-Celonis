# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 08:57:35 2019

@author: YoupSuurmeijer
"""
import time



start_time = time.time()
t = 0
for i in range(1000):
    t += 1
    time.sleep(0.01)
    speed = t*100/(time.time()-start_time)
    print("Speed1 =", t/(time.time()-start_time), "calls per second")
    if speed > MAX_SPEED:
        #print("Too fast")
        deltat = 1/(speed - MAX_SPEED)
        print("should delay by", deltat, "seconds")
        time.sleep(deltat)
    print("Speed2 =", t/(time.time()-start_time), "calls per second")
    
MAX_SPEED = 1000
 
t = 0
loop_start = time.time()
for i in range(100):
    t += 1
    start_time = time.time()
    time.sleep(0.001)
    time_passed = time.time()-start_time
    print(time_passed)
    if time_passed < 100/MAX_SPEED:
        deltat =  100/MAX_SPEED - time_passed 
        print("should delay by", deltat, "seconds")
        time.sleep(deltat)
    print("Average speed = ", 1/(time.time()-start_time), "calls per second")
