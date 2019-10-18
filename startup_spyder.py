#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:35:35 2019

@author: adonay

STARTUP spyder file
"""


import time
start = time.time()
pause_for = 0.0005
time.sleep(pause_for)
time_past = time.time() - start

print(f'took {time_past:.4f} instead of {pause_for:.4f}')
if time_past > pause_for*2:
    import appnope
    appnope.nope()
