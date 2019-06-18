#!/usr/bin/python
# coding: utf-8
import numpy as np

import socket
import time
import matplotlib.pyplot as plt
host = '127.0.0.1'
port = 65001
buff = 1024

fig, ax = plt.subplots(1, 1, figsize=(4,2))
x = list(range(1,11))
y = list(0 for i in range(10))
cnt = len(x)+1
ax.set_ylim(-0.2, 1.1)
ax.title.set_text('VAD_A')
# 初期化的に一度plotしなければならない
# そのときplotしたオブジェクトを受け取る受け取る必要がある．
# listが返ってくるので，注意
lines, = ax.plot(x, y, color='m')
if __name__ == '__main__':

  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  s.bind((host, port))

  while True:
    s.listen(1)
    print('Waiting for connection')
    cl, addr = s.accept()
    print('Established connection')

    while True:
        msg = cl.recv(buff)
        #cl.send('c')  # check if connection is alive
        try:
            y_pred = float(msg)
        except:
            continue
        x.append(cnt)
        y.append(y_pred)
        del x[0]; del y[0]
        lines.set_data(x, y)
        ax.set_xlim((min(x), max(x)))
        cnt+=1
        plt.pause(.01)
        #except socket.error:
        #cl.close()
        #break
