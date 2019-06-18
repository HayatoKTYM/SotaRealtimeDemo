#!/usr/bin/env python
# encoding: utf-8
import pandas as pd
import numpy as np
import sys
import subprocess
import atexit
import threading
import time
import socket
import matplotlib.pyplot as plt

host = '127.0.0.1'
port = 55580
buff = 1024
# openSMILE で特徴量を抽出するクラス

class SMILE():
    def __init__(self,cmd):
        self._is_running = True
        self.proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


    def get_lines(self):
        while self._is_running:
            line = self.proc.stdout.readline()
            if line:
                yield line

# 正規化するクラス
class Data_manager():
    def __init__(self):
        self.mean_list = []
        self.norm_list = np.zeros(105)
        self.norm_flag = True
        self.norm_count = 0

    def get_norm_list(self,X):
        if self.norm_flag:
            self.norm_count += 1
            self.mean_list = X
            self.norm_list = self.mean_list
            self.norm_flag = False

        elif X[0] > 0.0:
            self.norm_count += 1
            self.mean_list = (X + ((self.norm_count - 1) * self.mean_list))/self.norm_count
            self.norm_list = X - self.mean_list

        if self.norm_count > 10000:
            self.norm_list = X
            self.norm_count = 0

        return self.norm_list

    def normalize_plot_log(self,norm_list,X):
        with open("mean_data_write.csv","a") as f:
            f.write("{},{}".format(self.norm_list[0],X[0]))
            f.write("\n")

# 識別するクラス
class Recognition_unit():
    def __init__(self):
        self.speek_flag = True #
        self.plot_flag = True#False
        self.y = None

    def run(self):
        print_flag = True #False
        #s.send(str(0.0)+","+"認識を開始")
        #th = threading.Thread(target=self.speek_flag_maneger)
        #th.start()
        #model = L.Classifier(MLP())
        #chainer.serializers.load_npz("LSTM_mini_model_norm_sgd0,01.model",model)

        while True:
            smile = SMILE(cmd='./SMILExtract -C emobase2010_stream_A.conf')
            print("start openSMILE")
            for line in smile.get_lines():
                #print(np.shape(feature))
                s.send(line)
                print(len(line.split(',')))
                #time.sleep(0.01)
                continue
                if self.speek_flag:
                    print_flag = True
                    self.plot_flag = True
                    if len(feature) >= 10:
                        X = np.array(feature[0:10])
                        del feature[0:10]
                        y_pred = model.predict(X.reshape(-1,10,114))
                        print(y_pred)
                        plt.figure(figsize=(2,6))
                        #plt.xticks(["Look","Not Look"])
                        plt.ylim([0,1.2])
                        x = [i for i in range(len(y_pred))]

                        plt.bar(["VAD"],[np.max(y_pred)],width=0.05,color="green")
                        plt.savefig('VAD.png')
                    #s.send(str(self.y.data[0][0])+",話してる!!")

                elif self.speek_flag == False and print_flag:
                    if self.y.data.argmax() == 1:
                        #message = '話さないかも!!'
                        #s.send(str(self.y.data[0][0])+",話さないかも!!")
                        print_flag = False
                    else:
                        #message = 'まだ話しそう!!'
                        #s.send(str(self.y.data[0][0])+",まだ話しそう!!")
                        print_flag = False

    def speek_flag_maneger(self):
        while True:
            for data in gmfccserver_client.get():
                self.speek_flag = True
            self.speek_flag = False

def make_connection(host_, port_, user_):
    try:
      sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      sock.connect((host_, port_))
      sock.send(user_)

      flag = sock.recv(1024)
      print('connected')
      return sock
    except socket.error as e:
      print('failed to connect, try reconnect')

if __name__ == '__main__':
    #s = socket.socket()
    ##
    s = make_connection(host, port, "A")
    ##
    #data_manager = Data_manager()
    #gmfccserver_client = GmfccServerClient()
    recognition = Recognition_unit()
    recognition.run()
