#!/usr/bin/python
# coding: utf-8
import socket
import select
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
"""
複数のclientと通信を行い、システムのaction commandを生成するプログラム

client:1 sound of A(stream_system_A.py)
client:2 sound of B(stream_system_B.py)
client:3 camera of SOTA (sotacam_py_fujie_3.py)
client:4 speech recognition A(stream_speech_recognition_A.py) #このは・SotaWozsystem/Server下
client:5 speech recognition B(stream_speech_recognition_B.py) #これは・SotaWozsystem/Server下

役割は大きく分けて２つある
1. 各clientから文字列として特徴量を受け取る
2. 受け取った特徴量をmodelに入力し、VAD, GAZE, TARGET, ACTION の出力を得る

"""

#from nlu import *
#print("complete loading Word2vec model ####")
import keras
model = keras.models.load_model('utterance_lld_final10.h5')
gaze_model = keras.models.load_model('gaze_1017_middle50.h5')
from passive_model import Passive_multimodel_notstateful_model
passive_model = Passive_multimodel_notstateful_model()
passive_model.load_weights('PassiveNotStatefulLSTMNetworkMultitask.h5')
passive_model.summary()
import tensorflow as tf
graph = tf.get_default_graph()
print("complete loading VAD model ####")
print("complete loading gaze model ####")
print("complete loading action generate model ####")

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 接続待ちするサーバのホスト名とポート番号を指定
host = "127.0.0.1";
port = 55580
client_port = 65000
argument = (host, port)
sock.bind(argument)
# 5 ユーザまで接続を許可
sock.listen(7)
clients = []
feature = [[],[],[],[],[],[]]


"""
初期設定
"""
present_user = "A"
user2key = {"A":0, "B":1}

def make_connection(host_, port_):
  while True:
    try:
      sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      sock.connect((host_, port_))
      print('connected')
      return sock
    except socket.error as e:
      print('failed to connect, try reconnect')
      time.sleep(1)

def make_feature(msg, X_list):
    smile_list = msg.rstrip().split(",")
    #print(len(smile_list))
    if len(smile_list) == 114:
        try:
            X = map(float,smile_list) #python2
        except:
            smile_list[0] = '0.0'
            X = map(float,smile_list) #python2
        X = np.array(X).astype(np.float32)
        X_list.append(X)
    return X_list

# 接続済みクライアントは読み込みおよび書き込みを繰り返す
def loop_handler(connection, address, target):
    global graph
    global gaze_model
    X_A,X_B = [], [[0]*114]
    X_img = []
    global feature
    #print("connection",connection)
    print("address",address)
    print('user',target)
    #if "A" in target:
    #else: model_B =  keras.models.load_model('utterance_lld_final10.h5')
    connection.send('1')
    while True:
        #try:

        #クライアント側から受信する
        #res = connection.recv(4096)
        #print(res)
        if "A" == target:
            """
            AのLLD特徴量
            """
            res = connection.recv(4096)

            #time.sleep(0.001)
            X_A = make_feature(res,X_A)

            #time.sleep(0.001)
            if len(X_A) >= 10:
                X = np.array(X_A[0:10])
                del X_A[0:10]
                feature[0].append(X)
                X = X.reshape(1,10,114)
                #print(np.shape(X))

                with graph.as_default():
                    y_pred = model.predict(X)
                    #print("A",y_pred)


        elif "B" == target:
            """
            BのLLD特徴量
            """
            res = connection.recv(4096)

            #time.sleep(0.001)
            X_B = make_feature(res, X_B)

            #time.sleep(0.001)
            if len(X_B) >= 10:
                X = np.array(X_B[0:10])
                del X_B[0:10]
                feature[1].append(X)
                #global graph
                with graph.as_default():
                    y_pred = model.predict(X.reshape(-1,10,114))
                    #print("B",y_pred)
                #client_VAD_B.send(str(y_pred[0][0]))

        elif "gaze" in target:
            """
            Sotaの目線画像 + Sotaの顔向き
            """

            res = connection.recv(2**20)

            #time.sleep(0.001)
            smile_list = res.rstrip().split(",")
            try:
                X = map(float,smile_list)
            except:
                print("length",len(X))
                print(smile_list)
            try:
                X = np.reshape(X,(1,32,96,1))
            except:
                X = feature[2][-1]
                #X = np.reshape(X,(1,32,96,1))
            feature[2].append(X)
            feature[5].append(user2key[present_user])
            with graph.as_default():
                y_pred = gaze_model.predict(X)
                #print("gaze",y_pred)

            if len(feature[0]) >= 10 and len(feature[1]) >= 10 and len(feature[2]) >= 10 and len(feature[5]) >= 10:
                #print("prepared prediction!")
                #client.send("prepared prediction!")

                X_A_audio = np.reshape(feature[0][-10:],(1,10,10,114))
                X_B_audio = np.reshape(feature[1][-10:],(1,10,10,114))
                X_img = np.reshape(feature[2][-10:],(1,10,32,96,1))
                X_face = np.reshape(feature[5][-10:],(1,10))
                with graph.as_default():
                    y_pred = passive_model.predict([X_A_audio,X_B_audio,X_img,X_face])
                #print("y_pred::",y_pred)
                print(y_pred[3][0][0])
                client_VAD_A.send(str(y_pred[0][0][0]))
                client_VAD_B.send(str(y_pred[1][0][0]))
                client.send(str(y_pred[3][0][0])+","+str(y_pred[4][0][0]))
                #client.send(str(y_pred[4][0][0]))
                del feature[0][0]
                del feature[1][0]
                del feature[2][0]
                del feature[5][0]
            else:
                pass

        elif "lang_A" == target:
            """
            Aの音声認識結果
            """
            res = connection.recv(4096)
            print(res)
            with graph.as_default():
                y_pred = wakati(res)
            print("question" if y_pred==1 else 0)
            feature[3].append(res)

        elif "lang_B" == target:
            """
            Bの音声認識結果
            """
            res = connection.recv(4096)
            print(res)
            with graph.as_default():
                y_pred = wakati(res)
            print("question" if y_pred==1 else 0)
            feature[4].append(res)

        #print(np.shape(feature[0]),np.shape(feature[1]),np.shape(feature[2]),np.shape(feature[3]))
        #行動決定モデルの予測


            #client.send("waiting...")
        """
        LLD[A] : feature[key-10:key][0]
        LLD[B] : feature[key-10:key][1]
        EyeImage: feature[key-10:key][2]
        LangA: feature[key-10:key][3]
        LangB: feature[key-10:key][4]
        y_target: feature[key-10:key][5]
        """
        #except Exception as e:
        #    print(e)
        #    print('error')
        #    break

client = make_connection(host,client_port)
client_VAD_A = make_connection(host,65001)
client_VAD_B = make_connection(host,65002)

while True:
    try:
        # 接続要求を受信
        print("waiting........")
        conn, addr = sock.accept()
        user = conn.recv(1024)

    except KeyboardInterrupt:
        sock.close()
        exit()
        break
    # アドレス確認
    print("[アクセス元アドレス]=>{}".format(addr[0]))
    print("[アクセス元ポート]=>{}".format(addr[1]))
    print("\r\n")

    # 待受中にアクセスしてきたクライアントを追加
    clients.append((conn, addr))
    # スレッド作成
    thread = threading.Thread(target=loop_handler, args=(conn, addr, user), )
    # スレッドスタート
    thread.start()
