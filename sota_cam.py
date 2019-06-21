#! /usr/bin/python
# -*- coding: utf-8 -*-

_author__ = 'Hayato Katayama'

"""
Sotaの目線画像のリアルタイム抽出 && serverへ eye imageを送信するプログラム
[server]:multi_server.py　を起動してから実行する必要あり

memo:
server側に特徴量送信... 1ms
kerasで予測する... 3.6ms
cv2.imshow周りで時間かかっていた(?) ,, かかっていない
cv2.waitkey()で10fpsを調整することで時間がかかっていたので
set(5,fps)に変更して対処
"""

import cv2, dlib, time, os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

import socket
host = '127.0.0.1'
port = 55580
buff = 1024

import keras
keras.backend.set_learning_phase(0) #set test phase
model = keras.models.load_model('gaze_1017_middle50.h5')

import tensorflow as tf
graph = tf.get_default_graph()

import openface
modelDir = os.path.join('/Users/dialog/openface/models')
dlibModelDir = os.path.join(modelDir, 'dlib')
#openfaceModelDir = os.path.join(modelDir, 'openface')
align_path = os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat")
align = openface.AlignDlib(align_path)

def predict(model, x_data):
    """
    視線の推定を行う関数
    """
    y = model.predict(x_data)[0]
    #print(y)
    return y

def getRep(bgrImg):
    """
    読み込んだ画像から目の部分を抽出する関数
    """
    if bgrImg is None:
        raise Exception("Unable to load image/frame")


    rgbImg = cv2.cvtColor(bgrImg[100:500,200:600], cv2.COLOR_BGR2RGB)
    rgbImg_ = cv2.resize(rgbImg,(rgbImg.shape[1]//3,rgbImg.shape[0]//3))
    bb = align.getAllFaceBoundingBoxes(rgbImg_)

    if len(bb) is 0:
        img = [0]*3072
        str_img = map(str,img)
        str_img = ",".join(str_img)
        s.send(str_img)
        return [],None

    alignedFaces = []
    for box in bb:
        left,top,right,bottom = box.left()*3,box.top()*3,box.right()*3,box.bottom()*3
        face = align.align(96,rgbImg,dlib.rectangle(left,top,right,bottom),
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.equalizeHist(face)
        cv2.imshow('face',face)

        alignedFaces.append(face[:32, :].astype(np.float32) / 255.)
        img = np.reshape(alignedFaces,(-1,))
        img = map(str,img)
        str_img = ",".join(img)

        s.send(str_img)


    if alignedFaces is None:
        raise Exception("Unable to align the frame")
    return (np.array(alignedFaces, dtype=np.float32), [left,top,right,bottom])

def infer(img):
    """
    画像を読み込んで視線の予測結果を返す関数
    """
    repsAndBBs = getRep(img)
    reps = repsAndBBs[0]
    bbs = repsAndBBs[1]

    if len(reps) == 0:
        return ([], bbs)
    reps = reps.reshape(-1, 32, 96, 1)#
    y_pred = predict(model, reps)

    return (y_pred,bbs)

fps = 10 # 10フレームにしています．
spf = 1.0 / fps
spfm = max(int(spf * 1000 - 300), 10)
cap = cv2.VideoCapture(0)
cap.set(5,10)
ret, frame = cap.read()
if frame is None:
    print("No camera!")
    exit(1)
w,h = (frame.shape[1], frame.shape[0])
ow, oh = (int(w/2), int(h/2))
frame2 = cv2.resize(frame, (ow, oh))

### 書き出す際に時間遅れが生じていたら
### 同じフレームを埋め込むことで遅れを
### 取り戻すようにする．
### （逆に早いとフレームをスキップすること
###   になるが，それは非常にレアなケース）
class Recorder:
    def __init__(self, fps, width, height):
        self.writer=None
        self.fps=fps
        self.first_time=None
        self.count=0
        self.width = width
        self.height = height

    def release(self):
        self.writer.release()

    def recode(self, frame):
        current_time = time.time()
        if self.first_time is None:
            self.first_time = current_time
            ### 実験的にCODECをMJPG（モーションJPEG）にしています．
            ### モーションJPEGだと，画質の劣化が抑えられ，後で実験
            ### に使うときに好ましいです．その代わり，かなりファイル
            ### サイズが大きくなります．
            ### 見るためだけの目的であれば元に戻していいです．
            filename = "videos/" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f") + ".avi"
            self.writer = cv2.VideoWriter(filename,
                                          # cv2.VideoWriter_fourcc("D","I","V","3"),
                                          cv2.VideoWriter_fourcc("M","J","P","G"),
                                          self.fps, (self.width, self.height), 1)
        expected_total_frame_count = \
            int((current_time - self.first_time) * self.fps)
        while self.count <= expected_total_frame_count:
            print self.count
            self.writer.write(frame)
            self.count += 1
        ### 確認のため，
        ###   現在時刻（秒）
        ###   書き込んだフレーム数
        ###   現状のFPS
        ### を標準出力に出しています．不要なら消して構いません．
        if current_time - self.first_time > 0.0:
            fps = self.count / (current_time - self.first_time)
            # print "%d %d %f" % (current_time, self.count, fps)
            # print "stime::: " , self.first_time

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
    recorder = Recorder(fps, ow, oh)
    confidenceList = []
    cnt=0
    s = make_connection(host, port, "gaze")
    while True:
        ret, frame = cap.read()
        frame2 = cv2.resize(frame, (ow ,oh))
        confidences, bbs = infer(frame2)

        print(cnt)
        cnt+=1

        try:
            confidenceList.append('%.2f' % confidences[0])
        except:
            #cv2.putText(frame2, "{} ".format('NO FACE HERE'),
            #            (50, 300),
            #            cv2.FONT_HERSHEY_SIMPLEX, 2,
            #            (255 , 0, 255 ), 5)

            cv2.imshow('eye_detection', frame2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        notations = []
        if confidences[0] <= 0.5:  # 0.5 is kept as threshold for known face.
            notations.append("non-looking")
        else:
            notations.append("looking")
        k = {"non-looking":0,"looking":1}
        for idx, person in enumerate(notations):
            cv2.rectangle(frame2, (bbs[0]+200, bbs[1]+100),
                          (bbs[2]+200, bbs[3]+100),
                          (255 * int((k[person])), 0,
                           255 * (1 - int(k[person]))), 4)
            cv2.putText(frame2, "{} ".format(person),
                        (bbs[0]+200, bbs[3] + 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255*int(k[person]), 0, 255*(1-int(k[person]))), 2)
            cv2.rectangle(frame2,(15,380),(215,455),(25,25,25),-1)
            cv2.line(frame2,(30,400),(30+int(150*confidences[1]),400),
                     (0,0,255),20)
            cv2.line(frame2, (30, 425), (30+int(150*confidences[0]), 425),
                     (255, 0, 0), 20)
        cv2.imshow('eye_detection', frame2)

        if cv2.waitKey(spfm//10) == 27:
            break
        #recorder.recode(frame2)



    cap.release()
    recorder.release()
    cv2.destroyAllWindows()
