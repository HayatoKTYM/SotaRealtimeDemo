from font_ja import CvPutJaText
import time
from keras.models import Sequential , Model
from keras.models import load_model
from keras import backend as K
K.set_learning_phase(0) #set test phase
# with a Sequential model
import tensorflow as tf
graph = tf.get_default_graph()
model = load_model("gaze_1021_middle16.h5")
model_1 = load_model('gaze_1017_middle50.h5')
model_2 = load_model('gaze_1017_middle32.h5')
model_3 = load_model('gaze_1017_middle16.h5')
model_4 = load_model('gaze_1017_middle8.h5')
import argparse
import cv2
import os
from PIL import Image
import numpy as np
import dlib

np.set_printoptions(precision=2)
import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join('/Users/hayato/openface/models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

def predict(model, x_data):
    y = model.predict(x_data)[0]
    #y1 = model_1.predict(x_data)[0]
    #y2 = model_2.predict(x_data)[0]
    #y3 = model_3.predict(x_data)[0]
    #y4 = model_4.predict(x_data)[0]
    #y = np.array([y0[0]+y1[0]+y2[0]+y3[0]+y4[0],y0[1]+y1[1]+y2[1]+y3[1]+y4[1]])
    #y /= 5
    #print(y)
    return y

def getRep(bgrImg):
    if bgrImg is None:
        raise Exception("Unable to load image/frame")

    rgbImg = cv2.cvtColor(bgrImg[50:500,200:700], cv2.COLOR_BGR2RGB)
    rgbImg_ = cv2.resize(rgbImg,(rgbImg.shape[1]//2,rgbImg.shape[0]//2))
    bb = align.getAllFaceBoundingBoxes(rgbImg_)
    #cv2.imshow('r',rgbImg)    
    if bb is None:
        return None

    alignedFaces = []
    for box in bb:
        left,top,right,bottom = box.left()*2,box.top()*2,box.right()*2,box.bottom()*2
        face = align.align(args.imgDim,rgbImg,dlib.rectangle(left,top,right,bottom),
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.equalizeHist(face)
        #cv2.imshow('a',face)
        alignedFaces.append(face[:32, :].astype(np.float32) / 255.)

    if alignedFaces is None:
        raise Exception("Unable to align the frame")
    return (np.array(alignedFaces, dtype=np.float32), bb)

def infer(img, args):
    repsAndBBs = getRep(img)
    reps = repsAndBBs[0]
    bbs = repsAndBBs[1]
    reps = reps.reshape(-1, 32, 96, 1)#

    if len(reps) == 0:
        return ([], bbs)
    y_pred = predict(model, reps)

    return (y_pred,bbs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dlibFacePredictor',type=str,help="Path to dlib's face predictor.",
        default=os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat"))

    parser.add_argument('--networkModel',type=str,help="Path to Torch network model.",
        default=os.path.join( openfaceModelDir,'nn4.small2.v1.t7'))

    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)

    parser.add_argument('--captureDevice',type=int,default=0,
        help='Capture device. 0 for latop webcam and 1 for usb webcam')
    parser.add_argument('--width', type=int, default=600)
    parser.add_argument('--height', type=int, default=400)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    align = openface.AlignDlib(args.dlibFacePredictor)

    video_capture = cv2.VideoCapture(args.captureDevice)
    video_capture.set(3, args.width)
    video_capture.set(4, args.height)
    video_capture.set(5, args.fps)

    confidenceList = []
    cnt = 0
    while True:
        cnt += 1;print(cnt)
        ret, frame = video_capture.read()
        confidences, bbs = infer(frame, args)

        try:
            confidenceList.append('%.2f' % confidences[0])
        except:
            """
            cv2.putText(frame, "{} ".format('NO FACE HERE'),
                        (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255 , 0, 255 ), 5)
            """
            frame = CvPutJaText.puttext(frame, u"顔がありません", (40, 300), 60, (255, 0, 255))
            cv2.imshow('eye_detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        notations = []

        if confidences[0] <= args.threshold:  # 0.5 is kept as threshold for known face.
            notations.append("non-looking")
        else:
            notations.append("looking")


        k = {"non-looking":0,"looking":1}
        for idx, person in enumerate(notations):
            cv2.rectangle(frame, (bbs[idx].left()*2, bbs[idx].top()*2),
                          (bbs[idx].right()*2, bbs[idx].bottom()*2),
                          (255 * int((k[person])), 0,
                           255 * (1 - int(k[person]))), 4)
            cv2.putText(frame, "{} ".format(person),
                        (bbs[idx].left()*2, bbs[idx].bottom()*2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255*int(k[person]), 0, 255*(1-int(k[person]))), 2)


            cv2.rectangle(frame,(15,380),(215,455),(0,0,0),-1)
            cv2.putText(frame, "non-look", (18, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, "look", (16, 425),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.line(frame,(110,400),(110+int(50*confidences[1]),400),
                     (0,0,255),10)
            cv2.line(frame, (110, 425), (110+int(50*confidences[0]), 425),
                     (255, 0, 0), 10)
        cv2.imshow('eye_detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()



