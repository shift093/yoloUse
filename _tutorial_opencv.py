# coding=utf-8
from yoloOpencv import opencvYOLO
import cv2
import imutils
import time

#載入yolo model
yolo = opencvYOLO(modeltype="yolov3", \
    objnames="/media/sf_VMshare/cfg.hand_gesture/obj.names", \
    weights="/media/sf_VMshare/cfg.hand_gesture/weights/yolov3-tiny_60000.weights",\
    cfg="/media/sf_VMshare/cfg.hand_gesture/yolov3-tiny.cfg")

#選擇來源,設定(旋轉,路徑)
inputType = "video"  # webcam, image, video
media = "/media/sf_VMshare/hand1.MOV"
write_video = False
video_out = "/media/sf_VMshare/out_hand1_yolov3.avi"
output_rotate = True
rotate = 180

start_time = time.time()

if __name__ == "__main__":

    if(inputType == "webcam"):
        INPUT = cv2.VideoCapture(0)
    elif(inputType == "image"):
        INPUT = cv2.imread(media)
    elif(inputType == "video"):
        INPUT = cv2.VideoCapture(media)
    #如過是圖片
    if(inputType == "image"):
        yolo.getObject(INPUT, labelWant="", drawBox=True, bold=1, textsize=0.6, bcolor=(0,0,255), tcolor=(255,255,255))
        print ("Object counts:", yolo.objCounts)
        print("classIds:{}, confidences:{}, labelName:{}, bbox:{}".\
        format(len(yolo.classIds), len(yolo.scores), len(yolo.labelNames), len(yolo.bbox)) )
        cv2.imshow("Frame", imutils.resize(INPUT, width=850))

        k = cv2.waitKey(0)
        if k == 0xFF & ord("q"):
            out.release()
    #如果是video,webcam
    else:
        if(video_out!=""):
            #圖片長寬的值
            width = int(INPUT.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
            height = int(INPUT.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
            #寫入影片需求模組
            if(write_video is True):
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(video_out,fourcc, 30.0, (int(width),int(height)))
        
        frameID = 0

        while True:
            hasFrame, frame = INPUT.read()
            # Stop the program if reached end of video
            #如果影片沒有讀到 或結束了即就退出！
            if not hasFrame:
                print("Done processing !!!")
                # End time
                end = time.time()
                # Time elapsed
                seconds = end - start_time
                print ("Time taken : {0} seconds".format(seconds))
 
                # Calculate frames per second
                fps  = frameID / seconds;
                print ("Estimated frames per second : {0}".format(fps))
                break
            #是否翻轉的需求
            if(output_rotate is True):
                frame = imutils.rotate(frame, rotate)
            #分類圖片
            yolo.getObject(frame, labelWant="", drawBox=True, bold=2, textsize=1.2, bcolor=(0,255,0), tcolor=(0,0,255))
            print ("Object counts:", yolo.objCounts)
            #yolo.listLabels()
            print("classIds:{}, confidences:{}, labelName:{}, bbox:{}".\
                format(len(yolo.classIds), len(yolo.scores), len(yolo.labelNames), len(yolo.bbox)) )
            cv2.imshow("Frame", imutils.resize(frame, width=850))
            frameID += 1

            if(write_video is True):
                out.write(frame)

            k = cv2.waitKey(1)
            if k == 0xFF & ord("q"):
                if(write_video is True):
                    out.release()

                break
