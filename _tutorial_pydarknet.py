# coding=utf-8
from yoloPydarknet import pydarknetYOLO
import cv2
import imutils
import time

#載入yolo model
yolo = pydarknetYOLO(obdata="../darknet/cfg/coco.data", weights="yolov3.weights", cfg="../darknet/cfg/yolov3.cfg")
#輸出名稱
video_out = "yolo_output.avi"
start_time = time.time()











if __name__ == "__main__":
    #這邊直接使用webcam
    VIDEO_IN = cv2.VideoCapture(0)
        if(video_out!=""):
            width = int(VIDEO_IN.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
            height = int(VIDEO_IN.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(video_out,fourcc, 30.0, (int(width),int(height)))

    frameID = 0
    while True:
        hasFrame, frame = VIDEO_IN.read()
        # Stop the program if reached end of video
        if not hasFrame:
            end = time.time()

            seconds = end - start_time
            fps  = frameID / seconds;
            print("Done processing !!!")
            print ("Time taken : {0} seconds".format(seconds))
            print ("Estimated frames per second : {0}".format(fps))
            break

        yolo.getObject(frame, labelWant="", drawBox=True, bold=1, textsize=0.6, bcolor=(0,0,255), tcolor=(255,255,255))
        #可以
        #
        cv2.imshow("Frame", imutils.resize(frame, width=850))
        frameID += 1

        if(video_out!=""):
            out.write(frame)

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            out.release()
            break
