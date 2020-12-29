import tensorflow as tf
import cv2
import numpy as np
import os
import skvideo.io
from sort import *
from scipy.spatial import distance as dist
import time


ScoreTimestamps=[]

def tensorflow_init():
    MODEL_NAME = 'inference_graph'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    return detection_graph, image_tensor, boxes, scores, classes, num_detections


fps=None
video_path=None
def AnalyzeVideo(video_path,detection_graph, image_tensor, boxes, scores, classes, num_detections):
    global fps,videoAddress
    videoAddress=video_path
    cap = cv2.VideoCapture(videoAddress)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fps = cap.get(cv2.CAP_PROP_FPS)
    crf=17

    trackerBall=cv2.TrackerCSRT_create()
    trackerHoop = cv2.TrackerCSRT_create()
    trackerHoop2 = cv2.TrackerCSRT_create()




    # BallIDTracker = Sort(max_age=3, min_hits=1, iou_threshold=0.15)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (int(width), int(height)))

    BallTrackerInitialized=False
    HoopTrackerInitialized=False



    counter=0

    with tf.Session(graph=detection_graph) as sess:
        while True:
            BallFoundFlag = None
            HoopFound=None
            Hoop1Found=None

            start_time = time.time()
            ret, img = cap.read()
            print("1--- %s seconds ---" % (time.time() - start_time))
            counter+=1
            if ret == False:
                break

            # if(counter/fps<8):
            #     continue



            start_time = time.time()
            frame_expanded = np.expand_dims(img, axis=0)
            (boxesResult, scoresResult, classesResult, num_detectionsResult) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
            print("2--- %s seconds ---" % (time.time() - start_time))

            start_time = time.time()
            BoundingboxesBalls=[]
            BouningboxesHoops=[]
            OriginalImg = img.copy()


            for i, box in enumerate(boxesResult[0]):
                if (scoresResult[0][i] > 0.5):
                    print("FOUND")

                    ymin = int((box[0] * height))
                    xmin = int((box[1] * width))
                    ymax = int((box[2] * height))
                    xmax = int((box[3] * width))
                    NewBox=[xmin,ymin,xmax,ymax,scoresResult[0][i]]
                    NewBox=np.asarray(NewBox)



                    xCoor = int(np.mean([xmin, xmax]))
                    yCoor = int(np.mean([ymin, ymax]))


                    if (classesResult[0][i] == 1):  # basketball
                        BoundingboxesBalls.append(NewBox)

                    if (classesResult[0][i] == 2):  # Rim

                        BouningboxesHoops.append(NewBox)





            if(len(BoundingboxesBalls)==0):
                if(BallTrackerInitialized==False):
                    continue
                else:
                    (success, box) = trackerBall.update(img)

                    if(success):
                        (x, y, w, h) = [int(v) for v in box]
                        cv2.rectangle(img, (x, y), (x + w, y + h),
                                      (0, 255, 0), 2)
                        BallFoundFlag = [x, y, w, h]
            else:
                # if(BallTrackerInitialized==False):
                    HighestScore=0
                    BoundingboxOfBall=None
                    for i in BoundingboxesBalls:
                        if(i[4]>HighestScore):
                            HighestScore=i[4]
                            BoundingboxOfBall=i

                    trackerBall = cv2.TrackerCSRT_create()
                    trackerBall.init(img, (BoundingboxOfBall[0],BoundingboxOfBall[1],BoundingboxOfBall[2]-BoundingboxOfBall[0],BoundingboxOfBall[3]-BoundingboxOfBall[1]))
                    (success, box) = trackerBall.update(img)
                    BallFoundFlag=success

                    if(success):
                        (x, y, w, h) = [int(v) for v in box]
                        cv2.rectangle(img, (x, y), (x + w, y + h),
                                      (0, 255, 0), 2)
                        BallFoundFlag=[x,y,w,h]

                    BallTrackerInitialized=True

            #############HOOP Detection ##

            if(len(BouningboxesHoops)==0 ):
                if(HoopTrackerInitialized==False):
                    continue
                else:
                    (success, box) = trackerHoop.update(OriginalImg)

                    if(success):
                        (x, y, w, h) = [int(v) for v in box]
                        cv2.rectangle(img, (x, y), (x + w, y + h),
                                      (48, 124, 255), 2)

                        HoopFound=[x,y,w,h]

                    (success, box) = trackerHoop2.update(OriginalImg)

                    if(success):
                        (x, y, w, h) = [int(v) for v in box]
                        cv2.rectangle(img, (x, y), (x + w, y + h),
                                      (48, 124, 255), 2)
                        Hoop1Found = [x, y, w, h]
            else:
                # if(BallTrackerInitialized==False):


                    trackerHoop = cv2.TrackerCSRT_create()

                    trackerHoop2 = cv2.TrackerCSRT_create()

                    BoundingboxOfHoop=BouningboxesHoops[0]
                    trackerHoop.init(OriginalImg, (BoundingboxOfHoop[0]-30,BoundingboxOfHoop[1]-30,BoundingboxOfHoop[2]-BoundingboxOfHoop[0]+30,BoundingboxOfHoop[3]-BoundingboxOfHoop[1]+30))
                    (success, box) = trackerHoop.update(OriginalImg)

                    if(success):
                        (x, y, w, h) = [int(v) for v in box]
                        cv2.rectangle(img, (x, y), (x + w, y + h),
                                      (48, 124, 255), 5)

                        HoopFound = [x, y, w, h]



                    if(len(BouningboxesHoops)==2):
                        BoundingboxOfHoop=BouningboxesHoops[1]
                        trackerHoop2.init(OriginalImg, (BoundingboxOfHoop[0]-30,BoundingboxOfHoop[1]-30,BoundingboxOfHoop[2]-BoundingboxOfHoop[0]+30,BoundingboxOfHoop[3]-BoundingboxOfHoop[1]+30))
                        (success, box) = trackerHoop2.update(OriginalImg)

                        if(success):
                            (x, y, w, h) = [int(v) for v in box]
                            cv2.rectangle(img, (x, y), (x + w, y + h),
                                          (48, 124, 255), 5)
                            Hoop1Found = [x, y, w, h]
                    HoopTrackerInitialized=True


            if(BallFoundFlag is not None):
                MidXBall=int((BallFoundFlag[0]+BallFoundFlag[2])/2)
                MidYBall = int((BallFoundFlag[1] + BallFoundFlag[3]) / 2)

                if(Hoop1Found is not None):
                    MidXHoop = int((Hoop1Found[0] + Hoop1Found[2]) / 2)
                    MidYHoop = int((Hoop1Found[1] + Hoop1Found[3]) / 2)




                    D = dist.euclidean((MidXBall, MidYBall), (MidXHoop, MidYHoop))

                    if(D<30):
                        ScoreTimestamps.append(counter)
                        PutText = "SCOREDD : " + str(int(D))
                        cv2.putText(img, PutText, (int(height / 2), int(width / 2)),
                                    cv2.FONT_HERSHEY_COMPLEX, 3, (48, 124, 255), 8)
                    else:
                        PutText = "Dist " + str(int(D)) + " Frame : "+str(counter)+": "+str(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        cv2.putText(img, PutText, (int(height / 2), int(width / 2)),
                                    cv2.FONT_HERSHEY_COMPLEX, 3, (48, 124, 255), 8)


                if(HoopFound is not None):
                    MidXHoop = int((HoopFound[0] + HoopFound[2]) / 2)
                    MidYHoop = int((HoopFound[1] + HoopFound[3]) / 2)





                    D = dist.euclidean((MidXBall, MidYBall), (MidXHoop, MidYHoop))



                    if(D<30):
                        ScoreTimestamps.append(counter)
                        PutText = "SCOREDD : " + str(int(D))
                        cv2.putText(img, PutText, (int(height / 2), int(width / 2)),
                                    cv2.FONT_HERSHEY_COMPLEX, 3, (48, 124, 255), 8)
                    else:
                        PutText = "Dist " + str(int(D)) + " Frame : "+str(counter)+": "+str(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        cv2.putText(img, PutText, (int(height / 2), int(width / 2)),
                                    cv2.FONT_HERSHEY_COMPLEX, 3, (48, 124, 255), 8)


            print("3--- %s seconds ---" % (time.time() - start_time))









            out.write(img)
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def GenerateScoreingVideos():
    print(ScoreTimestamps)
    # ScoreTimestamps=[381, 382, 383, 384, 513, 514, 515, 516, 517, 518, 602, 603, 604, 605, 727, 728, 729, 812, 933, 935, 1032, 1033, 1034, 1162, 1164, 1165, 1166, 1266, 1267, 1268, 1269, 1270, 1271, 1272]

    fps=30
    # videoAddress='./Videos/qb basketball zayd nabaa 1.mp4'


    RefinedTimeStamps=[]
    RewindBackTime=1*fps

    cap = cv2.VideoCapture(videoAddress)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)


    counter=0


    for i in range(len(ScoreTimestamps)):
        if i==0:
            RefinedTimeStamps.append(ScoreTimestamps[i])
            continue
        else:

            if(ScoreTimestamps[i]-ScoreTimestamps[i-1]<fps):# 27 is the fps , removing multiple timestamps of same schots
                continue
            else:
                RefinedTimeStamps.append(ScoreTimestamps[i])

    SavingFlag=False
    SmallVideo=0
    while True:

        ret, img = cap.read()

        counter += 1
        if ret == False:
            if(SavingFlag):
                SavingFlag = False
                out.release()
                SmallVideo += 1


            break




        if(counter+RewindBackTime in RefinedTimeStamps):
            print("Started at : "+str(counter))
            SavingFlag=True
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            FileName="ScoringVideo"+str(SmallVideo)+".avi"
            out = cv2.VideoWriter(FileName, fourcc, fps, (int(width), int(height)))

        if(counter-RewindBackTime in RefinedTimeStamps):
            print("Ended at : "+str(counter))

            SavingFlag = False
            out.release()
            SmallVideo+=1

        if(SavingFlag):
            print("WRITING:"+str(counter))
            out.write(img)




    cap.release()















# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()
    AnalyzeVideo("./Videos/qb basketball zayd nabaa 1.mp4",detection_graph, image_tensor, boxes, scores, classes, num_detections)
    GenerateScoreingVideos()

