import tensorflow as tf
import cv2
import numpy as np
import os
import skvideo.io
from sort import *


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


def BallAndHoop(detection_graph, image_tensor, boxes, scores, classes, num_detections):

    pass

def AnalyzeVideo(video_path,detection_graph, image_tensor, boxes, scores, classes, num_detections):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    crf=17

    trackerBall=cv2.TrackerCSRT_create()
    trackerHoop = cv2.TrackerCSRT_create()

    #mot_tracker1 = Sort(max_age=3, min_hits=1, iou_threshold=0.15)


    writer = skvideo.io.FFmpegWriter('YOLO.avi',
                            inputdict={'-r': str(fps), '-s': '{}x{}'.format(int(width), int(height))},
                            outputdict={'-r': str(fps), '-c:v': 'libx264', '-crf': str(crf), '-preset': 'ultrafast',
                                        '-pix_fmt': 'yuv444p'}
                            )

    writer1 = skvideo.io.FFmpegWriter('YOLO_SORT.avi',
                            inputdict={'-r': str(fps), '-s': '{}x{}'.format(int(width), int(height))},
                            outputdict={'-r': str(fps), '-c:v': 'libx264', '-crf': str(crf), '-preset': 'ultrafast',
                                        '-pix_fmt': 'yuv444p'}
                            )






    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret, img = cap.read()
            if ret == False:
                break

            frame_expanded = np.expand_dims(img, axis=0)
            (boxesResult, scoresResult, classesResult, num_detectionsResult) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: frame_expanded})

            #track_bbs_ids = mot_tracker1.update(dets)

            Boundingboxes=[]
            OriginalImg = img.copy()

            for i, box in enumerate(boxesResult[0]):
                if (scoresResult[0][i] > 0.5):

                    ymin = int((box[0] * height))
                    xmin = int((box[1] * width))
                    ymax = int((box[2] * height))
                    xmax = int((box[3] * width))
                    NewBox=[xmin,ymin,xmax,ymax,scoresResult[0][i]]
                    NewBox=np.asarray(NewBox)
                    Boundingboxes.append(NewBox)


                    xCoor = int(np.mean([xmin, xmax]))
                    yCoor = int(np.mean([ymin, ymax]))

                    if (classesResult[0][i] == 1):  # basketball
                        trackerBall = cv2.TrackerCSRT_create()
                        trackerBall.init(img, (xmin,ymin,xmax-xmin,ymax-ymin))
                        cv2.circle(img=img, center=(xCoor, yCoor), radius=10,
                                   color=(255, 0, 0), thickness=-1)
                        cv2.putText(img, "BALL", (xCoor - 50, yCoor - 50),
                                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 8)

                    if (classesResult[0][i] == 2):  # Rim


                        cv2.circle(img=img, center=(xCoor, yCoor), radius=10,
                                   color=(255, 0, 0), thickness=-1)


                        cv2.putText(img, "HOOP", (xCoor - 65, yCoor - 65),
                                    cv2.FONT_HERSHEY_COMPLEX, 3, (48, 124, 255), 8)

            (success, box) = trackerBall.update(img)

            if(success):
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(img, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)


            # flag=False
            # Boundingboxes=np.asarray(Boundingboxes)
            # if(Boundingboxes.shape[0]==0):
            #     flag=True
            #     Boundingboxes=np.empty((0, 5))
            # else:
            #     track_bbs_ids = mot_tracker1.update(Boundingboxes)
            #
            # #mot_tracker1.
            #
            # for i in track_bbs_ids:
            #     cv2.rectangle(img, (int(i[0]), int(i[1])),
            #                   (int(i[2]), int(i[3])), (48, 124, 255), 3)



            writer.writeFrame(np.array(img))






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()
    AnalyzeVideo("./Videos/qb basketball zayd nabaa 1.mp4",detection_graph, image_tensor, boxes, scores, classes, num_detections)

