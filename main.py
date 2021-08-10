import numpy as np
import cv2
import modifiedcentroidtrack as cc

PROTOTXT = "MobileNetSSD_deploy.prototxt.txt.txt"
MODEL = "MobileNetSSD_deploy.caffemodel"
GENDERPROTO = "gender_deploy.prototxt.txt.txt"
GENDERMODEL = "gender_net.caffemodel"
GPU_SUPPORT = 0
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",  "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
gCLASSES = ['Female', 'Male']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


cap = cv2.VideoCapture(0)
w = cap.get(3)
h = cap.get(4)

line = int(3*(h/5))
aline = np.array([[0 , line],[w, line]], np.int32)
person = 0
personn = 0

ct = cc.CentroidTracker()

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
gnet = cv2.dnn.readNet(GENDERPROTO, GENDERMODEL)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

if GPU_SUPPORT:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

while True:
    ret, frame = cap.read()
    if not ret:
       break
    h, w = frame.shape[:2]
    print(frame.shape[:2])

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    gblob = cv2.dnn.blobFromImage(frame, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    net.setInput(blob)
    gnet.setInput(gblob)

    detections = net.forward()
    rects = []
    for i in np.arange(0, detections.shape[2]):

       confidence = detections[0, 0, i, 2]
       idx = int(detections[0, 0, i, 1])
       
       if idx == 15 :
           gdetections = gnet.forward()
           gender = gCLASSES[gdetections[0].argmax()]
           idx = int(detections[0, 0, i, 1])
           e = np.array([w, h, w, h])

           
           box = detections[0, 0, i, 3:7] * e
          
           s = box.astype("int")

           t1 = (300,400)
           t2 = (25,100)

           
                          
           (startX, startY, endX, endY) = s
           j1 = [startX,startY]
           j2 = [endX, endY]
           
           cv2.rectangle(frame, (t1), (t2), (255, 255, 255), 1)
           #cv2.rectangle(frame, (0,0), (620,480), (0, 255, 255), 1)

           if (t2[0]) < (startX) < (t1[0]) and (t2[1]) < (startY) < (t1[1]) and (t2[0]) < (endX) < (t1[0]) and (t2[1]) < (endY) < (t1[1]):
               print("MASHOK")
               rects.append(s)
               glabel = "// {}: {:.3f}%".format(gender, gdetections[0].max())
               label = "{}: {:.2f}%".format(CLASSES[idx],confidence*100)
               cv2.rectangle(frame, (startX, startY), (endX, endY),    COLORS[idx], 2)
               y = startY - 15 if startY - 15 > 15 else startY + 15
               cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
               cv2.putText(frame, glabel, (startX+150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

           '''rects.append(s)
           label = "{}: {:.2f}%".format(CLASSES[idx],confidence*100)
           glabel = "// {}: {:.3f}%".format(gender, gdetections[0].max())
           cv2.rectangle(frame, (startX, startY), (endX, endY),    COLORS[idx], 2)
           y = startY - 15 if startY - 15 > 15 else startY + 15
           cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
           cv2.putText(frame, glabel, (startX+150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)'''

    objects = ct.update(rects)
    ceex = ct.arraycount(objects, line)
    
    if objects is not None:
        for (objectID, centroid) in objects.items():
           text = "ID {}".format(objectID)
           cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
           cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
           frame = cv2.polylines(frame,[aline],True,(0,0,255),thickness=2)

           if ceex == 1:
               person += 1
           elif ceex == 0:
               personn += 1
           

    strPerson = 'Person In: '+ str(person)
    strPersonO = 'Person Out: '+ str(personn)
    
    print(strPerson)
    print(strPersonO)
    cv2.putText(frame, strPerson, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, strPersonO, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    cv2.imshow('Frame',frame)
    k = cv2.waitKey(2) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
