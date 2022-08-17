#Using FaceMesh of GOOGLE Mediapipe https://google.github.io/mediapipe/solutions/face_mesh

import cv2
import mediapipe as mp

import time #to see the frame rate

cap  = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('/Users/namangoyal/Desktop/Ph.D Learning/468_FaceLandmarksUsingOpenCV/PoseVideos/9.mp4')
pTime = 0 


mpDraw = mp.solutions.drawing_utils #this will help us to draw on faces
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2) #To create our object #number of faces to detect is 2 (Increase/Decrease as per requirement)

while True:
    success, img = cap.read()
    # Need to convert BGR image to RGB as MediaPipe reads a RGB Image or frame
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(imgRGB)
    
    if results.multi_face_landmarks:#if there is more than 1 face
        for faceLms in results.multi_face_landmarks: #for each landmark in each face
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION)
    
    cTime = time.time()
    fps  = 1 / (cTime-pTime) 
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}',(20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)