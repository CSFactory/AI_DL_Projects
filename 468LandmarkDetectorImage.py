
import cv2
import mediapipe as mp
import time
 
#cap = cv2.VideoCapture('/Users/namangoyal/Desktop/Ph.D Learning/468_FaceLandmarksUsingOpenCV/PoseVideos/9.mp4')
cap = cv2.imread('/Users/namangoyal/Desktop/Ph.D Learning/468_FaceLandmarksUsingOpenCV/Images/2.jpg')
pTime = 0
 
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=30)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
 
while True:
    #success, img = cap.read()
    imgRGB = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            #mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,
                                  #drawSpec,drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                #print(lm)
                ih, iw, ic = cap.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                print(id,x,y)
                cv2.putText(cap, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                1, (0, 255, 0), 1) #display each point id from 0 to 467 on face
 
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    #cv2.putText(cap, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                #3, (255, 0, 0), 3)
    cv2.imshow("Image", cap)
    cv2.waitKey(1)