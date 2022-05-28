import cv2   # from Opencv as we are dealing with the use of camera in this attendance project
import numpy as np   # Library to perform the mathematical operations
import os   # To read the Images
from datetime import datetime  # This module is required to obtain the date and time in order to mark the attendance in sheet
import face_recognition   # face_recognition library is required to perform the face detection and its recognition operation

record = 'Images'    # It gives the path of the directory in which all the record of images is present
images = []          # In this all the images are stored
Name = []            # In this all the names are stored
List = os.listdir(record)   # Contains the list of all the components of Images directory
print(List)                 # Return the list of images stored in Images directory

for spimg in List:  # This for loop is used to read all the images from the directory and split their text in order get the name of corresponding person
    persons_img = cv2.imread(f'{record}/{spimg}') # To read the image
    images.append(persons_img)  # Add the image to the list
    Name.append(os.path.splitext(spimg)[0]) # Add the name of person to list after splitting the text of image's name into 0 and 1 so we need 0 component which represents the name
print(Name)

# After this loop we wil get the list of images and the corresponding names.

def faceEncodings(images):  # A function to find the encodings of each and every image as per our wish
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converting the BGR(get from cv2) images into the RGB format
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
encodeListKnown = faceEncodings(images)
print("All Encodings Complete!")
# Ater execution of this faceEncodings function We will get an array of 128*n elemnts which contains 128 encodings for each image

def attendance(name):  # A function to mark the attendance of attendee in the sheet
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{dStr},{tStr}')  # To write the name of attendee along with it joining date and time in sheet


cam = cv2.VideoCapture(0)  # a variable to capture the video through camera

while True:
    ret, frame = cam.read()
    camface = cv2.resize(frame, (0,0), None, 0.5, 0.5)  # to resize the image capture by camera
    camface = cv2.cvtColor(camface, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(camface)  # To detect the location of face in the frame of camera
    encodesCurrentFrame = face_recognition.face_encodings(camface, facesCurrentFrame)  # To encode the faces obtained from the current frame

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # It will compare encoded face obtained from the camera frame  with the images present in known encoded list
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # To find the distance between the faces

        matchIndex = np.argmin(faceDis)   # to obtain the minimum face distance index value which will decide whether the faces match or not

        if matches[matchIndex]:    # If the face obtained from camera matches with an image in the Images directory then it will take the name of that attendee
               name = Name[matchIndex].upper()
           # Below 5 lines of code shows the formation of a rectangle around the face of attendee in the camera frame along with his/her name
               y1,x2,y2,x1 = faceLoc
               y1,x2,y2,x1 = y1*2,x2*2,y2*2,x1*2
               cv2.rectangle(frame,(x1,y1),(x2,y2), (0,255,0), 1)
               cv2.rectangle(frame, (x1, y2-35), (x2,y2),(0,255,0), cv2.FILLED)   # To form the rectangle
               cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)  # To put the name on the frame

               attendance(name)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(10) == 13:   # Turn off the camera on pressing enter key
        break

cam.release()
cv2.destroyAllWindows()
















































