import numpy as np
import cv2
import uuid
import os
from config import raw_data_dir

#Capture 200 images

if not os.path.exists(raw_data_dir):
    os.mkdir('./raw_data')


cap = cv2.VideoCapture(0)


def save_image(image):
    fname=os.path.join("./raw_data",str(uuid.uuid4())+".png")
    cv2.imwrite(fname,image)

itr=0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    small = cv2.resize(frame,(96,96))
    # Display the resulting frame
    cv2.imshow('frame',small)
    if np.random.uniform() < .1:
        save_image(small)
        itr+=1
    
    if cv2.waitKey(1) & 0xFF == ord('q') or itr==200:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
