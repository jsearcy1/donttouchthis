import cv2
import numpy as np
import os
import uuid
os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf
from model import model,preprocess
import time
import pickle

model_name='model_train_v2.h5'

model.load_weights(model_name)




cap = cv2.VideoCapture(0)
last_warning=0

def save_image(image,pos=False):
    if pos:
        fname=os.path.join("./pos_frames",str(uuid.uuid4())+".png")
    else:
        fname=os.path.join("./raw_data",str(uuid.uuid4())+".png")
    cv2.imwrite(fname,image)


    
bad_count=0
avg_prediction=[]
frame_array=[]

while(True):
    # Capture frame-by-frame
    
    ret, frame = cap.read()
    

    
    # Our operations on the frame come here
    small = cv2.resize(frame,(96,96))
    array=preprocess(np.expand_dims(small,0))
    
    # Display the resulting frame
    prediction=np.squeeze(model.predict_on_batch(array)[0])    
    avg_prediction.append(prediction)
    frame_array.append(small)

    if len(avg_prediction) >10:
        avg_prediction=avg_prediction[1:]
        frame_array=frame_array[1:]
    else:
        continue

    test_value=np.mean(prediction)
    print(test_value,end='\r')

    if test_value > .7:
        if time.time()-last_warning > 60:
            last_warning=time.time()
            os.system(r'spd-say "you cant touch this"')
        worst_frame=np.argmax(avg_prediction)
        save_image(frame_array[worst_frame],pos=True)
        print('AHHH!!!')
        avg_prediction=[]


    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
