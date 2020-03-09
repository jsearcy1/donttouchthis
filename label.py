import cv2
from glob import glob
import matplotlib
matplotlib.use('Agg')  #This dosen't steal focus
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os
from config import touching_dir,not_touching_dir,raw_data_dir,pos_frames_dir

raw_files=glob(raw_data_dir+'/*')+glob(pos_frames_dir+'/*')
 

if not os.path.exists(touching_dir):
    os.makedirs(touching_dir)

if not os.path.exists(not_touching_dir):
    os.makedirs(not_touching_dir)
    

print('Tap (y) for images of you touching your face (n) otherwise ')
for fname in raw_files:
    print('Are you touching your face in this frame (y/n)')
    base_name=os.path.basename(fname)

    image=cv2.imread(fname)
    image=cv2.resize(image,(300,300))
    cv2.imshow('Label',image)
    input_label=cv2.waitKey(0)


    if input_label==ord('y'):
        new_fname=os.path.join(touching_dir,base_name)
        print('Touching',new_fname)        

        shutil.move(fname,new_fname)

    elif input_label==ord('n'):
        new_fname=os.path.join(not_touching_dir,base_name)
        shutil.move(fname,new_fname)
        print('Not Touching',new_fname)

    else:
        print('Unkown label not saving')
    print(new_fname)
