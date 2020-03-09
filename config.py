import os

touching_dir='labeled_data/touching/'
not_touching_dir='labeled_data/not_touching/'
raw_data_dir='raw_data'
pos_frames_dir='pos_frames'

for i in touching_dir,not_touching_dir,raw_data_dir,pos_frames_dir:
    if not os.path.exists(i):
        os.makedirs(i)
