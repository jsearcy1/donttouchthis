
# DontTouchThis

I have a hard time avoiding touching my face, particularly while on a computer, so I trained a simple ML model that uses a webcam to complain if I do. This package will help you train your own.  

# 1. Getting Started

First Clone this repository

`git clone https://github.com/jsearcy1/donttouchthis.git
cd donttouchthis
`

We'll install this using a python virtual environment. In a shell type

```
python -m venv ./venv
source venv/bin/activate
pip install -r requirements.text
```

* Note every time you want to start the code from a new shell you must run

`source venv/bin/activate`

# 2. Building a Data set

* There is no public data set for this so you'll have to build to your own

## 1. Gather your data
* Point your webcam where you normally work and run
  `python capture.py`
* Your camera should now be recording. Work as usual for a while, and then (Now is probably a good time to wash your hands), touch your face as you would normally
   * You can run this command as many times as you want to get more data

## 2. Label your data
* You need to tell the computer what you want the machine to learn by creating labels for your training data

`python label.py`

* Make sure you click the image box and hit y for images where you are touching your face and n for images where you are not 

# 3. Train a Model
We'll use a pre-trained mobile-net model implemented in Keras. If you want to take a look, it lives in `model.py` otherwise go ahead and run.

`python train_model.py`

This command will take a  while to run, but once it's finished, you're ready to run it full time on the webcam.

# 4. Run

Start the model with

`python run.py`

Now, whenever your computer sees you touch your face, it will say 'you can't touch this' to remind you.
However, with this small data set, you'll probably run into one of the problems below right away.



# Troubleshooting 

## Too many false positives (I'm not touching my face, but my computer is yelling at me all the time)

Every time your computer yells at you, it will save the image that caused it to think you were touching your face. Go ahead and run.

`python label.py`

to label the new false positive images correctly and re-train with

`python train_model.py`



## Too many false negatives (I am touching my face, but my computer says nothing)

This is a little tricker than false positives. It's too much work to record and label every frame from your webcam, so the easiest thing to do it to wash your hands and start recording more examples of you touching your face by running.

`python capture.py`

labeling these images

`python label.py`

and re-training the model

`python train_model.py`



