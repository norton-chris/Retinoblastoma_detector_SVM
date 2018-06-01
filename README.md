# Retinoblastoma_detector
This repository contains the SVM program that can detect Retinoblastoma when compared with a normal eye up to 80%.

I have tested this using anaconda 2.

Set up instructions:
Download anaconda prompt 2 and cv2
follow instructions on http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html
setting up the program is kind of a pain, but its pretty simple to use when you get it set up

How to use the program:
On Anaconda prompt change directories(using "cd foldername")
When you are in the correct folder type "python Retinoblastoma_Detector.py" or any of the other program names.
If it doesn't work, look at the code. I have comments saying what to change so it shouldn't be to difficult even if you done know how to code. It will probably be something about missing an image file, and in the code just change image filename to the one you have.

What the program is doing?
First trains the neural network with the images I found on google images.
then finds the best C value
and then reads in the images in the test folder. (add your images here)
then displays the results.

I have commented out the code to figure out the accuracy.

What is Retinoblastoma?
Its a childhood cancer originating in the pupil (ages 0-3 )
Retinoblastoma is fairly easy to be treated, but if not treated the cancer will continue to grow until the child dies.

How can I detect it (I don't trust the program)?
Take a camera with a flash and take a picture of your child's face.
Look at the picture.
If the pupil has a white picture (like the pictures in the training folder) he or she is likely to have Retinoblastoma.
If the pupil is red, this is normal.
If the pupil is black, you probably have red eye correction or the flash is too high. I would try again to make sure.

Why am I making this?
I am currently a high school senior and I am required to make a senior thesis, and this is my thesis project.

How realiable is this program?
Not very :/
80% is not high enough for detecting a cancer that could kill someone.
Hopefully I can keep working on this and get the percent up, but for right now this is for fun and raise awareness for Retinoblastoma.

Hopes for the future:
I hope that this will be available on every device and not need someone with programming skills to operate
and get the accuracy close to 100%
