# Structure from motion

Author : Mathis Saunier
Git : https://github.com/mathis-saunier/3D_Reconstruction.git

## Introduction

This is a homework from the course "Computer Vision" in DGIST (South Korea).

The purpose was to build a 3D reconstruction of an object from several 2D images.
This project is far away from state of the art but more of a try to the well-known problem of Structure From Motion (sfm).

## How to

You will need to install some libraries :

`pip install requirements.txt`

Then, you can launch the main code :

`python 3d_reconstruction.py`

You will have to type '1' or '2' in order to choose a dataset.

## Dataset personalization

If you want to use your dataset of images you will have to calibrate your camera in order to calculate the intrinsic parameters of your camera.

The Jupyter Notebook "calib.ipynb" can do this with some images of a "calibration checkboard" that you can easily find on the web.

Then, you just have to change the values of the file "K.txt" in the file's dataset.