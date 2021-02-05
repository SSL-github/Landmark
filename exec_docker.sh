#! /bin/bash

sudo docker run -it --name nia01 -v /home/smart30/Yonsei_Dataset/train/:/Landmark0205/data --gpus all  nia-landmark
