#! /bin/bash

sudo docker run -it --name nia01 -v /home/smart30/Yonsei_Dataset/train/:/Landmark/data --gpus all  nia-landmark
