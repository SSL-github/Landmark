#! /bin/bash

sudo docker run -it --name nia -v /home/smart30/Yonsei_Dataset/train/:/Landmark/data --gpus all  nia-landmark
