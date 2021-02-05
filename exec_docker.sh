#! /bin/bash

sudo docker run -it --name nia -v /home/smart30/Yonsei_Dataset/train:/Last_Land/data --gpus all nia-landmark
