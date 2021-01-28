#! /bin/bash

sudo docker run -it --name yonsei -v /home/smart02/Yonsei_Dataset/:/Landmark/data --gpus all yonsei-landmark
