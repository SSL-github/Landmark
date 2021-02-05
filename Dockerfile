FROM tensorflow/tensorflow:nightly-gpu

RUN apt-get update && apt-get install -y git 

#RUN apt-get update && apt-get install -y nvidia-container-toolkit

RUN git clone https://github.com/SSL-github/Landmark.git

RUN git clone https://github.com/tensorflow/models.git

WORKDIR /yonsei_land

RUN python -m pip install --upgrade pip

RUN python -m pip install -r requirements.txt

#RUN bash ./models/research/delf/delf/python/training/install_delf.sh
