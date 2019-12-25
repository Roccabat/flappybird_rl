FROM ubuntu

RUN apt-get dist-upgrade
RUN apt-get update
RUN apt-get install tzdata -y 

RUN apt-get install python3.6 -y

RUN apt-get install python3-pip -y

RUN apt-get install python-opengl -y
#RUN apt install freeglut3-dev -y
RUN apt-get install xvfb -y

RUN apt-get install ffmpeg -y 

RUN pip3 install numpy tensorflow gym keras-rl pygame scipy

RUN apt-get install git-all -y

RUN git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
RUN pip3 install -e /PyGame-Learning-Environment

COPY ./flappy_gym_env /usr/codes/flappy_gym_env
RUN pip3 install -e /usr/codes/flappy_gym_env/

CMD ["bash"]