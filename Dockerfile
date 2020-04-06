FROM tensorflow/tensorflow:2.0.0-gpu-py3

MAINTAINER Hiroki Matsuzaki

RUN apt-get update \
    && apt-get install -y apt-utils \
    && apt-get install -y vim git locales sudo \
    libglib2.0-0 libsm6 libxrender1 libxext6

RUN pip install --upgrade pip

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

COPY Pipfile ./
COPY Pipfile.lock ./
RUN pip install pipenv
RUN pipenv install --system
RUN rm Pipfile.lock

# create user

RUN groupadd hiroki -g 1001
RUN useradd -m hiroki -u 1001 -g 1001
RUN usermod -s /bin/bash hiroki
RUN echo "hiroki:sigmoid" | chpasswd

RUN gpasswd -a hiroki sudo

WORKDIR /home/hiroki
RUN sudo -u hiroki mkdir -p .ssh
RUN sudo -u hiroki chmod 700 .ssh
RUN sudo -u hiroki touch .ssh/authorized_keys
RUN sudo -u hiroki chmod 600 .ssh/authorized_keys
RUN sudo -u hiroki echo "" > .ssh/authorized_keys

RUN groupadd atsushi -g 1002
RUN useradd -m atsushi -u 1002 -g 1002
RUN usermod -s /bin/bash atsushi
RUN echo "atsushi:sigmoid" | chpasswd

RUN gpasswd -a atsushi sudo

WORKDIR /home/atsushi
RUN sudo -u atsushi mkdir -p .ssh
RUN sudo -u atsushi chmod 700 .ssh
RUN sudo -u atsushi touch .ssh/authorized_keys
RUN sudo -u atsushi chmod 600 .ssh/authorized_keys
RUN sudo -u atsushi echo "pub key to be set" > .ssh/authorized_keys

RUN groupadd yhamajima -g 1003
RUN useradd -m yhamajima -u 1003 -g 1003
RUN usermod -s /bin/bash yhamajima
RUN echo "yhamajima:sigmoid" | chpasswd

WORKDIR /home/yhamajima
RUN sudo -u yhamajima mkdir -p .ssh
RUN sudo -u yhamajima chmod 700 .ssh
RUN sudo -u yhamajima touch .ssh/authorized_keys
RUN sudo -u yhamajima chmod 600 .ssh/authorized_keys
RUN sudo -u yhamajima echo "pub key to be set" > .ssh/authorized_keys


# cleanup
RUN apt-get autoremove && apt-get clean

WORKDIR /home/