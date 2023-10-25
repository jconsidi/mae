FROM ubuntu:22.04

# environment

ARG DEBIAN_FRONTEND=noninteractive

# baseline setup

RUN apt-get update

RUN apt-get update --fix-missing \
 && apt-get install -y \
  build-essential \
  openssl \
  python3.10 \
  python3.10-dev \
  python3-pip \
  tzdata \
  wget

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN pip3 install --upgrade pip

# custom start

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV HOME /mae
ENV TZ US/Eastern
WORKDIR $HOME

# Python modules

ADD requirements.txt $HOME
RUN pip3 install -r $HOME/requirements.txt

# real content here. changes frequently.

ADD *.py $HOME/
ADD util/*.py $HOME/util/

# compile all python files as sanity check
RUN python3 -m compileall *.py */*.py
