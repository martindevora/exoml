FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential libssl-dev python3.10 \
    python3-pip libbz2-dev libssl-dev libreadline-dev \
    libffi-dev libsqlite3-dev tk-dev libpng-dev libfreetype6-dev llvm-9 llvm-9-dev \
    gfortran gcc locales python3-tk libpython3.10 \
    nano git wget curl nano graphviz -y
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt --ignore-installed
#RUN python3.10 /usr/bin/exoml/iatson_planet.py