FROM ubuntu:22.04

#disable input interactive
ENV DEBIAN_FRONTEND noninteractive

# copy source file
RUN apt update -y && apt upgrade -y
RUN mkdir -p project/translator
# COPY project/translator project/translator
COPY . project/translator


# install python 3.11
RUN apt-get install -y software-properties-common
RUN apt update
RUN apt-add-repository -y ppa:deadsnakes/ppa
RUN apt update -y
RUN apt install -y python3.11
RUN apt install -y python3-pip

# chdir
WORKDIR "/project/translator"
RUN pip install -r requirements.txt
RUN python3 translator/gradio.py