FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx git
RUN git clone https://github.com/ostris/ai-toolkit.git && \
    cd ai-toolkit && \
    git submodule update --init --recursive

WORKDIR /app/ai-toolkit

RUN pip3 install -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

