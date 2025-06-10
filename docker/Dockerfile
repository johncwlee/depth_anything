FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04


ENV TZ Canada/Eastern
ENV DEBIAN_FRONTEND noninteractive

#* For Pytorch builds
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    FORCE_CUDA=1
#* 8.9 or 8.9+PTX is too new for the current cuda version
#* last version should have +PTX

ARG USERNAME
ARG RENDER_GID
ARG VIDEO_GID
ARG USER_UID=1000
ARG USER_GID=$USER_UID


SHELL ["/bin/bash", "-c"]


#? System Packages
COPY apt_packages.txt .
RUN apt-get update && \
    xargs -a apt_packages.txt apt-get install -y --no-install-recommends && \
    rm apt_packages.txt && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

#? Install python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
COPY requirements.txt .
RUN pip3 install -r requirements.txt && \
    rm requirements.txt

#? Create the user
RUN echo "Username is $USERNAME"
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

#? Create render group and add user to it
RUN groupadd --gid $RENDER_GID render \
    && usermod -aG render $USERNAME \
    && usermod -aG video $USERNAME

#? Add shortcut for jupyter notebook
RUN echo "alias jn='jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root'" >> /home/$USERNAME/.bashrc;

#? Set user and working directory
USER $USERNAME
WORKDIR /home/$USERNAME


#* Notes:
#* - check if there are multiple versions of nccl installed for debugging
    #* pip list | grep nccl
#* - sudo apt clean and docker builder prune if run into gpg key issues