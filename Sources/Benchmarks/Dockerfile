FROM ubuntu:jammy

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

ENV TZ=Europe/Moscow

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt upgrade -y --no-install-recommends

RUN apt install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc-arm-none-eabi \
    qemu-system-arm

RUN useradd -ms /bin/bash user && usermod -aG sudo user
RUN cp /root/.bashrc /home/user/
USER user
CMD /bin/bash
WORKDIR /home/user
