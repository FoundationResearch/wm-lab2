
conda create -n mine_env python=3.9 -y
conda activate mine_env

apt update -y && apt install -y software-properties-common && \
    add-apt-repository ppa:openjdk-r/ppa && apt update -y && \
    apt install -y openjdk-8-jdk

# 验证 Java 版本 (必须输出 1.8.x 或 openjdk version "1.8...")
java -version

sudo apt install xvfb xserver-xephyr tightvncserver python3-opengl ffmpeg

pip install "pip<24.0"
pip install "setuptools<60" "wheel<0.40"

pip install gym==0.21.0 --no-build-isolation

pip install minedojo

pip install "opencv-python<4.10"
pip install "numpy>=1.20,<2.0"
pip install "imageio==2.28"

