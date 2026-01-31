
ssh-keygen -t rsa -b 4096 -C "cpu2"
cat ~/.ssh/id_rsa.pub
git clone git@github.com:FoundationResearch/wm-lab2.git
cd wm-lab2

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n mc python=3.9 -y
conda activate mc

apt update -y && apt install -y software-properties-common && \
    add-apt-repository ppa:openjdk-r/ppa && apt update -y && \
    apt install -y openjdk-8-jdk

# 验证 Java 版本 (必须输出 1.8.x 或 openjdk version "1.8...")
java -version

sudo apt install -y xvfb xserver-xephyr tightvncserver python3-opengl ffmpeg

pip install "pip<24.0"
pip install "setuptools<60" "wheel<0.40"

pip install gym==0.21.0 --no-build-isolation

pip install minedojo

pip install "opencv-python<4.10"
pip install "numpy>=1.20,<2.0"
pip install "imageio==2.28"
pip install "imageio-ffmpeg==0.4.8"

cd /workspace/
git clone https://github.com/verityw/MixinGradle-dcfaf61 

cd /workspace/wm-lab2

buildscript {
    repositories {

        maven { url 'https://jitpack.io' }
        jcenter()
        mavenCentral()
        maven {
            url "file:/workspace/" // Local directory where the repository was cloned
        }
        maven {
            name = "forge"
            url = "https://maven.minecraftforge.net/"
        }
        maven {
            name = "sonatype"
            url = "https://oss.sonatype.org/content/repositories/snapshots/"
        }
    }
    dependencies {
        classpath 'org.ow2.asm:asm:6.0'
        // classpath('com.github.SpongePowered:MixinGradle:dcfaf61'){ // 0.6
        //     // Because forgegradle requires 6.0 (not -debug-all) while mixinGradle depends on 5.0
        //     // and putting mixin here places it before forge in the class loader
        //     exclude group: 'org.ow2.asm', module: 'asm-debug-all'
        // }
        classpath('MixinGradle-dcfaf61:MixinGradle:dcfaf61'){ // 0.6
            // Because forgegradle requires 6.0 (not -debug-all) while mixinGradle depends on 5.0
            // and putting mixin here places it before forge in the class loader
            exclude group: 'org.ow2.asm', module: 'asm-debug-all'
        }

        classpath 'com.github.brandonhoughton:ForgeGradle:FG_2.2_patched-SNAPSHOT'
    }
}