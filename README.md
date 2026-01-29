# wm-lab2
# ==========================================
# MineDojo & World Model Data Collection 
# 环境安装指南 (Tested 2026)
# ==========================================

# 1. 准备工作：系统级依赖 (Ubuntu/Debian)
# 如果你在服务器上跑（无显示器），xvfb 是必须的
sudo apt-get update
sudo apt-get install -y xvfb openjdk-8-jdk libgl1-mesa-glx

# 2. 创建干净的 Conda 环境
# 注意：强烈建议使用 Python 3.9，这是 MineDojo 最稳定的版本
conda create -n mine_env python=3.9 -y
conda activate mine_env

# 3. 安装 Java 8 (关键步骤)
# MineDojo 必须使用 Java 8，不能用 Java 11/17/21
conda install -c conda-forge openjdk=8 -y

# 验证 Java 版本 (必须输出 1.8.x 或 openjdk version "1.8...")
java -version

# 4. “时光倒流”：降级构建工具 (关键步骤)
# 解决 Gym 0.21.0 的 "metadata" 和 "setup" 错误
# 必须把 pip 降级到 24.0 以下，否则无法安装 gym
pip install "pip<24.0"
pip install "setuptools<60" "wheel<0.40"

# 5. 手动安装 Gym 0.21.0
# 使用 --no-build-isolation 确保使用我们降级过的 setuptools
pip install gym==0.21.0 --no-build-isolation

# 6. 安装 MineDojo 主包
pip install minedojo

# 7. 安装数据采集