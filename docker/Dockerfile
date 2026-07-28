# ベースイメージ: PyTorch GPU対応
FROM pytorch/pytorch:latest

# 環境変数の設定
ENV PYTHONPATH="/workspace:$PYTHONPATH"
ENV DEBIAN_FRONTEND=noninteractive

# 必要なシステムパッケージのインストール（COLMAP含む）
RUN apt-get update && apt-get install -y \
    wget \
    git \
    cmake \
    colmap \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# gsplatのインストール（CUDA対応）
RUN pip install --no-cache-dir \
    gsplat \
    opencv-python-headless \
    trimesh \
    pyyaml \
    configargparse \
    Pillow

# ワークディレクトリの設定
WORKDIR /workspace

# デフォルトコマンド
CMD ["bash"]
