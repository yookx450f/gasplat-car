# ベースイメージ: PyTorch GPU対応（CUDA 12.1 - devel版でCUDA toolkitを含む）
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

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
    libx11-6 \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# gsplatのインストール（CUDA対応）
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    opencv-python-headless \
    trimesh \
    pyyaml \
    configargparse \
    Pillow \
    packaging \
    scipy \
    gsplat

# ワークディレクトリの設定
WORKDIR /workspace

# デフォルトコマンド
CMD ["bash"]
