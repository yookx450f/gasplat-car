# ベースイメージ: NVIDIA gsplat公式イメージ
FROM nvidia/gsplat:latest

# 環境変数の設定
ENV PYTHONPATH="/workspace:$PYTHONPATH"
ENV DEBIAN_FRONTEND=noninteractive

# 必要なシステムパッケージのインストール（COLMAP用）
RUN apt-get update && apt-get install -y \
    wget \
    git \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# COLMAPのインストール
RUN wget https://github.com/colmap/colmap/releases/download/3.3.1/colmap-3.3.1-linux.run && \
    chmod +x colmap-3.3.1-linux.run && \
    ./colmap-3.3.1-linux.run --prefix /usr/local --notimestamp --yes && \
    rm colmap-3.3.1-linux.run

# 追加のPythonパッケージをインストール（gsplat公式イメージに含まれていないもの）
RUN pip install --no-cache-dir \
    opencv-python-headless \
    trimesh \
    pyyaml \
    configargparse \
    Pillow

# ワークディレクトリの設定
WORKDIR /workspace

# デフォルトコマンド
CMD ["bash"]
