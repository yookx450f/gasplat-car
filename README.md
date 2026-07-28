# gasplat-car

車の複数画像から3Dモデル（`.glb`, `.obj`）を生成するツール。

NVIDIA公式の [gsplat](https://github.com/NVlabs/gsplat) を使用し、 Gaussian Splatting による高品質な3D再構築を実現する。

## 特徴

- **見た目のリアルさ**: Gaussian Splatting により、車の質感を忠実に再現
- **自動生成**: 複数視点の画像から自動的に3Dモデルを生成
- **複数出力形式**: `.glb` および `.obj` 形式に対応

## 要件

- Windows 10/11 with WSL2
- NVIDIA GPU (RTX3090推奨)
- Docker & Docker Compose
- NVIDIA Container Toolkit

## インストール

### 1. Dockerイメージの取得

```bash
docker pull nvidia/gsplat:latest
```

## 使用方法

### 入力画像の準備

`data/input/` ディレクトリに車の写真（8枚推奨）を置く。

```
data/input/
├── 01.jpg
├── 02.jpg
├── 03.jpg
├── 04.jpg
├── 05.jpg
├── 06.jpg
├── 07.jpg
└── 08.jpg
```

### 実行

```bash
# Docker Composeで実行（推奨）
docker compose up

# 直接Pythonを実行する場合（ローカル環境）
python src/main.py --config config/config.yaml
```

### コマンドラインオプション

```bash
python src/main.py --help

# 設定ファイルの指定
python src/main.py --config config/config.yaml

# 入力ディレクトリの指定
python src/main.py --input-dir ./my_car_images

# 出力ディレクトリの指定
python src/main.py --output-dir ./my_output
```

## 処理フロー

```
入力画像 (8枚)
    │
    ▼
画像前処理 (リサイズ、カメラパラメータ抽出)
    │
    ▼
COLMAP (Camera Pose推定)
    │
    ▼
Gaussian Splatting (3D再構築)
    │
    ▼
メッシュ化 (Poisson Reconstruction)
    │
    ▼
出力 (.glb, .obj)
```

## 設定ファイル

`config/config.yaml` で各種パラメータを調整できる。

```yaml
input:
  image_dir: "data/input"       # 入力画像ディレクトリ
  num_images: 8                 # 画像数
  max_image_size: 1024          # 最大解像度

training:
  num_iterations: 30000         # 訓練イテレーション数
  seed: 42                      # 乱数シード

meshing:
  poisson_depth: 9              # ポアソン再構成の深さ
```

## 出力

`output/` ディレクトリに以下のファイルが生成される。

```
output/
├── car_model.glb      # GLB形式の3Dモデル
├── car_model.obj      # OBJ形式の3Dモデル
└── gsplat/            # 中間データ
```

## License

MIT
