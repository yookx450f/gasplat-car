#!/bin/bash
# gasplat-carを実行し、出力ファイルをローカルに取得するスクリプト

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "gasplat-car 実行 & 出力取得スクリプト"
echo "========================================"

# 1. Dockerコンテナを停止・削除
echo "[1/4] 既存のコンテナを停止..."
cd "$PROJECT_DIR/docker"
docker compose down 2>/dev/null || true

# 2. 出力ディレクトリの権限を修正
echo "[2/4] 出力ディレクトリの権限を修正..."
mkdir -p "$PROJECT_DIR/output"
chmod -R 777 "$PROJECT_DIR/output" 2>/dev/null || true

# 3. コンテナを実行（srcをマウント）
echo "[3/4] コンテナを起動して処理を実行..."
docker run --rm --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e QT_QPA_PLATFORM=offscreen \
  -v "$PROJECT_DIR/data:/workspace/data:rw" \
  -v "$PROJECT_DIR/output:/workspace/output:rw" \
  -v "$PROJECT_DIR/config:/workspace/config:ro" \
  -v "$PROJECT_DIR/src:/workspace/src:ro" \
  docker-gsplat-car:latest \
  python3 src/main.py --config config/config.yaml

# 4. 出力ファイルを確認
echo ""
echo "[4/4] 出力ファイルを確認..."
echo "========================================"
echo "ローカルのoutputディレクトリ:"
echo "========================================"
ls -la "$PROJECT_DIR/output/" 2>/dev/null || echo "出力ファイルはありません。"

# 5. 結果を表示
echo ""
echo "========================================"
echo "処理完了！"
echo "========================================"
echo ""
echo "3Dモデルを確認するには、以下のソフトウェアを使用してください:"
echo "  - GLBファイル: Microsoft 3D Viewer、Babylon.js Sandbox、またはOnline 3D Viewer"
echo "    URL: https://sandbox.babylonjs.com/"
echo "  - OBJファイル: Blender、MeshLab、またはUnity"
echo ""
echo "オンラインで確認する場合:"
echo "  1. https://3dviewer.net/ にアクセス"
echo "  2. GLBまたはOBJファイルをドラッグ＆ドロップ"
