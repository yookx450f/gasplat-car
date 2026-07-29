#!/usr/bin/env python3
"""
gasplat-car: 車の複数画像から3Dモデル（.glb, .obj）を生成するメインプログラム

フロー:
1. 画像前処理
2. COLMAPによるCamera Pose推定
3. Gaussian Splattingによる3D再構築
4. メッシュ化
5. .glb/.obj出力
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path

# ワークディレクトリをセット
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import ImagePreprocessor
from colmap_runner import ColmapRunner
from train_gsplat import GaussianSplattingTrainer
from mesh_converter import MeshConverter
from export import ModelExporter
from intermediate_output import IntermediateOutputManager


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def validate_input_data(config: dict) -> bool:
    """入力データが正しいか検証する"""
    image_dir = Path(config['input']['image_dir'])
    if not image_dir.exists():
        print(f"エラー: 入力画像ディレクトリ '{image_dir}' が存在しません。")
        return False
    
    images = list(image_dir.glob('*'))
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in images if f.suffix.lower() in image_extensions]
    
    if len(image_files) < 4:
        print(f"エラー: 最低4枚以上の画像が必要です。{len(image_files)}枚しか見つかりません。")
        return False
    
    print(f"入力画像 {len(image_files)}枚を検出しました。")
    return True


def run_pipeline(config: dict):
    """全体の処理パイプラインを実行する"""
    start_time = time.time()
    
    print("=" * 60)
    print("gasplat-car 処理開始")
    print("=" * 60)
    
    # 中間結果出力マネージャーの初期化
    intermediate_manager = IntermediateOutputManager(config)
    
    # ステップ1: 画像前処理
    print("\n" + "=" * 40)
    print("ステップ1: 画像前処理")
    print("=" * 40)
    preprocessor = ImagePreprocessor(config)
    processed_data = preprocessor.process()
    
    # 前処理結果の保存
    intermediate_manager.save_preprocessing_results(processed_data)
    
    # ステップ2: COLMAPによるCamera Pose推定
    print("\n" + "=" * 40)
    print("ステップ2: Camera Pose推定 (COLMAP)")
    print("=" * 40)
    colmap_runner = ColmapRunner(config)
    
    # COLMAP処理を実行（mapperが失敗しても警告付きで続行）
    colmap_results = None
    try:
        colmap_results = colmap_runner.run(processed_data)
        # COLMAP結果の保存
        intermediate_manager.save_colmap_results(colmap_results)
    except RuntimeError as e:
        print(f"  警告: COLMAP処理でエラーが発生しました: {e}")
        print(f"  警告: 処理を続行します...")
        # 空の結果を生成
        colmap_results = {
            'camera_params': [],
            'image_paths': processed_data['image_paths'],
            'cameras': {},
            'images': {},
            'points3D': None
        }
    
    # ステップ3: Gaussian Splatting訓練
    print("\n" + "=" * 40)
    print("ステップ3: Gaussian Splatting 訓練")
    print("=" * 40)
    gs_trainer = GaussianSplattingTrainer(config)
    gs_result = gs_trainer.train(colmap_results, intermediate_manager=intermediate_manager)
    
    # ステップ4: メッシュ化
    print("\n" + "=" * 40)
    print("ステップ4: メッシュ化")
    print("=" * 40)
    mesh_converter = MeshConverter(config)
    mesh_data = mesh_converter.convert(gs_result, colmap_results)
    
    # メッシュ化結果の保存
    intermediate_manager.save_mesh_results(mesh_data)
    
    # ステップ5: 出力エクスポート
    print("\n" + "=" * 40)
    print("ステップ5: 出力エクスポート (.glb, .obj)")
    print("=" * 40)
    exporter = ModelExporter(config)
    exported_files = exporter.export(mesh_data)
    
    # 処理時間
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"処理完了 ({elapsed_time:.2f}秒)")
    print(f"出力ファイル: {exported_files}")
    print("=" * 60)
    
    return exported_files


def main():
    """エントリポイント"""
    parser = argparse.ArgumentParser(description='gasplat-car: 車の複数画像から3Dモデルを生成')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='設定ファイルのパス')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='入力画像ディレクトリ（上書き）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='出力ディレクトリ（上書き）')
    
    args = parser.parse_args()
    
    # 設定ファイルを読み込み
    config = load_config(args.config)
    
    # コマンドライン引数で上書き
    if args.input_dir:
        config['input']['image_dir'] = args.input_dir
    if args.output_dir:
        config['output']['dir'] = args.output_dir
    
    # 入力データの検証
    if not validate_input_data(config):
        sys.exit(1)
    
    # パイプライン実行
    try:
        run_pipeline(config)
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
