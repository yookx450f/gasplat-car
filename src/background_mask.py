#!/usr/bin/env python3
"""
背景透過モジュール

rembg を使って、画像から背景を除去し車を切り抜く。
出力は透明なPNGまたは白背景のJPEG。

フェーズ1：入力データの最適化（前処理）
ステップ1：AIによる「車全体」の背景透過（Masking）の自動化
"""

import os
import glob
import json
from pathlib import Path
from typing import List, Dict
from PIL import Image, ImageDraw
import numpy as np
import rembg


class BackgroundMaskProcessor:
    """背景透過処理クラス"""

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            設定ファイル
        """
        self.config = config
        self.input_dir = Path(config['input']['masked_input_dir'])
        self.output_dir = Path(config['input']['masked_output_dir'])
        self.output_format = config['input'].get('masked_output_format', 'transparent')  # 'transparent' or 'white'
        self.engine = config['input'].get('rembg_engine', 'u2net')  # rembg エンジン
        
        # 出力ディレクトリを作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # rembg の初期化
        self._init_rembg()

    def _init_rembg(self):
        """rembg を初期化"""
        try:
            # rembg のモデルをロード（バックグラウンドでダウンロードされる場合あり）
            rembg.new_session(f"u2net")  # デフォルトエンジン
            print("[BackgroundMaskProcessor] rembg エンジン 'u2net' を初期化しました。")
        except Exception as e:
            print(f"[BackgroundMaskProcessor] rembg 初期化警告: {e}")
            print("  - 初回実行時はモデルをダウンロードしています。数分待つことがあります。")

    def _get_image_files(self) -> List[Path]:
        """入力画像ファイルを取得（ファイル名でソート）"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = set()
        for ext in extensions:
            image_files.update(glob.glob(str(self.input_dir / ext)))
            image_files.update(glob.glob(str(self.input_dir / ext.upper())))
        
        # ファイル名でソート
        sorted_files = sorted(list(image_files))
        print(f"[BackgroundMaskProcessor] 入力画像 {len(sorted_files)}枚を検出: {[Path(f).name for f in sorted_files]}")
        return [Path(f) for f in sorted_files]

    def _remove_background(self, img: Image.Image) -> Image.Image:
        """
        背景を除去
        
        Parameters
        ----------
        img : Image.Image
            入力画像（RGB）
        
        Returns
        -------
        Image.Image
            背景透過画像（RGBA）
        """
        try:
            # rembg で背景を除去
            # rembg.remove() は PIL Image または numpy array を受け取る
            result = rembg.remove(img, session=rembg.new_session("u2net"))
            return result
        except Exception as e:
            print(f"[BackgroundMaskProcessor] 背景除去エラー: {e}")
            # エラー時は元の画像を返す（RGBAに変換）
            return img.convert('RGBA')

    def _transparent_to_white_background(self, rgba_img: Image.Image) -> Image.Image:
        """
        透明画像を白背景の画像に変換
        
        Parameters
        ----------
        rgba_img : Image.Image
            RGBA画像
        
        Returns
        -------
        Image.Image
            白背景のRGB画像
        """
        # 白背景を作成
        white_bg = Image.new('RGBA', rgba_img.size, (255, 255, 255, 255))
        # 透過画像を白背景に合成
        white_bg.paste(rgba_img, (0, 0), rgba_img)
        # RGBに変換
        return white_bg.convert('RGB')

    def process(self) -> Dict:
        """
        背景透過処理を実行
        
        Returns
        -------
        dict
            処理結果（画像パスなど）
        """
        print(f"\n=== フェーズ1 ステップ1：背景透過処理 ===")
        print(f"入力ディレクトリ: {self.input_dir}")
        print(f"出力ディレクトリ: {self.output_dir}")
        print(f"出力形式: {self.output_format}")
        
        image_files = self._get_image_files()
        
        if len(image_files) == 0:
            raise ValueError(f"入力ディレクトリ '{self.input_dir}' に画像が見つかりません。")
        
        processed_images = []
        processing_log = []
        
        for i, img_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] 処理中: {img_path.name}...")
            
            try:
                # 画像を開く
                img = Image.open(img_path).convert('RGB')
                original_size = img.size
                
                # 背景を除去
                rgba_img = self._remove_background(img)
                
                # 出力形式に応じて処理
                if self.output_format == 'white':
                    # 白背景のJPEGとして保存
                    rgb_img = self._transparent_to_white_background(rgba_img)
                    output_path = self.output_dir / img_path.with_suffix('.jpg').name
                    rgb_img.save(str(output_path), 'JPEG', quality=95)
                else:
                    # 透明PNGとして保存
                    output_path = self.output_dir / img_path.with_suffix('.png').name
                    rgba_img.save(str(output_path), 'PNG')
                
                processed_images.append(str(output_path))
                
                print(f"  -> {output_path.name} ({rgba_img.size[0]}x{rgba_img.size[1]})")
                
                # ログに記録
                processing_log.append({
                    'input': str(img_path),
                    'output': str(output_path),
                    'original_size': list(original_size),
                    'output_size': list(rgba_img.size),
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"  -> エラー: {img_path.name} - {e}")
                processing_log.append({
                    'input': str(img_path),
                    'output': None,
                    'status': 'error',
                    'error': str(e)
                })
        
        # 結果をJSONとして保存
        metadata = {
            'processed_images': processed_images,
            'num_processed': len(processed_images),
            'total_input': len(image_files),
            'output_format': self.output_format,
            'rembg_engine': self.engine,
            'processing_log': processing_log
        }
        
        metadata_path = self.output_dir / 'masking_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 処理完了 ===")
        print(f"処理済み画像数: {len(processed_images)}/{len(image_files)}")
        print(f"メタデータ保存先: {metadata_path}")
        
        return metadata


def main():
    """メイン関数"""
    import configargparse
    
    parser = configargparse.ArgumentParser(description='背景透過処理（rembg）')
    parser.add('-c', '--config', is_config_file=True, help='設定ファイルパス')
    parser.add_argument('--input-dir', type=str, default='data/1raw_images',
                        help='入力画像ディレクトリ')
    parser.add_argument('--output-dir', type=str, default='data/2masked_images',
                        help='出力画像ディレクトリ')
    parser.add_argument('--format', type=str, default='transparent', choices=['transparent', 'white'],
                        help='出力形式: transparent(透明PNG) または white(白背景JPEG)')
    parser.add_argument('--engine', type=str, default='u2net',
                        help='rembg エンジン (u2net, u2netp, u2net_humans, etc.)')
    
    args = parser.parse_args()
    
    # 設定
    config = {
        'input': {
            'masked_input_dir': args.input_dir,
            'masked_output_dir': args.output_dir,
            'masked_output_format': args.format,
            'rembg_engine': args.engine,
        }
    }
    
    # 処理実行
    processor = BackgroundMaskProcessor(config)
    result = processor.process()
    
    print("\n=== 完了 ===")
    print(f"処理結果: {result['num_processed']}/{result['total_input']} 枚成功")


if __name__ == '__main__':
    main()
