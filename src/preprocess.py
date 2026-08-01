#!/usr/bin/env python3
"""
画像前処理モジュール

入力画像をリサイズ・正規化し、EXIF情報からカメラパラメータを抽出する。
"""

import os
import glob
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import cv2


class ImagePreprocessor:
    """画像前処理クラス"""
    
    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            設定ファイル
        """
        self.config = config
        self.image_dir = Path(config['input']['image_dir'])
        self.max_size = config['input'].get('max_image_size', 1024)
        self.output_dir = self.image_dir / 'processed'
        self.output_dir.mkdir(exist_ok=True)
    
    def _get_image_files(self) -> List[Path]:
        """入力画像ファイルを取得（ファイル名でソート、重複除去）"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = set()
        for ext in extensions:
            image_files.update(glob.glob(str(self.image_dir / ext)))
            image_files.update(glob.glob(str(self.image_dir / ext.upper())))
        
        # ファイル名でソート（01.jpg, 02.jpg...の順）
        sorted_files = sorted(list(image_files))
        print(f"    [Preprocess] 入力画像 {len(sorted_files)}枚を検出: {[f.split('/')[-1] for f in sorted_files]}")
        return [Path(f) for f in sorted_files]
    
    def _resize_image(self, img: Image.Image, max_size: int) -> Image.Image:
        """画像をリサイズ（アスペクト比保持）"""
        width, height = img.size
        
        if max(width, height) <= max_size:
            return img
        
        # アスペクト比を維持してリサイズ
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # LANCZOSフィルタを使用して高品質にリサイズ
        return img.resize((new_width, new_height), Image.LANCZOS)
    
    def _enhance_image_for_feature_detection(self, img: Image.Image) -> Image.Image:
        """
        特徴点検出用に画像を強化
        
        コントラスト強化とエッジ強調を行い、SIFT特徴点を検出しやすくする。
        
        Parameters
        ----------
        img : Image.Image
            入力画像
        
        Returns
        -------
        Image.Image
            処理後の画像
        """
        try:
            # NumPy配列に変換
            img_array = np.array(img, dtype=np.float32)
            
            # RGBをグレースケールに変換
            gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
            
            # 統計量を計算
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            
            if std_val < 10:  # コントラストが非常に低い場合
                # アダプティブなコントラスト強化
                # ヒストグラム均等化を適用
                gray_uint8 = ((gray - np.min(gray)) / (np.max(gray) - np.min(gray) + 1e-8) * 255).astype(np.uint8)
                
                # CLAHE（Contrast Limited Adaptive Histogram Equalization）
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_gray = clahe.apply(gray_uint8)
                
                # 元の画像の色情報を保持
                enhanced_array = np.stack([enhanced_gray] * 3, axis=-1).astype(np.uint8)
                return Image.fromarray(enhanced_array, 'RGB')
            
            # 通常のコントラスト強化
            if std_val < 50:
                # コントラストを強化
                factor = 50.0 / max(std_val, 1.0)
                gray = (gray - mean_val) * factor + mean_val
            
            # エッジ強調（アンシャープマスキング）
            ksize = 3  # 奇数のみ許可
            gray_blur = cv2.GaussianBlur(gray.astype(np.uint8), (ksize, ksize), 0)
            enhanced = gray + 1.5 * (gray - gray_blur)
            
            # 値域を0-255にクリップ
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            # 元の画像の色情報を復元（コントラスト強化を適用）
            img_array = np.array(img, dtype=np.float32)
            
            # グレースケールの強さをRGBに適用
            ratio = 255.0 / max(np.max(enhanced) - np.min(enhanced), 1.0)
            enhanced_rgb = img_array.copy()
            for c in range(3):
                channel = img_array[:,:,c].astype(np.float32)
                channel_enhanced = (channel - np.mean(channel)) * ratio + np.mean(channel)
                enhanced_rgb[:,:,c] = channel_enhanced
            
            enhanced_rgb = np.clip(enhanced_rgb, 0, 255).astype(np.uint8)
            return Image.fromarray(enhanced_rgb, 'RGB')
            
        except Exception as e:
            print(f"  警告: 画像強化に失敗しました ({e})。元の画像を使用します。")
            return img
    
    def _extract_camera_params(self, img_path: Path, resized_width: int, resized_height: int) -> Dict:
        """
        EXIF情報からカメラパラメータを抽出
        
        Parameters
        ----------
        img_path : Path
            画像ファイルパス
        resized_width : int
            リサイズ後の幅
        resized_height : int
            リサイズ後の高さ
        
        Returns
        -------
        dict
            カメラパラメータ
        """
        params = {
            'width': resized_width,
            'height': resized_height,
            'focal_length_px': None,
            'principal_point': None,
            'distortion_coeffs': []
        }
        
        try:
            from PIL.ExifTags import TAGS, GPSTAGS
            
            img = Image.open(img_path)
            exif_data = img._getexif()
            
            if exif_data:
                # 焦点距離の取得（35mm判換算）
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    
                    if tag_name == 'FocalLength':
                        # 焦点距離（mm）
                        focal_length_mm = value
                        
                    elif tag_name == 'ImageWidth':
                        original_width = value
                    elif tag_name == 'ImageHeight':
                        original_height = value
                    elif tag_name == 'PixelXDimension':
                        params['sensor_width_px'] = value
                    elif tag_name == 'PixelYDimension':
                        params['sensor_height_px'] = value
            
            # EXIFから取得できない場合は、簡易推定
            # 一般的なスチルカメラのセンサーサイズを仮定
            sensor_width_mm = 6.17  # 1インチセンサー相当
            sensor_height_mm = sensor_width_mm * (resized_height / resized_width)
            
            # 焦点距離（mm）から focal length in pixels を計算
            # 一般的な焦点距離 25-50mm を仮定（問題があればEXIFで上書き）
            assumed_focal_length_mm = 35.0
            
            # focal length in pixels = (focal length in mm / sensor width in mm) * image width in pixels
            params['focal_length_px'] = (assumed_focal_length_mm / sensor_width_mm) * resized_width
            params['principal_point'] = [resized_width / 2, resized_height / 2]
            params['assumed'] = True  # EXIFから取得できなかった場合のフラグ
            
        except Exception as e:
            print(f"警告: カメラパラメータの抽出に失敗しました ({img_path}): {e}")
            # デフォルト値を設定
            params['focal_length_px'] = resized_width * 1.2
            params['principal_point'] = [resized_width / 2, resized_height / 2]
            params['assumed'] = True
        
        return params
    
    def process(self) -> Dict:
        """
        画像の前処理を実行
        
        Returns
        -------
        dict
            処理結果（画像パス、カメラパラメータなど）
        """
        print(f"入力ディレクトリ: {self.image_dir}")
        
        image_files = self._get_image_files()
        print(f"画像ファイル数: {len(image_files)}")
        
        if len(image_files) == 0:
            raise ValueError(f"入力ディレクトリ '{self.image_dir}' に画像が見つかりません。")
        
        processed_images = []
        camera_params_list = []
        
        for i, img_path in enumerate(image_files):
            print(f"  処理中: {img_path.name}...")
            
            # 画像を開く
            img = Image.open(img_path).convert('RGB')
            
            # 特徴点検出用に画像を強化
            enhanced_img = self._enhance_image_for_feature_detection(img)
            
            # リサイズ
            resized_img = self._resize_image(enhanced_img, self.max_size)
            width, height = resized_img.size
            
            # 保存
            output_path = self.output_dir / img_path.name
            resized_img.save(str(output_path), 'JPEG', quality=95)
            
            # カメラパラメータ抽出
            cam_params = self._extract_camera_params(img_path, width, height)
            
            processed_images.append(str(output_path))
            camera_params_list.append(cam_params)
            
            print(f"    -> {output_path} ({width}x{height})")
        
        # 結果をJSONとして保存
        metadata = {
            'images': processed_images,
            'camera_params': camera_params_list,
            'num_images': len(processed_images),
            'max_size': self.max_size
        }
        
        metadata_path = self.output_dir / 'camera_params.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\n処理結果を保存: {metadata_path}")
        
        return {
            'image_paths': processed_images,
            'camera_params': camera_params_list,
            'metadata_path': str(metadata_path)
        }
