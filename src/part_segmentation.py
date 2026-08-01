#!/usr/bin/env python3
"""
パーツ別セグメンテーションモジュール

SAM2 (Segment Anything Model 2) を使って、車写真を「窓ガラス」「ホイール」「ボディ」等にセグメント化。
白黒マスク画像を出力する。

フェーズ1：入力データの最適化（前処理）
ステップ2：AIによる「パーツ別」セグメンテーション
"""

import os
import glob
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw
import cv2


class PartSegmentationProcessor:
    """パーツ別セグメンテーション処理クラス"""

    # サポートされるパーツ定義
    PARTS = {
        'glass': {'name': '窓ガラス', 'color': (255, 255, 255), 'description': '窓ガラス部分'},
        'wheels': {'name': 'ホイール', 'color': (255, 255, 255), 'description': 'タイヤ・ホイール部分'},
        'body': {'name': 'ボディ', 'color': (255, 255, 255), 'description': '車体部分'},
        'other': {'name': 'その他', 'color': (255, 255, 255), 'description': 'その他の部分（ライト、グリル等）'},
    }

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            設定ファイル
        """
        self.config = config
        self.input_dir = Path(config['input']['part_masks_input_dir'])
        self.output_base_dir = Path(config['input']['part_masks_output_dir'])
        self.image_size = config['input'].get('part_image_size', None)  # Noneで元サイズを維持
        
        # 出力ディレクトリを作成
        for part_name in self.PARTS:
            (self.output_base_dir / part_name).mkdir(parents=True, exist_ok=True)
        
        # SAM2の初期化（可能であれば）
        self.sam2_model = None
        self._init_sam2()

    def _init_sam2(self):
        """SAM2を初期化"""
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.sam2_model = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
            print("[PartSegmentationProcessor] SAM2モデルをロードしました。")
        except ImportError:
            print("[PartSegmentationProcessor] SAM2がインストールされていません。ヒューリスティック処理を使用します。")
            self.sam2_enabled = False
        except Exception as e:
            print(f"[PartSegmentationProcessor] SAM2初期化エラー: {e}。ヒューリスティック処理を使用します。")
            self.sam2_enabled = False

    def _get_image_files(self) -> List[Path]:
        """入力画像ファイルを取得（ファイル名でソート）"""
        extensions = ['*.png', '*.jpg', '*.jpeg']
        image_files = set()
        for ext in extensions:
            image_files.update(glob.glob(str(self.input_dir / ext)))
        
        sorted_files = sorted(list(image_files))
        print(f"[PartSegmentationProcessor] 入力画像 {len(sorted_files)}枚を検出: {[Path(f).name for f in sorted_files]}")
        return [Path(f) for f in sorted_files]

    def _segment_with_sam2(self, img: Image.Image, prompts: Dict[str, List[Tuple[int, int]]]) -> Dict[str, np.ndarray]:
        """
        SAM2を使ってセグメンテーション
        
        Parameters
        ----------
        img : Image.Image
            入力画像
        prompts : dict
            パーツ名 -> ポイント座標のリスト
            
        Returns
        -------
        dict
            パーツ名 -> マスク（numpy配列）
        """
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import torch
            
            # 画像をnumpy配列に変換
            img_array = np.array(img)
            
            # セットアップ
            self.sam2_model.set_image(img_array)
            
            masks = {}
            for part_name, points in prompts.items():
                if not points:
                    masks[part_name] = np.zeros((img.height, img.width), dtype=np.uint8)
                    continue
                
                # ポイントをnumpy配列に変換
                input_points = np.array(points, dtype=np.float32)
                
                # SAM2でセグメンテーション
                masks_input = {
                    "point_coords": input_points,
                    "point_labels": np.ones(len(input_points), dtype=np.int32),
                }
                
                with torch.no_grad():
                    output = self.sam2_model.predict(
                        image=img_array,
                        point_coords=input_points if len(input_points) > 0 else None,
                        point_labels=np.ones(len(input_points), dtype=np.int32) if len(input_points) > 0 else None,
                        multimask_output=True,
                    )
                
                # 最良のマスクを選択
                mask = output.masks[output.iou_preds.argmax()]
                masks[part_name] = (mask * 255).astype(np.uint8)
            
            return masks
            
        except Exception as e:
            print(f"[PartSegmentationProcessor] SAM2セグメンテーションエラー: {e}")
            return None

    def _segment_heuristic(self, img: Image.Image) -> Dict[str, np.ndarray]:
        """
        ヒューリスティックな手法でパーツをセグメント化
        
        SAM2が利用できない場合のフォールバック。
        色の特性を使って各パーツを推測する。
        
        Parameters
        ----------
        img : Image.Image
            入力画像
            
        Returns
        -------
        dict
            パーツ名 -> マスク（numpy配列）
        """
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        masks = {}
        
        # RGBをHSVに変換
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # 窓ガラスのセグメンテーション
        # ガラスは青みがかった色で、明るい領域として検出
        lower_blue = np.array([100, 40, 150])
        upper_blue = np.array([130, 255, 255])
        glass_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # モルフォロジー処理でノイズ除去
        kernel = np.ones((5, 5), np.uint8)
        glass_mask = cv2.dilate(glass_mask, kernel, iterations=2)
        glass_mask = cv2.erode(glass_mask, kernel, iterations=1)
        
        masks['glass'] = glass_mask
        
        # ホイールのセグメンテーション
        # 黒に近い色として検出（下部領域に集中）
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([30, 50, 80])
        wheel_mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # 画像の下部60%に限定（ホイールは通常下部にある）
        mask_roi = np.zeros_like(wheel_mask)
        mask_roi[int(h * 0.3):, :] = 255
        wheel_mask = cv2.bitwise_and(wheel_mask, wheel_mask, mask=mask_roi)
        
        # モルフォロジー処理
        wheel_mask = cv2.dilate(wheel_mask, kernel, iterations=3)
        wheel_mask = cv2.erode(wheel_mask, kernel, iterations=2)
        
        masks['wheels'] = wheel_mask
        
        # ボディのセグメンテーション
        # 車体の色は多様なので、ガラスとホイール以外の領域をボディとして扱う
        body_mask = np.ones((h, w), dtype=np.uint8) * 255
        body_mask = cv2.bitwise_and(body_mask, body_mask, mask=cv2.bitwise_not(glass_mask))
        body_mask = cv2.bitwise_and(body_mask, body_mask, mask=cv2.bitwise_not(wheel_mask))
        
        masks['body'] = body_mask
        
        # その他のパーツ（ライト等）
        # 明るい白色/黄色の領域として検出
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([30, 30, 255])
        light_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # ガラスとボディの領域を除外
        light_mask = cv2.bitwise_and(light_mask, light_mask, mask=cv2.bitwise_not(glass_mask))
        
        masks['other'] = light_mask
        
        return masks

    def _create_prompts_for_sam2(self, img: Image.Image) -> Dict[str, List[Tuple[int, int]]]:
        """
        SAM2用のプロンプト（ポイント）を自動生成
        
        Parameters
        ----------
        img : Image.Image
            入力画像
            
        Returns
        -------
        dict
            パーツ名 -> ポイント座標のリスト
        """
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        prompts = {}
        
        # 窓ガラスのポイントを推定（画像の上部中央付近）
        glass_points = []
        # 前窓ガラスの位置を推定
        glass_y_start = int(h * 0.15)
        glass_y_end = int(h * 0.45)
        glass_x_start = int(w * 0.2)
        glass_x_end = int(w * 0.8)
        
        # グリッドでポイントを配置
        for y in range(glass_y_start, glass_y_end, max(1, (glass_y_end - glass_y_start) // 3)):
            for x in range(glass_x_start, glass_x_end, max(1, (glass_x_end - glass_x_start) // 3)):
                glass_points.append((x, y))
        prompts['glass'] = glass_points
        
        # ホイールのポイントを推定（画像の下部）
        wheel_points = []
        # 左前輪
        wheel_y_start = int(h * 0.6)
        wheel_y_end = int(h * 0.9)
        wheel_x_left_start = int(w * 0.1)
        wheel_x_left_end = int(w * 0.3)
        
        for y in range(wheel_y_start, wheel_y_end, max(1, (wheel_y_end - wheel_y_start) // 2)):
            for x in range(wheel_x_left_start, wheel_x_left_end, max(1, (wheel_x_left_end - wheel_x_left_start) // 2)):
                wheel_points.append((x, y))
        
        # 右前輪
        wheel_x_right_start = int(w * 0.7)
        wheel_x_right_end = int(w * 0.9)
        
        for y in range(wheel_y_start, wheel_y_end, max(1, (wheel_y_end - wheel_y_start) // 2)):
            for x in range(wheel_x_right_start, wheel_x_right_end, max(1, (wheel_x_right_end - wheel_x_right_start) // 2)):
                wheel_points.append((x, y))
        
        prompts['wheels'] = wheel_points
        
        # ボディのポイントを推定（ガラスとホイールの間）
        body_points = []
        body_y_start = int(h * 0.45)
        body_y_end = int(h * 0.6)
        body_x_start = int(w * 0.1)
        body_x_end = int(w * 0.9)
        
        for y in range(body_y_start, body_y_end, max(1, (body_y_end - body_y_start) // 2)):
            for x in range(body_x_start, body_x_end, max(1, (body_x_end - body_x_start) // 3)):
                body_points.append((x, y))
        
        prompts['body'] = body_points
        
        # その他のポイント（ライト等）
        other_points = []
        # ヘッドライトの位置を推定（画像の両端、下部）
        headlight_y = int(h * 0.5)
        for x in range(int(w * 0.05), int(w * 0.15)):
            other_points.append((x, headlight_y))
        for x in range(int(w * 0.85), int(w * 0.95)):
            other_points.append((x, headlight_y))
        
        prompts['other'] = other_points
        
        return prompts

    def process(self) -> Dict:
        """
        パーツ別セグメンテーションを実行
        
        Returns
        -------
        dict
            処理結果
        """
        print(f"\n=== フェーズ1 ステップ2：パーツ別セグメンテーション ===")
        print(f"入力ディレクトリ: {self.input_dir}")
        print(f"出力ディレクトリ: {self.output_base_dir}")
        
        image_files = self._get_image_files()
        
        if len(image_files) == 0:
            raise ValueError(f"入力ディレクトリ '{self.input_dir}' に画像が見つかりません。")
        
        processing_log = []
        
        for i, img_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] 処理中: {img_path.name}...")
            
            try:
                # 画像を開く
                img = Image.open(img_path).convert('RGB')
                
                if self.sam2_model is not None:
                    # SAM2を使用
                    print("  - SAM2セグメンテーションを実行中...")
                    prompts = self._create_prompts_for_sam2(img)
                    masks = self._segment_with_sam2(img, prompts)
                    
                    if masks is None:
                        print("  - SAM2失敗、ヒューリスティック処理にフォールバック")
                        masks = self._segment_heuristic(img)
                else:
                    # ヒューリスティック処理を使用
                    print("  - ヒューリスティックセグメンテーションを実行中...")
                    masks = self._segment_heuristic(img)
                
                # マスクを保存
                output_paths = {}
                for part_name, mask in masks.items():
                    # PIL画像に変換
                    mask_pil = Image.fromarray(mask, mode='L')
                    
                    # 出力パス
                    output_path = self.output_base_dir / part_name / img_path.with_suffix('.png').name
                    mask_pil.save(str(output_path), 'PNG')
                    output_paths[part_name] = str(output_path)
                    
                    print(f"  - {self.PARTS[part_name]['name']}: {output_path.name}")
                
                processing_log.append({
                    'input': str(img_path),
                    'masks': output_paths,
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"  - エラー: {img_path.name} - {e}")
                processing_log.append({
                    'input': str(img_path),
                    'status': 'error',
                    'error': str(e)
                })
        
        # メタデータを保存
        metadata = {
            'num_processed': len([l for l in processing_log if l['status'] == 'success']),
            'total_input': len(image_files),
            'parts': {k: v['name'] for k, v in self.PARTS.items()},
            'processing_log': processing_log
        }
        
        metadata_path = self.output_base_dir / 'segmentation_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 処理完了 ===")
        print(f"処理済み画像数: {metadata['num_processed']}/{metadata['total_input']}")
        print(f"メタデータ保存先: {metadata_path}")
        
        return metadata


def main():
    """メイン関数"""
    import configargparse
    
    parser = configargparse.ArgumentParser(description='パーツ別セグメンテーション（SAM2）')
    parser.add_argument('--input-dir', type=str, default='data/2masked_images',
                        help='入力画像ディレクトリ（ステップ1の出力）')
    parser.add_argument('--output-dir', type=str, default='data/3part_masks',
                        help='出力ディレクトリ')
    parser.add_argument('--use-sam2', action='store_true',
                        help='SAM2を使用する（インストールされている場合）')
    
    args = parser.parse_args()
    
    # 設定
    config = {
        'input': {
            'part_masks_input_dir': args.input_dir,
            'part_masks_output_dir': args.output_dir,
            'part_image_size': None,
        }
    }
    
    # 処理実行
    processor = PartSegmentationProcessor(config)
    if args.use_sam2:
        # SAM2を強制有効化（エラー時はフォールバック）
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            processor.sam2_model = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
            print("SAM2を有効にしました。")
        except Exception as e:
            print(f"SAM2の有効化に失敗しました: {e}。ヒューリスティック処理を使用します。")
    
    result = processor.process()
    
    print("\n=== 完了 ===")
    print(f"処理結果: {result['num_processed']}/{result['total_input']} 枚成功")


if __name__ == '__main__':
    main()
