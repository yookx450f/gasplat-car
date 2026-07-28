#!/usr/bin/env python3
"""
COLMAP実行モジュール

SIFT特徴量抽出、Feature Matching、Bundle AdjustmentによるCamera Pose推定を実行する。
"""

import os
import subprocess
import shutil
import json
import numpy as np
from pathlib import Path
from typing import Dict, List


class ColmapRunner:
    """COLMAP実行クラス"""
    
    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            設定ファイル
        """
        self.config = config
        self.colmap_config = config.get('colmap', {})
        
        # 出力ディレクトリ
        self.work_dir = Path('data/colmap_work')
        self.db_path = self.work_dir / 'database.db'
        self.images_dir = self.work_dir / 'images'
        self.sparse_dir = self.work_dir / 'sparse'
        
        # 作業ディレクトリを作成
        self.work_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.sparse_dir.mkdir(exist_ok=True)
    
    def _prepare_images(self, processed_data: Dict) -> None:
        """
        処理済み画像をCOLMAP用ディレクトリにコピー
        
        Parameters
        ----------
        processed_data : dict
            前処理の結果
        """
        image_paths = processed_data['image_paths']
        
        # 画像をコピー（COLMAPはファイル名をインデックスとして使用）
        for i, img_path in enumerate(image_paths):
            dest = self.images_dir / f'image_{i:04d}.jpg'
            shutil.copy2(img_path, str(dest))
        
        print(f"  {len(image_paths)}枚の画像をコピーしました。")
    
    def _run_colmap_command(self, command: List[str]) -> subprocess.CompletedProcess:
        """COLMAPコマンドを実行"""
        cmd = ['colmap'] + command
        print(f"  実行: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=str(self.work_dir),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"  エラー: {' '.join(cmd)}")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
            raise RuntimeError(f"COLMAP command failed: {' '.join(cmd)}")
        
        return result
    
    def run_feature_extraction(self) -> None:
        """特徴量抽出を実行"""
        extractor_config = self.colmap_config.get('extractor', {})
        
        max_size = extractor_config.get('max_image_size', 1024)
        camera_model = extractor_config.get('camera_model', 'simple_pinhole')
        num_threads = extractor_config.get('num_threads', 8)
        
        self._run_colmap_command([
            'feature_extractor',
            '--database_path', str(self.db_path),
            '--image_path', str(self.images_dir),
            '--ImageReader.camera_model', camera_model,
            '--ImageReader.single_camera', '1',
            '--ImageReader.camera_params', f'0,{max_size},0,{max_size/2},{max_size/2}',
            '--SiftExtraction.max_num_features', str(extractor_config.get('max_features', 4096)),
            '--SiftExtraction.gpu_index', '-1',  # CPU使用（COLMAPの制限）
            '--SiftExtraction.num_threads', str(num_threads),
            '--SiftExtraction.augmentation', '0'
        ])
    
    def run_feature_matching(self) -> None:
        """特徴量マッチングを実行"""
        matcher_config = self.colmap_config.get('matcher', {})
        matcher_type = matcher_config.get('type', 'exhaustive')
        
        if matcher_type == 'exhaustive':
            self._run_colmap_command([
                'exhaustive_matcher',
                '--database_path', str(self.db_path),
                '--SiftMatching.max_num_matches', '1024',
            ])
        elif matcher_type == 'vocab_tree':
            vocab_tree_file = matcher_config.get('vocabulary_tree_file', '')
            if not vocab_tree_file:
                raise ValueError('vocabulary_tree_file must be specified for vocab_tree matcher')
            self._run_colmap_command([
                'vocab_tree_matcher',
                '--database_path', str(self.db_path),
                '--VocabTreeMatching.vocab_tree_path', vocab_tree_file,
            ])
        else:
            raise ValueError(f'Unknown matcher type: {matcher_type}')
    
    def run_bundle_adjustment(self) -> Dict:
        """
        マッパー（バンドルアジャストメント）を実行
        
        Returns
        -------
        dict
            COLMAP処理結果（カメラパラメータ、3Dポイントなど）
        """
        mapper_config = self.colmap_config.get('mapper', {})
        min_num_matches = mapper_config.get('min_num_matches', 12)
        
        self._run_colmap_command([
            'mapper',
            '--database_path', str(self.db_path),
            '--image_path', str(self.images_dir),
            '--output_path', str(self.sparse_dir),
            '--mapper.min_num_matches', str(min_num_matches),
            '--Mapper.num_threads', str(self.colmap_config.get('extractor', {}).get('num_threads', 8)),
            '--Mapper.min_focal_length_ratio', '0.1',
            '--Mapper.max_focal_length_ratio', '10',
            '--Mapper.max_num_iterations', str(mapper_config.get('max_num_iterations', 100)),
            '--Mapper.allowed_range_ratio', str(mapper_config.get('allowed_range_ratio', 0.05)),
        ])
        
        return self._parse_colmap_results()
    
    def _parse_colmap_results(self) -> Dict:
        """
        COLMAPの処理結果をパース
        
        Returns
        -------
        dict
            カメラパラメータ、3Dポイント雲などのデータ
        """
        cameras_json = self.sparse_dir / 'cameras' / 'cameras.bin'
        images_json = self.sparse_dir / 'images' / 'images.bin'
        points_json = self.sparse_dir / 'points3D' / 'points3D.bin'
        
        result = {
            'camera_params': [],
            'image_paths': [],
            'cameras': {},
            'images': {},
            'points3D': None
        }
        
        # cameras.binの読み込み
        if cameras_json.exists():
            result['cameras'] = self._read_cameras_binary(str(cameras_json))
        
        # images.binの読み込み（カメラ行列と位置を取得）
        if images_json.exists():
            result['images'] = self._read_images_binary(str(images_json))
        
        # points3D.binの読み込み
        if points_json.exists():
            result['points3D'] = self._read_points3d_binary(str(points_json))
        
        return result
    
    def _read_cameras_binary(self, filepath: str) -> Dict:
        """cameras.binを読み込み"""
        import struct
        
        cameras = {}
        with open(filepath, 'rb') as f:
            num_cameras = struct.unpack('Q', f.read(8))[0]
            
            for _ in range(num_cameras):
                camera_id = struct.unpack('Q', f.read(8))[0]
                camera_type = f.read(64).decode('utf-8').strip('\x00')
                
                if camera_type in ['SIMPLE_PINHOLE', 'PINHOLE', 'SIMPLE_RADIAL', 'RADIAL']:
                    params = list(struct.unpack('d' * 8, f.read(64)))
                    
                    if camera_type == 'SIMPLE_PINHOLE':
                        focal_length = params[0]
                        cx, cy = params[1], params[2]
                        width, height = params[3], params[4]
                        params_dict = {
                            'model': 'simple_pinhole',
                            'width': int(width),
                            'height': int(height),
                            'fx': focal_length,
                            'fy': focal_length,
                            'cx': cx,
                            'cy': cy
                        }
                    elif camera_type == 'PINHOLE':
                        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                        width, height = params[4], params[5]
                        params_dict = {
                            'model': 'pinhole',
                            'width': int(width),
                            'height': int(height),
                            'fx': fx,
                            'fy': fy,
                            'cx': cx,
                            'cy': cy
                        }
                    else:
                        params_dict = {
                            'model': camera_type.lower(),
                            'params': params
                        }
                    
                    cameras[camera_id] = params_dict
        
        return cameras
    
    def _read_images_binary(self, filepath: str) -> Dict:
        """images.binを読み込み（カメラ行列と位置）"""
        import struct
        
        images = {}
        with open(filepath, 'rb') as f:
            num_images = struct.unpack('Q', f.read(8))[0]
            
            for _ in range(num_images):
                image_id = struct.unpack('Q', f.read(8))[0]
                qtw, qzx, qzy, qzw = struct.unpack('dddd', f.read(32))
                qw, qx, qy, qz = qtw, qzx, qzy, qzw
                
                tx, ty, tz = struct.unpack('ddd', f.read(24))
                
                name = f.read(256).decode('utf-8').strip('\x00')
                
                # ロdrigues変換で回転行列を取得（簡易版）
                R = self._quaternion_to_rotation_matrix(qx, qy, qz, qw)
                t = np.array([tx, ty, tz])
                
                # 外部カメラ行列: R|t
                # 画像の姿勢は R^T * (-R^T * t) = -t 簡易的に R^T, -R^T*t を返す
                RT = R.T
                cam_R = RT
                cam_t = -RT @ t
                
                images[image_id] = {
                    'name': name,
                    'rotation': cam_R,
                    'translation': cam_t,
                    'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                    'tx': tx, 'ty': ty, 'tz': tz
                }
        
        return images
    
    def _quaternion_to_rotation_matrix(self, qx, qy, qz, qw):
        """クォータニオンから回転行列を生成"""
        return np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
    
    def _read_points3d_binary(self, filepath: str) -> np.ndarray:
        """points3D.binを読み込み"""
        import struct
        
        points3D = []
        with open(filepath, 'rb') as f:
            num_points = struct.unpack('Q', f.read(8))[0]
            
            for _ in range(num_points):
                point_id = struct.unpack('q', f.read(8))[0]
                x, y, z = struct.unpack('ddd', f.read(24))
                rgb = struct.unpack('BBB', f.read(3))
                error = struct.unpack('d', f.read(8))[0]
                
                points3D.append([x, y, z, rgb[0], rgb[1], rgb[2]])
        
        return np.array(points3D) if points3D else np.empty((0, 6))
    
    def run(self, processed_data: Dict) -> Dict:
        """
        COLMAPパイプラインを実行
        
        Parameters
        ----------
        processed_data : dict
            前処理の結果
        
        Returns
        -------
        dict
            COLMAP処理結果
        """
        print("画像を準備中...")
        self._prepare_images(processed_data)
        
        print("特徴量抽出中...")
        self.run_feature_extraction()
        
        print("特徴量マッチング中...")
        self.run_feature_matching()
        
        print("バンドルアジャストメント中...")
        results = self.run_bundle_adjustment()
        
        # 画像パスのマッピングを追加
        image_paths = processed_data['image_paths']
        results['image_paths'] = [str(self.images_dir / f'image_{i:04d}.jpg') for i in range(len(image_paths))]
        
        print(f"  推定されたカメラ数: {len(results['images'])}")
        if results['points3D'] is not None:
            print(f"  3Dポイント数: {len(results['points3D'])}")
        
        return results
