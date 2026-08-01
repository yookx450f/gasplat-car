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
        
        # 出力ディレクトリ（後で絶対パスに上書き）
        self.work_dir = None
        self.db_path = None
        self.images_dir = None
        self.sparse_dir = None
    
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
            cwd=str(os.path.abspath(self.work_dir)),
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
        max_num_features = extractor_config.get('max_num_features', 4096)
        sift_peak_threshold = extractor_config.get('sift_peak_threshold', 0.0066666666666666671)
        sift_edge_threshold = extractor_config.get('sift_edge_threshold', 10)
        
        # SIMPLE_RADIAL: focal_length, cx, cy, k
        # 画像サイズに合わせた適切なカメラパラメータを設定
        # 焦点距離は画像の最大辺の長さの1.5倍程度（一般的なレンズ相当）
        focal = float(max_size) * 1.5  # 焦点距離
        cx = float(max_size) / 2.0     # 主点X
        cy = float(max_size) / 2.0     # 主点Y
        k = 0.0                         # 歪み係数（後で最適化）
        
        camera_params_str = f'{focal},{cx},{cy},{k}'
        
        self._run_colmap_command([
            'feature_extractor',
            '--database_path', str(self.db_path),
            '--image_path', str(self.images_dir),
            '--ImageReader.camera_model', camera_model,
            '--ImageReader.single_camera', '1',
            '--ImageReader.camera_params', camera_params_str,
            '--ImageReader.default_focal_length_factor', '1.2',
            '--SiftExtraction.max_num_features', str(max_num_features),
            '--SiftExtraction.peak_threshold', str(sift_peak_threshold),
            '--SiftExtraction.edge_threshold', str(sift_edge_threshold),
            '--SiftExtraction.use_gpu', '0'  # CPUモード（コンテナ内OpenGL問題回避）
        ])
    
    def run_feature_matching(self) -> None:
        """特徴量マッチングを実行"""
        matcher_config = self.colmap_config.get('matcher', {})
        matcher_type = matcher_config.get('type', 'exhaustive')
        max_num_matches = matcher_config.get('max_num_matches', 65536)
        mutual_filter = matcher_config.get('mutual_filter', 1)
        
        if matcher_type == 'exhaustive':
            self._run_colmap_command([
                'exhaustive_matcher',
                '--database_path', str(self.db_path),
                '--SiftMatching.max_num_matches', str(max_num_matches),
                '--SiftMatching.use_gpu', '0'  # CPUモード（コンテナ内OpenGL問題回避）
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
        min_num_matches = mapper_config.get('min_num_matches', 3)  # 低い値で初期ペアを見つけやすくする
        max_num_iterations = mapper_config.get('max_num_iterations', 50)
        init_min_tri_angle = mapper_config.get('init_min_tri_angle', 0.1)  # 三角測量の最小角度を緩和
        ba_global_max_num_iterations = mapper_config.get('ba_global_max_num_iterations', 200)
        ba_global_max_refit_iterations = mapper_config.get('ba_global_max_refit_iterations', 10)
        
        # 出力ディレクトリを事前に作成（COLMAP要求）
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        
        # COLMAP mapperを実行（エラー時も警告として処理を続行）
        # COLMAP 3.6 のデフォルト値を調整して、少ない画像数でも動作するようにする
        cmd = [
            'mapper',
            '--database_path', str(self.db_path),
            '--image_path', str(self.images_dir),
            '--output_path', str(self.sparse_dir),
            '--Mapper.min_num_matches', '1',  # マッチ数が少ない場合でも動作するように
            '--Mapper.num_threads', str(self.colmap_config.get('extractor', {}).get('num_threads', 8)),
            '--Mapper.min_focal_length_ratio', '0.01',
            '--Mapper.max_focal_length_ratio', '100',
            '--Mapper.ba_global_max_num_iterations', str(ba_global_max_num_iterations),
            '--Mapper.init_max_error', '5.0',  # 許容誤差を緩和
            '--Mapper.init_min_num_inliers', '5',  # 最小インライア数を下げる
            '--Mapper.init_max_forward_motion', '0.999',
            '--Mapper.init_min_tri_angle', '1.0',
            '--Mapper.init_max_reg_trials', '10',  # 登録試行回数を増やす
            '--Mapper.ba_global_use_pba', '0',  # PBAを使わない（CUDAなし対応）
            '--Mapper.multiple_models', '0',
            '--Mapper.ignore_watermarks', '1',
            '--Mapper.min_model_size', '2',  # 最小モデルサイズを下げる
            '--Mapper.max_num_models', '100',
        ]
        
        print(f"  実行: {' '.join(cmd)}")
        result = subprocess.run(
            ['colmap'] + cmd,
            cwd=str(os.path.abspath(self.work_dir)),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # mapperが失敗しても警告として処理を続行
            print(f"  警告: COLMAP mapperがエラーを返しました: {result.stderr[:500]}")
            print(f"  警告: 初期画像ペアが見つからなかった可能性があります。")
            print(f"  警告: 処理を続行します...")
            
            # 既に作成されたsparseディレクトリが存在するか確認（COLMAP 3.7+対応）
            found_images = False
            for images_path in [self.sparse_dir / '0' / 'images.bin', self.sparse_dir / '0' / 'images' / 'images.bin', self.sparse_dir / 'images' / 'images.bin']:
                if images_path.exists():
                    found_images = True
                    break
            
            if not self.sparse_dir.exists() or not found_images:
                # 空の結果を返す
                print(f"  警告: マップ結果が存在しないため、空の結果を返します。")
                return {
                    'camera_params': [],
                    'image_paths': [],
                    'cameras': {},
                    'images': {},
                    'points3D': None
                }
        
        return self._parse_colmap_results()
    
    def _ensure_text_format(self) -> Path:
        """sparse/0/のバイナリをsparse/text/にテキスト変換して返す"""
        text_dir = self.sparse_dir / 'text'
        if not text_dir.exists() or not any(text_dir.iterdir()):
            text_dir.mkdir(parents=True, exist_ok=True)
            
            # COLMAP 3.7+ではバイナリファイルが存在するか確認
            binary_cameras = self.sparse_dir / '0' / 'cameras.bin'
            binary_images = self.sparse_dir / '0' / 'images.bin'
            binary_points = self.sparse_dir / '0' / 'points3D.bin'
            
            if binary_cameras.exists() and binary_images.exists() and binary_points.exists():
                # バイナリからテキストに変換
                self._run_colmap_command([
                    'model_converter',
                    '--input_path', str(self.sparse_dir / '0'),
                    '--output_path', str(text_dir),
                    '--output_type', 'txt'
                ])
            else:
                # 既にtext形式が存在する可能性があるので、sparse/0/text/も確認
                alt_text_dir = self.sparse_dir / '0' / 'text'
                if alt_text_dir.exists() and any(alt_text_dir.iterdir()):
                    # 既存のtextファイルを現在のtextディレクトリにコピー
                    for file in alt_text_dir.glob('*'):
                        shutil.copy2(str(file), str(text_dir / file.name))
                else:
                    # text形式が存在しない場合はバイナリファイルから直接読み込む
                    print("  警告: テキスト形式の変換ができません。バイナリ形式で読み込みます。")
                    return None
        
        return text_dir
    
    def _parse_colmap_results(self) -> Dict:
        """
        COLMAPの処理結果をパース（テキスト形式を使用、バイナリにフォールバック）
        
        Returns
        -------
        dict
            カメラパラメータ、3Dポイント雲などのデータ
        """
        text_dir = self._ensure_text_format()
        
        result = {
            'camera_params': [],
            'image_paths': [],
            'cameras': {},
            'images': {},
            'points3D': None
        }
        
        if text_dir is not None:
            # テキスト形式で読み込み
            cameras_txt = text_dir / 'cameras.txt'
            images_txt = text_dir / 'images.txt'
            points3D_txt = text_dir / 'points3D.txt'
            
            if cameras_txt.exists():
                result['cameras'] = self._read_cameras_text(str(cameras_txt))
            if images_txt.exists():
                result['images'] = self._read_images_text(str(images_txt))
            if points3D_txt.exists():
                result['points3D'] = self._read_points3d_text(str(points3D_txt))
        else:
            # バイナリ形式にフォールバック
            print("  警告: テキスト形式が利用できないため、バイナリ形式で読み込みます。")
            
            # COLMAP 3.7+のバイナリパス
            binary_cameras = self.sparse_dir / '0' / 'cameras.bin'
            binary_images = self.sparse_dir / '0' / 'images.bin'
            binary_points = self.sparse_dir / '0' / 'points3D.bin'
            
            # 古いバイナリパスも確認
            if not binary_cameras.exists():
                binary_cameras = self.sparse_dir / 'cameras.bin'
            if not binary_images.exists():
                binary_images = self.sparse_dir / 'images.bin'
            if not binary_points.exists():
                binary_points = self.sparse_dir / 'points3D.bin'
            
            if binary_cameras.exists():
                result['cameras'] = self._read_cameras_binary(str(binary_cameras))
            if binary_images.exists():
                result['images'] = self._read_images_binary(str(binary_images))
            if binary_points.exists():
                result['points3D'] = self._read_points3d_binary(str(binary_points))
        
        return result
    
    def _read_cameras_text(self, filepath: str) -> Dict:
        """cameras.txtを読み込み"""
        cameras = {}
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # コメント行または空行をスキップ
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # カメラデータのパース
            parts = line.split()
            if len(parts) >= 7:
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                
                # パラメータ数モデルによって異なる
                if model == 'SIMPLE_PINHOLE':
                    focal_length = float(parts[4])
                    cx, cy = float(parts[5]), float(parts[6])
                    params_dict = {
                        'model': 'simple_pinhole',
                        'width': width,
                        'height': height,
                        'fx': focal_length,
                        'fy': focal_length,
                        'cx': cx,
                        'cy': cy
                    }
                elif model == 'PINHOLE':
                    fx, fy, cx, cy = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    params_dict = {
                        'model': 'pinhole',
                        'width': width,
                        'height': height,
                        'fx': fx,
                        'fy': fy,
                        'cx': cx,
                        'cy': cy
                    }
                elif model == 'SIMPLE_RADIAL':
                    focal_length = float(parts[4])
                    cx, cy = float(parts[5]), float(parts[6])
                    k = float(parts[7])
                    params_dict = {
                        'model': 'simple_radial',
                        'width': width,
                        'height': height,
                        'fx': focal_length,
                        'fy': focal_length,
                        'cx': cx,
                        'cy': cy,
                        'k': k
                    }
                elif model == 'RADIAL':
                    focal_length = float(parts[4])
                    cx, cy = float(parts[5]), float(parts[6])
                    k1, k2 = float(parts[7]), float(parts[8])
                    params_dict = {
                        'model': 'radial',
                        'width': width,
                        'height': height,
                        'fx': focal_length,
                        'fy': focal_length,
                        'cx': cx,
                        'cy': cy,
                        'k1': k1,
                        'k2': k2
                    }
                elif model == 'OPENCV':
                    fx, fy, cx, cy = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    k1, k2, p1, p2 = float(parts[8]), float(parts[9]), float(parts[10]), float(parts[11])
                    params_dict = {
                        'model': 'opencv',
                        'width': width,
                        'height': height,
                        'fx': fx,
                        'fy': fy,
                        'cx': cx,
                        'cy': cy,
                        'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2
                    }
                elif model == 'OPENCV_FISHEYE':
                    fx, fy, cx, cy = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    k1, k2, k3, k4 = float(parts[8]), float(parts[9]), float(parts[10]), float(parts[11])
                    params_dict = {
                        'model': 'opencv_fisheye',
                        'width': width,
                        'height': height,
                        'fx': fx,
                        'fy': fy,
                        'cx': cx,
                        'cy': cy,
                        'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4
                    }
                else:
                    # その他のモデルはパラメータとして保存
                    num_params = len(parts) - 4
                    params = [float(p) for p in parts[4:]]
                    params_dict = {
                        'model': model.lower(),
                        'width': width,
                        'height': height,
                        'params': params
                    }
                
                cameras[camera_id] = params_dict
            
            i += 1
        
        return cameras
    
    def _read_images_text(self, filepath: str) -> Dict:
        """images.txtを読み込み（カメラ行列と位置）"""
        images = {}
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # コメント行または空行をスキップ
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # イメージデータのパース
            # COLMAP 3.7+ フォーマット: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            # または 古いフォーマット: qiw qx qy qz (行1), tx ty tz (行2), image_name (行3), camera_id (行4)
            parts = line.split()
            
            if len(parts) >= 10:
                # COLMAP 3.7+ フォーマット: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                image_id = int(parts[0])
                qw = float(parts[1])
                qx = float(parts[2])
                qy = float(parts[3])
                qz = float(parts[4])
                tx = float(parts[5])
                ty = float(parts[6])
                tz = float(parts[7])
                camera_id = int(parts[8])
                image_name = ' '.join(parts[9:])  # ファイル名にスペースが含まれる可能性
                
                # 次の行はPOINTS2Dデータなのでスキップ
                i += 2
            elif len(parts) == 4:
                # 古いフォーマット: クォータニオン形式 (qiw qx qy qz)
                qw = float(parts[0])
                qx = float(parts[1])
                qy = float(parts[2])
                qz = float(parts[3])
                
                # 次の行からtとimage_nameを読み込む
                i += 1
                if i >= len(lines):
                    break
                t_parts = lines[i].strip().split()
                tx, ty, tz = float(t_parts[0]), float(t_parts[1]), float(t_parts[2])
                
                i += 1
                if i >= len(lines):
                    break
                image_name = lines[i].strip()
                
                i += 1
                if i >= len(lines):
                    break
                camera_id = int(lines[i].strip())
                i += 1
            else:
                # ロdriguesベクトル形式（簡易処理）
                i += 1
                if i >= len(lines):
                    break
                t_parts = lines[i].strip().split()
                tx, ty, tz = float(t_parts[0]), float(t_parts[1]), float(t_parts[2])
                
                i += 1
                if i >= len(lines):
                    break
                image_name = lines[i].strip()
                
                i += 1
                if i >= len(lines):
                    break
                camera_id = int(lines[i].strip())
                i += 1
                
                # ロdriguesベクトルからクォータニオンに変換（簡易版）
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            
            # クォータニオンから回転行列を生成
            R = self._quaternion_to_rotation_matrix(qx, qy, qz, qw)
            t = np.array([tx, ty, tz])
            
            # カメラ座標系に変換: R^T, -R^T*t
            RT = R.T
            cam_R = RT
            cam_t = -RT @ t
            
            images[camera_id] = {
                'name': image_name,
                'rotation': cam_R,
                'translation': cam_t,
                'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                'tx': tx, 'ty': ty, 'tz': tz
            }
        
        return images
    
    def _read_points3d_text(self, filepath: str) -> np.ndarray:
        """points3D.txtを読み込み"""
        points3D = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            # コメント行または空行をスキップ
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 14:
                point_id = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                error = float(parts[7])
                
                # 観測された画像のリスト（残りは省略）
                points3D.append([x, y, z, r, g, b])
        
        return np.array(points3D) if points3D else np.empty((0, 6))
    
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
        # 作業ディレクトリを絶対パスで設定（COLMAP要求）
        self.work_dir = Path(os.path.abspath('data/colmap_work'))
        self.db_path = self.work_dir / 'database.db'
        self.images_dir = self.work_dir / 'images'
        self.sparse_dir = self.work_dir / 'sparse'
        
        # 既存のデータベースを削除（再実行対応）
        if self.db_path.exists():
            self.db_path.unlink()
        
        # 既存の作業ディレクトリを完全に削除
        import shutil
        if self.work_dir.exists():
            shutil.rmtree(str(self.work_dir))
        
        # 作業ディレクトリを事前に作成（COLMAP要求）
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        
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
