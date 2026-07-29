#!/usr/bin/env python3
"""
中間結果出力モジュール

各処理ステップの中間結果を保存し、可視化・確認できるようにする。
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# 画像処理（オプション）
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# trimesh（オプション）
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


class IntermediateOutputManager:
    """中間結果の保存を管理するクラス"""
    
    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('intermediate_output', {}).get('enabled', False)
        if not self.enabled:
            return
            
        self.base_dir = Path(config['intermediate_output']['dir'])
        self.gs_render_interval = config['intermediate_output'].get('gs_render_interval', 5000)
        
        # ディレクトリ構造の作成
        self._setup_directories()

    def _setup_directories(self):
        """出力ディレクトリの作成"""
        dirs = [
            self.base_dir / "preprocessing/enhanced_images",
            self.base_dir / "colmap",
            self.base_dir / "gsplat",
            self.base_dir / "mesh"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def save_preprocessing_results(self, processed_data: dict):
        """
        前処理結果の保存
        
        Parameters
        ----------
        processed_data : dict
            前処理の結果データ（image_paths, camera_params など）
        """
        if not self.enabled:
            return
            
        print("  [Intermediate] 前処理結果を保存中...")
        
        # カメラパラメータの保存
        cam_params_path = self.base_dir / "preprocessing/camera_params.json"
        camera_params = processed_data.get('camera_params', {})
        with open(cam_params_path, 'w', encoding='utf-8') as f:
            json.dump(camera_params, f, indent=4, ensure_ascii=False)
        print(f"    -> カメラパラメータ保存: {cam_params_path}")
        
        # 前処理済み画像のコピー
        image_paths = processed_data.get('image_paths', [])
        copied_count = 0
        for img_path in image_paths:
            src = Path(img_path)
            if src.exists():
                dst = self.base_dir / "preprocessing/enhanced_images" / src.name
                shutil.copy2(str(src), str(dst))
                copied_count += 1
        
        print(f"    -> 画像コピー完了: {copied_count}枚")

    def save_colmap_results(self, colmap_results: Optional[dict]):
        """
        COLMAP結果の保存
        
        Parameters
        ----------
        colmap_results : dict or None
            COLMAP処理結果（cameras, images, points3D など）
        """
        if not self.enabled or colmap_results is None:
            return
            
        print("  [Intermediate] COLMAP結果を保存中...")
        
        # カメラ位置情報の保存
        cam_pos_path = self.base_dir / "colmap/camera_positions.json"
        camera_info = {}
        
        images = colmap_results.get('images', {})
        for cam_id, cam_data in images.items():
            rotation = cam_data.get('rotation', None)
            translation = cam_data.get('translation', None)
            
            if rotation is not None and translation is not None:
                camera_info[str(cam_id)] = {
                    'rotation': rotation.tolist() if hasattr(rotation, 'tolist') else rotation,
                    'translation': translation.tolist() if hasattr(translation, 'tolist') else translation,
                    'name': cam_data.get('name', f'image_{cam_id}')
                }
        
        with open(cam_pos_path, 'w', encoding='utf-8') as f:
            json.dump(camera_info, f, indent=4, ensure_ascii=False)
        print(f"    -> カメラ位置情報保存: {cam_pos_path}")
        
        # ポイントクラウドの簡易可視化（trimeshが利用可能な場合）
        points3D = colmap_results.get('points3D', None)
        if points3D is not None and len(points3D) > 0 and TRIMESH_AVAILABLE:
            self._visualize_point_cloud(points3D)

    def _visualize_point_cloud(self, points3D: np.ndarray):
        """
        ポイントクラウドの可視化画像を生成
        
        Parameters
        ----------
        points3D : np.ndarray
            3Dポイントデータ (N, 6) - xyz + rgb
        """
        try:
            output_path = self.base_dir / "colmap/point_cloud.png"
            
            # trimesh でポイントクラウドを作成
            vertices = points3D[:, :3]
            colors = points3D[:, 3:6] if points3D.shape[1] >= 6 else None
            
            point_cloud = trimesh.PointCloud(vertices, colors=colors)
            
            # シーンを作成してレンダリング
            scene = trimesh.Scene([point_cloud])
            scene.save(str(output_path), file_type='png')
            
            print(f"    -> ポイントクラウド可視化保存: {output_path}")
        except Exception as e:
            print(f"    警告: ポイントクラウドの可視化に失敗しました: {e}")

    def save_gsplat_render(self, iteration: int, render_image: np.ndarray):
        """
        Gaussian Splatting 訓練中のレンダリング結果の保存
        
        Parameters
        ----------
        iteration : int
            現在のイテレーション数
        render_image : np.ndarray
            レンダリング画像 (H, W, 3) RGB形式
        """
        if not self.enabled:
            return
            
        # 指定された間隔で保存、または最初のイテレーション
        if iteration % self.gs_render_interval == 0 or iteration == 1:
            print(f"  [Intermediate] 訓練中レンダリング保存: iteration {iteration}")
            img_path = self.base_dir / f"gsplat/render_step_{iteration:06d}.png"
            
            # OpenCV が利用可能な場合は使用、そうでなければ PIL にフォールバック
            if CV2_AVAILABLE:
                bgr_image = cv2.cvtColor(render_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(img_path), bgr_image)
            else:
                from PIL import Image
                img = Image.fromarray(render_image.astype(np.uint8))
                img.save(str(img_path))
            
            print(f"    -> 保存完了: {img_path}")

    def save_mesh_results(self, mesh_data):
        """
        メッシュ化結果の保存
        
        Parameters
        ----------
        mesh_data : MeshData or dict
            メッシュデータ（vertices, faces など）
        """
        if not self.enabled:
            return
            
        print("  [Intermediate] メッシュ化結果を保存中...")
        
        # MeshData オブジェクトまたは辞書として処理
        if hasattr(mesh_data, 'vertices'):
            # MeshData オブジェクトの場合
            vertices = mesh_data.vertices
            faces = mesh_data.faces
            vertex_colors = mesh_data.vertex_colors if hasattr(mesh_data, 'vertex_colors') else None
            gaussian_params = None
        else:
            # 辞書の場合
            vertices = mesh_data.get('vertices', None)
            faces = mesh_data.get('faces', None)
            vertex_colors = mesh_data.get('vertex_colors', None)
            gaussian_params = mesh_data.get('gaussian_params', None)
        
        if vertices is not None and TRIMESH_AVAILABLE:
            try:
                mesh_path = self.base_dir / "mesh/mesh_intermediate.ply"
                
                # カラー情報があれば使用
                if vertex_colors is not None:
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
                else:
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                
                mesh.export(str(mesh_path))
                print(f"    -> メッシュ保存: {mesh_path}")
            except Exception as e:
                print(f"    警告: メッシュの保存に失敗しました: {e}")
        
        # ガウシアン分布データの保存（あれば）
        if gaussian_params is not None:
            try:
                npz_path = self.base_dir / "mesh/gaussian_distribution.npz"
                np.savez(str(npz_path), **gaussian_params)
                print(f"    -> ガウシアン分布データ保存: {npz_path}")
            except Exception as e:
                print(f"    警告: ガウシアン分布データの保存に失敗しました: {e}")
