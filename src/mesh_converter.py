#!/usr/bin/env python3
"""
メッシュ化モジュール

Gaussian Splattingの結果から3Dメッシュ（頂点、法線、テクスチャ）を生成する。
"""

import os
import subprocess
import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
from PIL import Image


@dataclass
class MeshData:
    """メッシュデータ"""
    vertices: np.ndarray = None       # 頂点座標 (V, 3)
    faces: np.ndarray = None          # 面定義 (F, 3)
    vertex_colors: np.ndarray = None  # 頂点色 (V, 3)
    uvs: np.ndarray = None            # UV座標 (V, 2)
    image_size: tuple = None          # テクスチャ画像サイズ


class MeshConverter:
    """メッシュ変換クラス"""
    
    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            設定ファイル
        """
        self.config = config
        self.meshing_config = config.get('meshing', {})
        
        # 出力ディレクトリ
        self.output_dir = Path(config['output']['dir']) / 'mesh'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # COLMAP作業ディレクトリ
        self.colmap_work_dir = Path(config['data']['colmap_work_dir'])
    
    def _load_gaussian_data(self, gs_result) -> Dict:
        """Gaussian Splattingの結果を読み込む"""
        gaussian_data = {
            'means': gs_result.means,
            'quats': gs_result.quats,
            'scales': gs_result.scales,
            'opacities': gs_result.opacities,
            'colors': gs_result.colors
        }
        
        return gaussian_data
    
    def _filter_gaussians_by_opacity(self, gaussian_data: Dict,
                                       opacity_threshold: float = 0.01) -> Dict:
        """
        不透明度が高いガウシアンだけをフィルタリング
        
        Parameters
        ----------
        gaussian_data : dict
            Gaussian Splattingの結果
        opacity_threshold : float
            不透明度の閾値
        
        Returns
        -------
        dict
            フィルタリングされたガウシアンデータ
        """
        means = gaussian_data['means']
        colors = gaussian_data['colors']
        opacities = gaussian_data['opacities']
        
        # 不透明度でフィルタリング
        # opacitiesはlog値なので、sigmoidで逆変換
        sigmoid_opacities = 1.0 / (1.0 + np.exp(-opacities))
        
        mask = sigmoid_opacities > opacity_threshold
        filtered_means = means[mask]
        filtered_colors = colors[mask]
        
        print(f"  不透明度フィルタ適用: {len(means)} -> {len(filtered_means)} points "
              f"(閾値: {opacity_threshold})")
        
        return {
            'means': filtered_means,
            'colors': filtered_colors
        }
    
    def _generate_fallback_mesh(self, gaussian_data: Dict) -> trimesh.Trimesh:
        """
        ガウシアンデータからフォールバックメッシュを生成
        
        COLMAPや凸包が機能しない場合、ガウシアン中心点の分布から
        車の形状を推測して、ボックスまたは球体を生成する。
        
        Parameters
        ----------
        gaussian_data : dict
            Gaussian Splattingの結果
        
        Returns
        -------
        trimesh.Trimesh
            生成されたフォールバックメッシュ
        """
        means = gaussian_data['means']
        colors = gaussian_data['colors']
        
        if len(means) == 0:
            print("  警告: ガウシアンデータが空です。デフォルトのボックスを生成します。")
            return self._create_default_box()
        
        print(f"  ガウシアン中心点からフォールバックメッシュを生成: {len(means)} points")
        
        # 中心点の統計量を計算
        min_coords = np.min(means, axis=0)
        max_coords = np.max(means, axis=0)
        center = np.mean(means, axis=0)
        
        # 色の平均を計算
        avg_color = np.mean(colors, axis=0)
        
        # 軸方向の大きさを計算
        size = max_coords - min_coords
        max_size = np.max(size)
        
        # 車の形状を推測してボックスを生成
        # 車は通常、横長・低めの形状をしている
        width, height, depth = size
        
        # 色の付いたボックスを生成
        box_mesh = self._create_colored_box(center, size, avg_color)
        
        print(f"  フォールバックボックス生成: 寸法={size}, 中心={center}")
        
        return box_mesh
    
    def _create_default_box(self) -> trimesh.Trimesh:
        """
        デフォルトのボックスを生成
        
        Returns
        -------
        trimesh.Trimesh
            デフォルトのボックスメッシュ
        """
        # 単位ボックスを生成
        box = trimesh.creation.box(extents=[1.0, 0.5, 0.8])
        
        # 車らしい色（グレー）を付加
        vertex_colors = np.tile([128, 128, 128, 255], (len(box.vertices), 1))
        box.visual = trimesh.visual.ColorVisuals(box)
        box.visual.vertex_colors = vertex_colors
        
        print("  デフォルトボックスを生成しました")
        
        return box
    
    def _create_colored_box(self, center: np.ndarray, size: np.ndarray,
                            color: np.ndarray) -> trimesh.Trimesh:
        """
        色付きのボックスを生成
        
        Parameters
        ----------
        center : np.ndarray
            ボックスの中心座標 (3,)
        size : np.ndarray
            ボックスの寸法 (3,)
        color : np.ndarray
            頂点の色 (3,)
        
        Returns
        -------
        trimesh.Trimesh
            色付きのボックスメッシュ
        """
        # ボックスを生成
        box = trimesh.creation.box(extents=size)
        
        # 中心を移動
        box.vertices += center - size / 2
        
        # 色を付加
        vertex_colors = np.tile(color.reshape(1, 3).astype(np.uint8) * 255,
                                (len(box.vertices), 1))
        vertex_colors = np.concatenate([vertex_colors, np.full((len(box.vertices), 1), 255, dtype=np.uint8)], axis=1)
        box.visual = trimesh.visual.ColorVisuals(box)
        box.visual.vertex_colors = vertex_colors
        
        return box
    
    def _try_colmap_dense_reconstruction(self, colmap_results: Dict) -> Optional[trimesh.Trimesh]:
        """
        COLMAPのdense reconstructionを試みる
        
        Parameters
        ----------
        colmap_results : dict
            COLMAP処理結果
        
        Returns
        -------
        trimesh.Trimesh or None
            生成されたメッシュ
        """
        print("  COLMAP dense reconstructionを試みる...")
        
        try:
            # COLMAPのステレオエンジンを使用して密なポイントクラウドを生成
            image_dir = self.colmap_work_dir / 'images'
            sparse_dir = self.colmap_work_dir / 'sparse' / '0'
            
            if not image_dir.exists() or not sparse_dir.exists():
                print("  警告: COLMAPデータが見つかりません。")
                return None
            
            # ステレオエンジンを実行
            stereo_cmd = [
                'colmap', 'stereo_matching',
                '--database_path', str(self.colmap_work_dir / 'database.db'),
                '--image_path', str(image_dir),
                '--input_path', str(image_dir),
                '--Output.path', str(self.output_dir / 'colmap_dense')
            ]
            
            result = subprocess.run(stereo_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("  COLMAP dense reconstruction成功")
                # 密なポイントクラウドを読み込む
                dense_point_cloud = self.output_dir / 'colmap_dense' / 'reconstruction' / 'points3D.xyz'
                if dense_point_cloud.exists():
                    return self._load_dense_point_cloud(dense_point_cloud)
            else:
                print(f"  COLMAP dense reconstruction失敗: {result.stderr[:200]}")
                
        except subprocess.TimeoutExpired:
            print("  COLMAP dense reconstructionタイムアウト")
        except Exception as e:
            print(f"  COLMAP dense reconstructionエラー: {e}")
        
        return None
    
    def _load_dense_point_cloud(self, points_file: Path) -> Optional[trimesh.Trimesh]:
        """
        密なポイントクラウドを読み込んでメッシュを生成
        
        Parameters
        ----------
        points_file : Path
            ポイントクラウドファイルパス
        
        Returns
        -------
        trimesh.Trimesh or None
            生成されたメッシュ
        """
        try:
            # XYZファイルを読み込む
            points = np.loadtxt(str(points_file))
            
            if len(points) == 0:
                return None
            
            print(f"  密なポイントクラウド: {len(points)} points")
            
            # 凸包ではなく、Poisson Reconstructionを試みる
            return self._create_mesh_from_point_cloud(points)
            
        except Exception as e:
            print(f"  ポイントクラウド読み込み失敗: {e}")
            return None
    
    def _create_mesh_from_point_cloud(self, points: np.ndarray) -> Optional[trimesh.Trimesh]:
        """
        ポイントクラウドからメッシュを生成
        
        複数のアプローチを試みる:
        1. Poisson Reconstruction（試行）
        2. 凸包（Convex Hull）- 最終フォールバック
        
        Parameters
        ----------
        points : np.ndarray
            ポイントクラウド (N, 3)
        
        Returns
        -------
        trimesh.Trimesh or None
            生成されたメッシュ
        """
        mesh = None
        
        # アプローチ1: Poisson Reconstructionを試みる
        print("  アプローチ1: Poisson Reconstructionを試みる...")
        try:
            import pyembree
            from pyscg import ScreenedPoissonReconstruction
            
            # Screened Poisson Reconstructionを実行
            mesh_data = ScreenedPoissonReconstruction(
                points,
                depth=8,  # 解像度
                point_weight=1.0,
                sample_extent=0.05
            )
            
            vertices, faces = mesh_data
            
            if len(vertices) > 0 and len(faces) > 0:
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                print(f"  Poisson Reconstruction成功: {len(vertices)} 頂点, {len(faces)} 面")
                return mesh
            
        except ImportError:
            print("  警告: pyembree/pyscgがインストールされていません。")
        except Exception as e:
            print(f"  Poisson Reconstruction失敗: {e}")
        
        # アプローチ2: 凸包（Convex Hull）- 最終フォールバック
        print("  凸包（Convex Hull）を生成する...")
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            
            mesh = trimesh.Trimesh(
                vertices=points,
                faces=hull.simplices
            )
            
            print(f"  凸包生成: {len(mesh.vertices)} 頂点, {len(mesh.faces)} 面")
            
        except Exception as e:
            print(f"  凸包生成失敗: {e}")
            mesh = trimesh.Trimesh()
        
        return mesh
    
    def _extract_mesh_from_gaussians(self, gaussian_data: Dict,
                                      colmap_results: Dict = None) -> trimesh.Trimesh:
        """
        ガウシアンからメッシュを抽出
        
        複数のアプローチを試みる:
        1. COLMAP dense reconstruction（優先）
        2. ガウシアン中心点からの凸包生成
        3. ガウシアン中心点からフォールバックボックスを生成
        
        Parameters
        ----------
        gaussian_data : dict
            Gaussian Splattingの結果
        colmap_results : dict
            COLMAP処理結果（オプション）
        
        Returns
        -------
        trimesh.Trimesh
            抽出されたメッシュ
        """
        means = gaussian_data['means']
        colors = gaussian_data['colors']
        
        # 不透明度でフィルタリング（閾値を低下）
        filtered_data = self._filter_gaussians_by_opacity(gaussian_data, opacity_threshold=0.01)
        points = filtered_data['means']
        point_colors = (filtered_data['colors'] * 255).astype(np.uint8)
        
        print(f"ポイントクラウド: {len(points)} points")
        
        mesh = None
        
        # アプローチ1: COLMAP dense reconstructionを試みる
        if colmap_results is not None:
            mesh = self._try_colmap_dense_reconstruction(colmap_results)
        
        # アプローチ2: ガウシアン中心点からメッシュ化
        if mesh is None or len(mesh.faces) < 1000:
            print("  ガウシアン中心点からメッシュ化を試みる...")
            try:
                # ポイントクラウドがある場合は凸包を生成
                if len(points) > 4:
                    mesh = self._create_mesh_from_point_cloud(points)
                    
                    # 色を付加
                    if mesh is not None and len(mesh.vertices) > 0 and len(point_colors) >= len(mesh.vertices):
                        mesh.visual = trimesh.visual.ColorVisuals(mesh)
                        mesh.visual.vertex_colors = point_colors[:len(mesh.vertices)]
                else:
                    print("  ポイント数が不足しているため、フォールバックボックスを生成します。")
            except Exception as e:
                print(f"  ガウシアン中心点からのメッシュ化失敗: {e}")
        
        # 最終フォールバック: ガウシアンデータからボックスを生成
        if mesh is None or len(mesh.faces) < 10:
            print("  最終フォールバック: ガウシアンからボックスを生成...")
            try:
                mesh = self._generate_fallback_mesh(gaussian_data)
            except Exception as e:
                print(f"  フォールバックボックス生成失敗: {e}")
                mesh = self._create_default_box()
        
        return mesh
    
    def _extract_textures(self, gaussian_data: Dict, mesh: trimesh.Trimesh, 
                          camera_params: list) -> Optional[Image.Image]:
        """
        メッシュからテクスチャを抽出
        
        Parameters
        ----------
        gaussian_data : dict
            Gaussian Splattingの結果
        mesh : trimesh.Trimesh
            メッシュデータ
        camera_params : list
            カメラパラメータ
        
        Returns
        -------
        PIL.Image.Image or None
            抽出されたテクスチャ画像
        """
        # 複数のカメラ視点から色を統合
        # 簡易版：平均的な色を使用
        
        # メッシュの頂点色を直接使用
        if hasattr(mesh, 'vertex_colors') and mesh.vertex_colors is not None:
            return None  # 頂点色があるのでテクスラは不要
        
        return None
    
    def convert(self, gs_result, colmap_results: Dict = None) -> MeshData:
        """
        Gaussian Splatting結果からメッシュを生成
        
        Parameters
        ----------
        gs_result : GaussianData
            Gaussian Splattingの訓練結果
        colmap_results : dict
            COLMAP処理結果（オプション）
        
        Returns
        -------
        MeshData
            生成されたメッシュデータ
        """
        print("メッシュを抽出中...")
        
        # Gaussianデータを読み込む
        gaussian_data = self._load_gaussian_data(gs_result)
        
        # メッシュを抽出
        mesh = self._extract_mesh_from_gaussians(gaussian_data, colmap_results)
        
        # MeshDataを作成
        mesh_data = MeshData(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_colors=getattr(mesh, 'vertex_colors', None),
            image_size=None
        )
        
        # メッシュを保存
        self._save_mesh(mesh_data)
        
        print(f"メッシュ抽出完了: {len(mesh.vertices)} 頂点, {len(mesh.faces)} 面")
        
        return mesh_data
    
    def _save_mesh(self, mesh_data: MeshData):
        """メッシュを保存"""
        # trimeshオブジェクトを作成
        mesh = trimesh.Trimesh(
            vertices=mesh_data.vertices,
            faces=mesh_data.faces,
            vertex_colors=mesh_data.vertex_colors
        )
        
        # 保存
        mesh.export(str(self.output_dir / 'mesh.ply'))
        print(f"  メッシュ保存: {self.output_dir / 'mesh.ply'}")
