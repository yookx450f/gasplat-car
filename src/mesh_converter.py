#!/usr/bin/env python3
"""
メッシュ化モジュール

Gaussian Splattingの結果から3Dメッシュ（頂点、法線、テクスチャ）を生成する。
"""

import os
import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, Optional
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
    
    def _extract_mesh_from_gaussians(self, gaussian_data: Dict) -> trimesh.Trimesh:
        """
        ガウシアンからメッシュを抽出
        
        Poisson ReconstructionまたはDense Point Cloudからのメッシュ化を使用。
        
        Parameters
        ----------
        gaussian_data : dict
            Gaussian Splattingの結果
        
        Returns
        -------
        trimesh.Trimesh
            抽出されたメッシュ
        """
        means = gaussian_data['means']
        colors = gaussian_data['colors']
        
        # ガウシアン中心点をポイントクラウドとして使用
        points = means
        
        # 色を付加（RGB: 0-255）
        point_colors = (colors * 255).astype(np.uint8)
        
        print(f"ポイントクラウド: {len(points)} points")
        
        # trimeshを使用してメッシュ化
        try:
            # ポイントクラウドからメッシュを生成
            # Poisson Reconstructionを使用
            mesh = trimesh.creation.point_cloud(points)
            
            # 法線を推定
            mesh.estimate_normals()
            
            # ポアソン再構成
            depth = self.meshing_config.get('poisson_depth', 9)
            mesh, faces = trimesh.reconstruction.polygon_mesh(
                points,
                vertex_colors=point_colors,
                depth=depth
            )
            
        except Exception as e:
            print(f"警告: ポアソン再構成に失敗しました ({e})。簡易メッシュを使用します。")
            # 簡易的なメッシュ生成（Convex Hull）
            mesh = trimesh.convex(points)
            point_colors = np.tile([128, 128, 128], (len(mesh.vertices), 1))
        
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
    
    def convert(self, gs_result) -> MeshData:
        """
        Gaussian Splatting結果からメッシュを生成
        
        Parameters
        ----------
        gs_result : GaussianData
            Gaussian Splattingの訓練結果
        
        Returns
        -------
        MeshData
            生成されたメッシュデータ
        """
        print("メッシュを抽出中...")
        
        # Gaussianデータを読み込む
        gaussian_data = self._load_gaussian_data(gs_result)
        
        # メッシュを抽出
        mesh = self._extract_mesh_from_gaussians(gaussian_data)
        
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
