#!/usr/bin/env python3
"""
出力エクスポートモジュール

メッシュデータを.glbおよび.obj形式で出力する。
"""

import os
import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from mesh_converter import MeshData


class ModelExporter:
    """モデルエクスポートクラス"""
    
    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            設定ファイル
        """
        self.config = config
        self.output_config = config.get('output', {})
        
        # 出力ディレクトリ
        self.output_dir = Path(self.output_config['dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル名プレフィックス
        self.filename_prefix = self.output_config.get('filename_prefix', 'car_model')
        
        # 出力形式
        self.formats = self.output_config.get('formats', ['glb', 'obj'])
    
    def export(self, mesh_data: MeshData) -> List[str]:
        """
        メッシュデータを指定形式で出力
        
        Parameters
        ----------
        mesh_data : MeshData
            メッシュデータ
        
        Returns
        -------
        list[str]
            出力されたファイルパスのリスト
        """
        print(f"出力形式: {self.formats}")
        
        exported_files = []
        
        if 'glb' in self.formats:
            glb_path = self._export_glb(mesh_data)
            if glb_path:
                exported_files.append(str(glb_path))
        
        if 'obj' in self.formats:
            obj_path = self._export_obj(mesh_data)
            if obj_path:
                exported_files.append(str(obj_path))
        
        return exported_files
    
    def _export_glb(self, mesh_data: MeshData) -> Optional[Path]:
        """
        GLB形式で出力
        
        Parameters
        ----------
        mesh_data : MeshData
            メッシュデータ
        
        Returns
        -------
        Path or None
            出力ファイルパス
        """
        try:
            # trimeshオブジェクトを作成
            mesh = trimesh.Trimesh(
                vertices=mesh_data.vertices,
                faces=mesh_data.faces,
                vertex_colors=mesh_data.vertex_colors
            )
            
            # GLBファイルパス
            glb_path = self.output_dir / f"{self.filename_prefix}.glb"
            
            # GLB形式で保存
            mesh.export(str(glb_path))
            
            file_size = os.path.getsize(glb_path)
            print(f"  GLB出力: {glb_path} ({self._format_file_size(file_size)})")
            
            return glb_path
            
        except Exception as e:
            print(f"エラー: GLB出力に失敗しました ({e})")
            return None
    
    def _export_obj(self, mesh_data: MeshData) -> Optional[Path]:
        """
        OBJ形式で出力
        
        Parameters
        ----------
        mesh_data : MeshData
            メッシュデータ
        
        Returns
        -------
        Path or None
            出力ファイルパス
        """
        try:
            # trimeshオブジェクトを作成
            mesh = trimesh.Trimesh(
                vertices=mesh_data.vertices,
                faces=mesh_data.faces,
                vertex_colors=mesh_data.vertex_colors
            )
            
            # OBJファイルパス
            obj_path = self.output_dir / f"{self.filename_prefix}.obj"
            
            # OBJ形式で保存
            mesh.export(str(obj_path))
            
            file_size = os.path.getsize(obj_path)
            print(f"  OBJ出力: {obj_path} ({self._format_file_size(file_size)})")
            
            return obj_path
            
        except Exception as e:
            print(f"エラー: OBJ出力に失敗しました ({e})")
            return None
    
    def _format_file_size(self, size_bytes: int) -> str:
        """ファイルサイズを人間 readable な形式に変換"""
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
