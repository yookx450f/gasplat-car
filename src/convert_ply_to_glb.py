#!/usr/bin/env python3
"""
PLYファイルからGLBファイルを生成するスクリプト

output/intermediate/mesh/mesh_intermediate.ply から
output/glb/car_model.glb を生成する。
"""

import sys
import os
import numpy as np
from pathlib import Path

# trimeshをインポート
import trimesh


def convert_ply_to_glb(input_ply: str, output_glb: str) -> bool:
    """
    PLYファイルをGLB形式に変換する
    
    Parameters
    ----------
    input_ply : str
        入力PLYファイルパス
    output_glb : str
        出力GLBファイルパス
    
    Returns
    -------
    bool
        変換の成功/失敗
    """
    # 入力ファイルの確認
    if not os.path.exists(input_ply):
        print(f"エラー: 入力ファイル '{input_ply}' が見つかりません。")
        return False
    
    # PLYファイルを読み込む
    print(f"PLYファイルを読み込み中: {input_ply}")
    try:
        mesh = trimesh.load(input_ply, file_type='ply')
    except Exception as e:
        print(f"エラー: PLYファイルの読み込みに失敗しました: {e}")
        return False
    
    # Trimeshオブジェクトか確認
    if not isinstance(mesh, trimesh.Trimesh):
        print(f"エラー: 読み込んだデータがTrimeshオブジェクトではありません。型: {type(mesh)}")
        return False
    
    # メッシュ情報の表示
    print(f"  頂点数: {len(mesh.vertices)}")
    print(f"  面数: {len(mesh.faces)}")
    
    # バウディングボックスの確認
    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    print(f"  バウディングボックス:")
    print(f"    最小: [{bbox_min[0]:.4f}, {bbox_min[1]:.4f}, {bbox_min[2]:.4f}]")
    print(f"    最大: [{bbox_max[0]:.4f}, {bbox_max[1]:.4f}, {bbox_max[2]:.4f}]")
    
    # センターとスケールを計算
    center = (bbox_min + bbox_max) / 2
    scale = bbox_max - bbox_min
    print(f"  センター: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"  スケール: [{scale[0]:.4f}, {scale[1]:.4f}, {scale[2]:.4f}]")
    
    # GLB形式で保存
    output_path = Path(output_glb)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"GLBファイルを保存中: {output_glb}")
    try:
        mesh.export(str(output_path))
    except Exception as e:
        print(f"エラー: GLBファイルの保存に失敗しました: {e}")
        return False
    
    # ファイルサイズを確認
    file_size = os.path.getsize(output_glb)
    print(f"  出力ファイルサイズ: {file_size / (1024*1024):.2f} MB")
    
    return True


def main():
    """メイン処理"""
    # デフォルトパス
    base_dir = Path(__file__).parent.parent
    input_ply = base_dir / "output" / "intermediate" / "mesh" / "mesh_intermediate.ply"
    output_glb = base_dir / "output" / "glb" / "car_model.glb"
    
    print("=" * 60)
    print("PLY -> GLB 変換")
    print("=" * 60)
    print(f"入力: {input_ply}")
    print(f"出力: {output_glb}")
    print()
    
    success = convert_ply_to_glb(str(input_ply), str(output_glb))
    
    if success:
        print()
        print("=" * 60)
        print("変換完了!")
        print("=" * 60)
    else:
        print()
        print("=" * 60)
        print("変換に失敗しました。")
        print("=" * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()
