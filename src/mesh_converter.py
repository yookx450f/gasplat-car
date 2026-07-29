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
from typing import Dict, Optional, List, Tuple
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
    
    def _estimate_normals_from_gaussians(self, gaussian_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        ガウシアンのスケールと回転から法線を推定する。
        最小スケールの軸を法線方向とする。

        Returns
        -------
        tuple (means, normals)
        """
        means = gaussian_data['means']
        quats = gaussian_data['quats']
        scales = gaussian_data['scales']

        # 回転行列の構築 (クォータニオン -> 3x3)
        # 簡易的な実装として、各ガウシアンの最小スケール軸を抽出
        # 本来は回転行列 R を計算して R * [0, 0, scale_min] を計算すべき
        
        normals = []
        for i in range(len(means)):
            q = quats[i]
            s = scales[i]
            
            # クォータニオンから回転行列への変換 (簡易版)
            qw, qx, qy, qz = q
            R = np.array([
                [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
            ])
            
            # 最小スケールの軸（法線）を計算
            # スケールが対数(log)で保存されている場合を考慮
            # ここでは単純に、回転行列の第3列（z軸方向）を法線とする
            normal = R[:, 2]
            normals.append(normal)

        return means, np.array(normals)

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
            Gaussian Splintingの結果
        
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
        avg_color = np.mean(colors, as_array=True, axis=0)
        
        # 軸方向の大きさを計算
        size = max_coords - min_coords
        
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
    
    def _generate_dense_points_from_gaussians(self, gaussian_data: Dict) -> np.ndarray:
        """
        ガウシアン中心点からポイントクラウドを生成
        
        不透明度が高いガウシアンを優先し、車の形状に適した密度でポイントを生成する。
        ガウシアンのスケール情報を使用して、表面付近にポイントを密集させる。
        
        Parameters
        ----------
        gaussian_data : dict
            Gaussian Splattingの結果
            
        Returns
        -------
        np.ndarray
            ポイントクラウド (M, 3)
        """
        means = gaussian_data['means']
        scales = gaussian_data['scales']
        opacities = gaussian_data['opacities']
        quats = gaussian_data['quats']
        
        # 不透明度でフィルタリング（sigmoid変換）
        sigmoid_opacities = 1.0 / (1.0 + np.exp(-opacities))
        mask = sigmoid_opacities > 0.05  # 不透明度が5%以上のガウシアンを使用（閾値を下げる）
        
        filtered_means = means[mask]
        filtered_scales = scales[mask] if scales is not None else None
        filtered_quats = quats[mask] if quats is not None else None
        filtered_opacities = sigmoid_opacities[mask]
        
        print(f"  ポイントクラウド生成: {len(filtered_means)} points (不透明度閾値: 0.05)")
        
        # ガウシアンのスケール情報を使用して表面付近にポイントを生成
        if filtered_scales is not None and filtered_quats is not None:
            dense_points = self._sample_gaussian_surface_points(
                filtered_means, filtered_scales, filtered_quats, filtered_opacities
            )
            print(f"  ガウシアン表面サンプリング: {len(dense_points)} points")
            return dense_points
        
        return filtered_means
    
    def _sample_gaussian_surface_points(self, means: np.ndarray, scales: np.ndarray,
                                         quats: np.ndarray, opacities: np.ndarray) -> np.ndarray:
        """
        ガウシアンのスケールと回転から表面付近のポイントをサンプリング
        
        スケールの小さいガウスアン（平坦な表面）では中心付近を、
        スケールの大きいガウシアン（曲がった表面）では縁を重点的にサンプリング。
        
        Parameters
        ----------
        means : np.ndarray
            ガウシアン中心点 (N, 3)
        scales : np.ndarray
            ガウシアンスケール (N, 3) - ログ空間の場合あり
        quats : np.ndarray
            ガウシアン回転（クォータニオン）(N, 4)
        opacities : np.ndarray
            不透明度 (N,)
            
        Returns
        -------
        np.ndarray
            サンプリングされたポイントクラウド (M, 3)
        """
        sampled_points = []
        
        # スケールが対数(log)で保存されている可能性を考慮
        # 一般的なガウススプラッティングでは、スケールはexp(log_scale)で保存される
        # ここでは、スケールの最大値が1未満の場合は対数とみなして逆変換
        max_scale = np.max(scales)
        if max_scale > 5:
            # 既に線形空間と判断
            actual_scales = scales
        else:
            # 対数空間と判断して逆変換
            actual_scales = np.exp(scales)
        
        for i in range(len(means)):
            center = means[i]
            scale = actual_scales[i]
            quat = quats[i]
            opacity = opacities[i]
            
            # クォータニオンから回転行列を構築
            qw, qx, qy, qz = quat
            R = np.array([
                [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
            ])
            
            # スケールが極端に小さい場合はスキップ
            if np.min(scale) < 1e-6:
                continue
            
            # 表面付近の点を生成（スケールの半分くらいの位置）
            # 不透明度が高いほど多くのポイントを生成
            num_samples = max(1, int(opacity * 5))  # 最大5サンプル
            
            for _ in range(num_samples):
                # ガウシアンの主軸方向に沿って表面付近の点を生成
                # スケールの平均の位置を中心とした球面上のランダム点
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                
                # 回転行列を適用してガウシアンの向きに合わせる
                direction = R @ direction
                
                # スケールの影響を受けた距離（表面付近）
                distance = np.mean(scale) * (0.5 + 0.3 * np.random.randn())
                
                point = center + direction * distance
                sampled_points.append(point)
        
        if len(sampled_points) > 0:
            return np.array(sampled_points)
        else:
            return means
    
    def _create_mesh_from_point_cloud(self, points: np.ndarray) -> Optional[trimesh.Trimesh]:
        """
        ポイントクラウドからPoisson Surface Reconstructionでメッシュを生成
        
        Open3DのPoisson Reconstructionを使用し、
        失敗した場合はAlpha Shape、最終的に凸包にフォールバックする。
        
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
        
        # アプローチ1: Open3DのPoisson Reconstructionを試みる
        print("  アプローチ1: Poisson Surface Reconstruction (Open3D) を試みる...")
        try:
            import open3d as o3d
            
            # Open3DのPointCloudオブジェクトを作成
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # 法線を推定
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            
            # Poisson Reconstruction
            mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=8  # 解像度 (6-10, 車用に8)
            )
            
            # 密度が低い頂点を削除（マスクの論理反転を修正）
            vertices_to_keep = densities > np.quantile(densities, 0.01)
            mesh_o3d.remove_vertices_by_mask(~vertices_to_keep)
            
            # trimeshに変換
            mesh = trimesh.Trimesh(
                vertices=np.asarray(mesh_o3d.vertices),
                faces=np.asarray(mesh_o3d.triangles)
            )
            
            print(f"  Poisson Reconstruction成功: {len(mesh.vertices)} 頂点, {len(mesh.faces)} 面")
            return mesh
            
        except ImportError:
            print("  警告: Open3Dがインストールされていません。")
        except Exception as e:
            print(f"  Poisson Reconstruction失敗: {e}")
        
        # アプローチ2: Alpha Shape（非凸形状）- 複数のalpha値で試行
        print("  アプローチ2: Alpha Shapeを試みる...")
        try:
            # ポイントの分布範囲を計算して、適切なalpha値を見つける
            extent = np.max(points, axis=0) - np.min(points, axis=0)
            avg_extent = np.mean(extent)
            
            # 複数のalpha値で試行（大きい値から小さい値へ）
            alphas = [avg_extent * 0.5, avg_extent * 0.3, avg_extent * 0.2, avg_extent * 0.1]
            
            for alpha in alphas:
                try:
                    vertices, triangles = trimesh.alpha_shape_mesh(points, alpha=alpha)
                    if len(vertices) > 100 and len(triangles) > 10:
                        alpha_shape = trimesh.Trimesh(vertices=vertices, faces=triangles)
                        # 最大の連結成分だけを取得（ノイズを除去）
                        components = alpha_shape.split(connected=True)
                        if components:
                            # 最大の面を持つ成分を選択
                            largest_component = max(components, key=lambda c: len(c.faces))
                            mesh = largest_component
                            print(f"  Alpha Shape成功 (alpha={alpha:.4f}): {len(mesh.vertices)} 頂点, {len(mesh.faces)} 面")
                            return mesh
                except Exception:
                    continue
            
            print("  Alpha Shape: 全てのalpha値で失敗")
            
        except Exception as e:
            print(f"  Alpha Shape失敗: {e}")
        
        # アプローチ3: 凸包（Convex Hull）- 最終フォールバック
        print("  アプローチ3: 凸包（Convex Hull）を生成する...")
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            
            # 修正: 凸包の頂点だけを抽出して使用する
            hull_vertices = points[hull.vertices]
            
            mesh = trimesh.Trimesh(
                vertices=hull_vertices,
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
        2. ガウシアン中心点からのPoisson Surface Reconstruction
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
        
        # ポイントクラウド生成（不透明度でフィルタリング）
        points = self._generate_dense_points_from_gaussians(gaussian_data)
        point_colors = (gaussian_data['colors'] * 255).astype(np.uint8)
        
        print(f"ポイントクラウド: {len(points)} points")
        
        mesh = None
        
        # アプローチ1: COLMAP dense reconstructionを試みる
        if colmap_results is not None:
            mesh = self._try_colmap_dense_reconstruction(colmap_results)
        
        # アプローチ2: ガウシアン中心点からメッシュ化（Poisson Surface Reconstruction）
        if mesh is None or len(mesh.faces) < 1000:
            print("  ガウシアン中心点からメッシュ化を試みる...")
            try:
                # ポイントクラウドがある場合はPoisson Surface Reconstructionを実行
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
