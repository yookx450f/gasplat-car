#!/usr/bin/env python3
"""
Gaussian Splatting訓練モジュール

COLMAPで推定したカメラパラメータと画像から、3Dガウシアン分布を最適化する。
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# gsplat関連のインポート
gsplat_available = False
try:
    from gsplat import rasterization, rendering, proj, fully_fused_projection
    from gsplat import spherical_harmonics
    gsplat_available = True
except ImportError:
    print("警告: gsplatがインストールされていません。pip install gsplat を実行してください。")

# 画像処理
from PIL import Image


@dataclass
class TrainingConfig:
    """訓練設定"""
    num_iterations: int = 30000
    seed: int = 42
    learning_rate: float = 0.01
    optimizer: dict = field(default_factory=lambda: {
        'type': 'adam',
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-15
    })


@dataclass
class GaussianData:
    """Gaussian Splattingの3Dデータ"""
    means: np.ndarray = None           # 位置 (N, 3)
    quats: np.ndarray = None           # 回転（クォータニオン） (N, 4)
    scales: np.ndarray = None          # スケール (N, 3)
    opacities: np.ndarray = None       # 不透明度 (N,)
    colors: np.ndarray = None          # 色 (N, 3)
    camera_params: List[Dict] = None   # カメラパラメータ
    image_paths: List[str] = None      # 画像パス


class GaussianSplattingTrainer:
    """Gaussian Splatting訓練クラス"""
    
    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            設定ファイル
        """
        self.config = config
        self.training_config = TrainingConfig(**config.get('training', {}))
        
        # 乱数シードの設定
        torch.manual_seed(self.training_config.seed)
        np.random.seed(self.training_config.seed)
        
        # GPUの確認
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用デバイス: {self.device}")
        
        if not torch.cuda.is_available():
            print("警告: GPUが検出されません。訓練が遅くなります。")
        
        # 出力ディレクトリ
        self.output_dir = Path(config['output']['dir']) / 'gsplat'
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_images(self, image_paths: List[str]) -> List[torch.Tensor]:
        """画像を読み込み、テンソルに変換（すべて同じサイズにリサイズ）"""
        images = []
        target_size = None
        
        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            
            # 最初の画像からターゲットサイズを取得
            if target_size is None:
                target_size = img.size  # (width, height)
            
            # 画像をリサイズ
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.BICUBIC)
            
            img_tensor = torch.tensor(np.array(img), dtype=torch.float32, device=self.device)
            img_tensor = img_tensor.permute(2, 0, 1) / 255.0  # (C, H, W)
            images.append(img_tensor)
        return images
    
    def _initialize_gaussians(self, colmap_results: Dict, num_gaussians: int = 100000) -> Dict:
        """
        ガウシアンを初期化
        
        COLMAPの3Dポイントから初期位置を抽出し、スケールと回転を適応的に設定する。
        
        Parameters
        ----------
        colmap_results : dict
            COLMAP処理結果
        num_gaussians : int
            初期化するガウシアン数
        
        Returns
        -------
        dict
            初期化されたGaussianデータ
        """
        points3D = colmap_results.get('points3D')
        
        if points3D is None or len(points3D) == 0:
            print("警告: 3Dポイントが見つかりません。ランダム初期化を使用します。")
            # ランダム初期化
            means = np.random.uniform(-5, 5, (num_gaussians, 3))
            # カラーもランダムに生成
            colors = np.random.rand(num_gaussians, 3)
        else:
            # 3Dポイントをサンプリング
            if len(points3D) > num_gaussians:
                indices = np.random.choice(len(points3D), num_gaussians, replace=False)
                sampled_points = points3D[indices]
            else:
                sampled_points = points3D
            
            means = sampled_points[:, :3]
            colors = sampled_points[:, 3:6] / 255.0
            
            # ガウシアン数を調整
            num_gaussians = len(means)
        
        # スケールを計算（近傍ポイントに基づいて）- gsplatはlogスケールを期待
        if len(means) > 1:
            dists = np.linalg.norm(means[1:] - means[:-1], axis=1)
            median_dist = np.median(dists) if len(dists) > 0 else 0.1
            # logスケール: 小さな値で初期化（後で最適化される）
            # 極端に小さい値にならないように下限を緩和（0.001 → 0.01）
            scales = np.full((num_gaussians, 3), np.log(max(median_dist * 0.5, 0.01)))
        else:
            # デフォルト: log(0.01) ≈ -4.6
            scales = np.full((num_gaussians, 3), np.log(0.01))
        
        # 回転（クォータニオン）- 初期値はアイデンティティ
        quats = np.zeros((num_gaussians, 4))
        quats[:, 0] = 1.0  # qw = 1, qx = qy = qz = 0
        
        # 不透明度 - 初期値を少し高い値に設定（sigmoid前に適度な値）
        # 低い値だとガウシアンが透明すぎてレンダリングが真っ黒になる
        opacities = np.full(num_gaussians, -0.5)  # sigmoid(-0.5) ≈ 0.38
        
        gaussian_data = {
            'means': means.astype(np.float32),
            'quats': quats.astype(np.float32),
            'scales': scales.astype(np.float32),
            'opacities': opacities.astype(np.float32),
            'colors': colors.astype(np.float32) if colors is not None else np.random.rand(num_gaussians, 3).astype(np.float32)
        }
        
        return gaussian_data
    
    def _prepare_camera_matrices(self, colmap_results: Dict) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        カメラ行列を準備
        
        Parameters
        ----------
        colmap_results : dict
            COLMAP処理結果
        
        Returns
        -------
        tuple
            カメラ行列、変換行列、画像サイズ
        """
        images = colmap_results.get('images', {})
        cameras = colmap_results.get('cameras', {})
        
        print(f"  COLMAP結果: {len(images)}画像, {len(cameras)}カメラ")
        
        camera_matrices = []
        image_sizes = []
        
        for cam_id, cam_data in images.items():
            if cam_id not in cameras:
                continue
            
            cam_params = cameras[cam_id]
            R = cam_data['rotation']
            t = cam_data['translation']
            
            # カメラ行列を構築
            if 'fx' in cam_params:
                fx = cam_params['fx']
                fy = cam_params.get('fx', fx)  # simple_pinhoneの場合
                cx = cam_params['cx']
                cy = cam_params['cy']
                width = cam_params['width']
                height = cam_params['height']
                
                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
            else:
                # デフォルト値
                width = 1024
                height = 1024
                K = np.array([
                    [width * 1.2, 0, width / 2],
                    [0, height * 1.2, height / 2],
                    [0, 0, 1]
                ], dtype=np.float32)
            
            # 外部行列: R|t
            RT = np.hstack([R, t.reshape(3, 1)])  # (3, 4)
            P = K @ RT  # (3, 4)
            
            camera_matrices.append({
                'K': torch.tensor(K, device=self.device),
                'RT': torch.tensor(RT, device=self.device),
                'width': width,
                'height': height,
                'image_size': (height, width)
            })
            
            image_sizes = (height, width)
        
        print(f"  準備されたカメラ行列数: {len(camera_matrices)}")
        
        return camera_matrices, image_sizes
    
    def _create_fallback_camera_matrices(self, images: List[torch.Tensor], image_size: tuple) -> List[Dict]:
        """
        カメラ行列がない場合のフォールバックカメラ行列を生成
        
        Parameters
        ----------
        images : list
            画像テンソルのリスト
        image_size : tuple
            画像サイズ (height, width) - 空の場合、画像から取得
        
        Returns
        -------
        list
            フォールバックカメラ行列のリスト
        """
        camera_matrices = []
        
        # 画像サイズを取得（image_sizeが空の場合は画像から取得）
        if image_size and len(image_size) == 2:
            height, width = image_size
        else:
            # 画像テンソルからサイズを取得 (C, H, W)
            first_img = images[0]
            height, width = first_img.shape[1], first_img.shape[2]
        
        # 単純な正面カメラ行列を生成
        fx = width * 1.2
        fy = height * 1.2
        cx = width / 2
        cy = height / 2
        
        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)
        
        # 単位回転行列（正面）
        RT = torch.eye(3, 4, dtype=torch.float32, device=self.device)
        # カメラを少し遠くに配置
        RT[2, 3] = -5.0  # z軸方向に5単位
        
        for _ in range(len(images)):
            camera_matrices.append({
                'K': K,
                'RT': RT.clone(),
                'width': width,
                'height': height,
                'image_size': (height, width)
            })
        
        return camera_matrices
    
    def train(self, colmap_results: Dict, intermediate_manager=None) -> GaussianData:
        """
        Gaussian Splattingの訓練を実行
        
        Parameters
        ----------
        colmap_results : dict
            COLMAP処理結果
        intermediate_manager : IntermediateOutputManager or None
            中間結果出力マネージャー（オプション）
        
        Returns
        -------
        GaussianData
            訓練されたGaussianデータ
        """
        print("画像を読み込み中...")
        image_paths = colmap_results['image_paths']
        images = self._load_images(image_paths)
        
        print("カメラ行列を準備中...")
        camera_matrices, image_size = self._prepare_camera_matrices(colmap_results)
        
        # カメラ行列が空の場合、フォールバックを使用
        if len(camera_matrices) == 0:
            print("  警告: カメラ行列が見つかりません。フォールバックカメラを使用します。")
            camera_matrices = self._create_fallback_camera_matrices(images, image_size)
        
        print("ガウシアンを初期化中...")
        gaussian_init = self._initialize_gaussians(colmap_results)
        
        # 追加デバッグログ
        print(f"  [DEBUG] COLMAP points3D数: {len(colmap_results.get('points3D', [])) if colmap_results.get('points3D') is not None else 0}")
        print(f"  [DEBUG] COLMAP images数: {len(colmap_results.get('images', {}))}")
        print(f"  [DEBUG] ガウシアン初期化後 - means形状: {gaussian_init['means'].shape}")
        print(f"  [DEBUG] ガウシアン初期化後 - colors形状: {gaussian_init['colors'].shape}")
        
        # PyTorchテンソルに変換
        means = torch.tensor(gaussian_init['means'], dtype=torch.float32, device=self.device)
        quats = torch.tensor(gaussian_init['quats'], dtype=torch.float32, device=self.device)
        scales = torch.tensor(gaussian_init['scales'], dtype=torch.float32, device=self.device)
        opacities = torch.tensor(gaussian_init['opacities'], dtype=torch.float32, device=self.device)
        colors = torch.tensor(gaussian_init['colors'], dtype=torch.float32, device=self.device)
        
        # 最適化可能なパラメータを設定
        means.requires_grad = True
        quats.requires_grad = True
        scales.requires_grad = True
        opacities.requires_grad = True
        color_lrs = color = torch.tensor(gaussian_init['colors'], dtype=torch.float32, device=self.device)
        color_lrs.requires_grad = True
        
        # オプタイマイザ
        params = [
            {'params': [means], 'lr': self.training_config.learning_rate * 0.01},
            {'params': [quats], 'lr': self.training_config.learning_rate * 0.01},
            {'params': [scales], 'lr': self.training_config.learning_rate * 0.01},
            {'params': [opacities], 'lr': self.training_config.learning_rate * 0.1},
            {'params': [color_lrs], 'lr': self.training_config.learning_rate},
        ]
        
        optimizer = torch.optim.Adam(params, lr=self.training_config.learning_rate * 0.001)
        
        num_iterations = self.training_config.num_iterations
        num_images = len(images)
        
        # デバッグログ: パラメータの確認
        print(f"訓練開始: {num_iterations} イテレーション, {num_images} 画像")
        print(f"  ガウシアン数: {means.shape[0]}")
        print(f"  means範囲: [{means.min():.4f}, {means.max():.4f}]")
        print(f"  scales範囲: [{scales.min():.4f}, {scales.max():.4f}]")
        print(f"  opacities範囲: [{opacities.min():.4f}, {opacities.max():.4f}]")
        print(f"  colors範囲: [{colors.min():.4f}, {colors.max():.4f}]")
        print(f"  カメラ数: {len(camera_matrices)}")
        if len(camera_matrices) > 0:
            cam0 = camera_matrices[0]
            print(f"  カメラ0 K:\n{cam0['K']}")
            print(f"  カメラ0 RT:\n{cam0['RT']}")
            print(f"  カメラ0 サイズ: {cam0['width']}x{cam0['height']}")
        
        # 最初のイテレーションでフォールバックカメラを使用している場合、警告
        if len(camera_matrices) == 0:
            print("  警告: カメラ行列が空です。レンダリングが正しく行われない可能性があります。")
        
        # 訓練ループ
        for i in range(num_iterations):
            # ランダムな画像を選択
            img_idx = np.random.randint(num_images)
            target_image = images[img_idx]
            
            # カメラ行列が空の場合はスキップ
            if len(camera_matrices) == 0:
                if i == 0:
                    print("  エラー: カメラ行列が空のため訓練できません。COLMAP結果を確認してください。")
                break
            
            cam = camera_matrices[img_idx]
            
            K = cam['K']
            viewmat = cam['RT']
            height, width = cam['image_size']
            
            # viewmatを4x4行列に変換
            viewmat_4x4 = torch.eye(4, device=self.device)
            viewmat_4x4[:3, :] = viewmat
            viewmats = viewmat_4x4[None, :, :]  # (1, 4, 4)
            
            # Kを(1, 3, 3)行列に変換
            Ks = K[None, :, :]  # (1, 3, 3)
            
            # ガウシアンをカメラ座標に変換
            means_3d = means  # (N, 3)
            
            # gsplat 1.5.3 APIを使用: rasterization関数で直接レンダリング
            # ライブラリ内部で投影処理が行われる
            render_colors, render_alphas, meta = rasterization(
                means_3d,
                quats,
                scales,
                opacities,
                color_lrs,
                viewmats,
                Ks,
                width,
                height,
            )
            
            # 損失計算 (render_colors shape: (height, width, 3))
            # target_imageは(3, height, width)なので、(height, width, 3)に変換
            target_rgb = target_image.permute(1, 2, 0)
            rgb_loss = torch.nn.functional.l1_loss(render_colors[0], target_rgb)
            
            # 各種罰則項
            scale_reg = scales.abs().max(dim=-1).values.mean()
            
            loss = rgb_loss + 0.001 * scale_reg
            
            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # ノーマライゼーション（クォータニオン）
            with torch.no_grad():
                quats_norm = torch.norm(quats, dim=-1, keepdim=True)
                quats = quats / quats_norm
            
            # 中間結果出力（レンダリング画像の保存）
            if intermediate_manager is not None:
                render_image = (render_colors[0].detach().cpu().numpy() * 255).astype(np.uint8)
                # 値域を0-255にクリップ
                render_image = np.clip(render_image, 0, 255)
                intermediate_manager.save_gsplat_render(i + 1, render_image)
            
            # レンダリング結果の追加デバッグ（最初の10イテレーションと每5000イテレーション）
            if i < 10 or (i + 1) % 5000 == 0:
                # alphaチャンネルの平均（透明すぎる場合、ガウシアンが正しくラスタライズされていない）
                if render_alphas is not None:
                    alpha_stats = render_alphas[0].detach().cpu().numpy()
                    print(f"    [ALPHA]  mean={alpha_stats.mean():.4f}, min={alpha_stats.min():.4f}, max={alpha_stats.max():.4f}, >0.01={np.sum(alpha_stats > 0.01)}ピクセル")
            
            # 最初の数イテレーションでレンダリング結果の統計を出力
            if i < 5 or (i + 1) % 1000 == 0:
                render_mean = render_colors[0].mean().item()
                render_std = render_colors[0].std().item()
                render_min = render_colors[0].min().item()
                render_max = render_colors[0].max().item()
                
                # alphaチャンネルの統計
                if render_alphas is not None:
                    alpha_mean = render_alphas[0].mean().item()
                    alpha_min = render_alphas[0].min().item()
                    alpha_max = render_alphas[0].max().item()
                else:
                    alpha_mean = alpha_min = alpha_max = -1
                
                # カメラ座標系でのガウシアン位置を確認
                cam = camera_matrices[img_idx]
                RT = cam['RT']
                # ガウシアンの一部をカメラ座標に変換（最初の10個のみ）
                num_debug = min(10, means_3d.shape[0])
                means_cam = (RT[:3, :3] @ means_3d[:num_debug].T).T + RT[:3, 3]
                
                print(f"  イテレーション {i + 1}/{num_iterations}, Loss: {loss.item():.4f}")
                print(f"    render_colors: mean={render_mean:.4f}, std={render_std:.4f}, min={render_min:.4f}, max={render_max:.4f}")
                print(f"    render_alphas: mean={alpha_mean:.4f}, min={alpha_min:.4f}, max={alpha_max:.4f}")
                print(f"    ガウシアン位置(カメラ座標): xyz範囲=[{means_cam[:,2].min():.2f}, {means_cam[:,2].max():.2f}]")
        
        # 結果を保存
        print("結果を保存中...")
        self._save_results(means, quats, scales, opacities, color_lrs, camera_matrices)
        
        # GaussianDataを返す
        gaussian_data = GaussianData(
            means=means.detach().cpu().numpy(),
            quats=quats.detach().cpu().numpy(),
            scales=scales.detach().cpu().numpy(),
            opacities=opacities.detach().cpu().numpy(),
            colors=color_lrs.detach().cpu().numpy(),
            camera_params=camera_matrices,
            image_paths=image_paths
        )
        
        print(f"訓練完了。出力: {self.output_dir}")
        
        return gaussian_data
    
    def _save_results(self, means, quats, scales, opacities, colors, camera_matrices):
        """訓練結果を保存"""
        # NumPy配列に変換して保存（勾配情報を削除）
        np.savez(
            str(self.output_dir / 'gaussian_params.npz'),
            means=means.detach().cpu().numpy(),
            quats=quats.detach().cpu().numpy(),
            scales=scales.detach().cpu().numpy(),
            opacities=opacities.detach().cpu().numpy(),
            colors=colors.detach().cpu().numpy()
        )
        
        # カメラパラメータをJSONとして保存
        camera_data = []
        for cam in camera_matrices:
            camera_data.append({
                'K': cam['K'].cpu().numpy().tolist(),
                'RT': cam['RT'].cpu().numpy().tolist(),
                'width': cam['width'],
                'height': cam['height']
            })
        
        with open(str(self.output_dir / 'cameras.json'), 'w', encoding='utf-8') as f:
            json.dump(camera_data, f, indent=2)
        
        print(f"  保存先: {self.output_dir}")
