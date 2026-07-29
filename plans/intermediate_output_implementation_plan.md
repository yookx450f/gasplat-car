# 中間結果出力機能の実装計画

## 現状の確認

| ファイル | ステータス | 備考 |
|---------|-----------|------|
| [`config/config.yaml`](config/config.yaml) | ✅ 完了 | `intermediate_output` セクションが追加済み |
| [`src/intermediate_output.py`](src/intermediate_output.py) | ⚠️ 改善必要 | 作成済みだが、COLMAP/Mesh保存ロジックが未完成 |
| [`src/main.py`](src/main.py) | ❌ 未実装 | インポートと各ステップでの呼び出しが必要 |
| [`src/train_gsplat.py`](src/train_gsplat.py) | ⚠️ オプション | 訓練中のレンダリング出力を追加可能 |

---

## 変更計画

### 1. [`src/intermediate_output.py`](src/intermediate_output.py) の改善

#### 現状の問題点
- [`save_colmap_results()`](src/intermediate_output.py:49) - コメントのみで実際の保存ロジックがない
- [`save_mesh_results()`](src/intermediate_output.py:74) - コメントのみで実際の保存ロジックがない

#### 改善内容
```
追加する機能:
├── save_colmap_results() の実装
│   ├── カメラ位置情報を JSON に保存
│   └── ポイントクラウドの簡易可視化画像を生成（trimesh使用）
├── save_mesh_results() の実装
│   └── メッシュデータを .ply として中間出力先にコピー
└── save_preprocessing_results() の改善
    └── 前処理済み画像のコピーを追加
```

### 2. [`src/main.py`](src/main.py) の変更

#### 追加する処理フロー
```
run_pipeline():
│
├── [NEW] IntermediateOutputManager(config) を初期化
│
├── ステップ1: 画像前処理
│   └── [NEW] intermediate_manager.save_preprocessing_results(processed_data)
│
├── ステップ2: COLMAP
│   └── [NEW] intermediate_manager.save_colmap_results(colmap_results)
│
├── ステップ3: Gaussian Splatting 訓練
│   └── [OPTIONAL] 訓練中のレンダリング出力（train_gsplat.py側で実装）
│
├── ステップ4: メッシュ化
│   └── [NEW] intermediate_manager.save_mesh_results(mesh_data)
│
└── ステップ5: 出力エクスポート
```

#### 具体的な変更箇所
| 行番号 | 変更内容 |
|-------|---------|
| 23行付近 | `from intermediate_output import IntermediateOutputManager` を追加 |
| 58行付近 | `intermediate_manager = IntermediateOutputManager(config)` を追加 |
| 69行後 | `intermediate_manager.save_preprocessing_results(processed_data)` を追加 |
| 79行後 | `intermediate_manager.save_colmap_results(colmap_results)` を追加 |
| 104行後 | `intermediate_manager.save_mesh_results(mesh_data)` を追加 |

### 3. [`src/train_gsplat.py`](src/train_gsplat.py) の変更（オプション）

#### 訓練中のレンダリング出力を追加
```
train() メソッドのループ内 (353行目付近):
├── gs_render_interval ごとにレンダリング結果を保存
└── intermediate_manager.save_gsplat_render(iteration, render_image) を呼び出す
```

---

## 出力ディレクトリ構造

```
output/intermediate/
├── preprocessing/
│   ├── enhanced_images/      # 前処理済み画像のコピー
│   └── camera_params.json    # カメラパラメータ
├── colmap/
│   ├── point_cloud.png       # ポイントクラウド可視化
│   └── camera_positions.json # カメラ位置情報
├── gsplat/
│   ├── render_step_000001.png  # 訓練開始時
│   ├── render_step_005000.png  # 5000イテレーション目
│   └── ...
└── mesh/
    ├── mesh_intermediate.ply   # メッシュデータ
    └── gaussian_distribution.npz # ガウシアン分布データ
```

---

## 実装順序

1. **`src/intermediate_output.py` の改善** - 未完成のメソッドを実装
2. **`src/main.py` の変更** - インポートと各ステップでの呼び出しを追加
3. **`src/train_gsplat.py` の変更**（オプション）- 訓練中のレンダリング出力
4. **動作確認** - Docker でビルド・実行
