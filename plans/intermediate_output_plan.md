# 中間結果出力計画

## 概要
複数の画像から3Dモデリングを生成するプロセスにおいて、各ステップの中間結果を出力して可視化・確認できるようにします。

## 出力先ディレクトリ構造
```
output/
├── intermediate/           # 中間結果全般
│   ├── preprocessing/      # 前処理結果
│   │   ├── enhanced_images/    # 特徴点検出用強化画像
│   │   └── camera_params.json  # カメラパラメータ
│   ├── colmap/             # COLMAP結果
│   │   ├── point_cloud.png       # ポイントクラウド可視化
│   │   └── camera_positions.json # カメラ位置情報
│   ├── gsplat/             # Gaussian Splatting結果
│   │   ├── render_step_0000.png  # 訓練開始時レンダリング
│   │   ├── render_step_5000.png  # 訓練中レンダリング
│   │   └── render_step_30000.png # 訓練終了時レンダリング
│   └── mesh/               # メッシュ化結果
│       ├── point_cloud_vis.png   # ポイントクラウド可視化
│       ├── mesh_vis.png          # メッシュ可視化
│       └── gaussian_distribution.npz # ガウシアン分布データ
```

## ステップ1: 設定ファイルの更新

### config/config.yaml に追加する設定
```yaml
# 中間結果出力設定
intermediate_output:
  # 出力の有効化
  enabled: true
  # 出力ディレクトリ
  dir: "output/intermediate"
  # Gaussian Splatting訓練中の出力間隔（イテレーション数）
  gs_render_interval: 5000
```

## 実装予定ファイル

1. `src/intermediate_output.py` - 中間結果出力用の新しいモジュール
2. `config/config.yaml` - 設定ファイルの更新（上記追加）
3. `src/main.py` - 中間結果出力処理の組み込み

## 変更予定ファイル詳細

### config/config.yaml
- 末尾に `intermediate_output` セクションを追加

### src/intermediate_output.py（新規作成）
- 前処理後の画像保存
- COLMAP後のポイントクラウド可視化
- Gaussian Splatting訓練中のレンダリング結果出力
- メッシュ化後の可視化

### src/main.py
- `intermediate_output.py` のインポート追加
- 各ステップ後に中間結果出力を呼び出す処理追加

## 次のステップ
1. Codeモードに切り替え
2. 設定ファイルを更新
3. `src/intermediate_output.py` を作成
4. `src/main.py` を更新
