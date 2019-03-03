# Classification


## Usage

### install requirements
we use ncc module

### train models
`python train.py`


## Implementation
- [tensorflowの実装](https://github.com/tks10/segmentation_unet/blob/master/util/loader.py)を参考にKerasで実装した。
- ネットワーク構造は[Semgentation Model](https://github.com/qubvel/segmentation_models)を使い自由に変更できる。
- Reporterクラスがすべての処理をやってくれる。
- 実行した日付時間で自動にフォルダを作成。結果を以下のディレクトリ構造で保存

Project Organization
------------
    ├── train.py
    │       
    ├── utils
    │       ├── reporter.py  # 画像の読み込み、結果の保存
    │       ├── reader.py  # 各自ファイルロードのコード
    │       ├── parser.py  # 設定
    │       ├── model.py  # convulutional neural networks
    │ 
    ├── result
          ├── 日付時間(結果A)
          │      ├── image  # 推論サンプル
          │      ├── info  # 設定ファイル
          │      ├── learning  # 学習履歴 
          │      ├── model  # 最良モデル & 最終モデル
          │           
          ├── 日付時間(結果B)
