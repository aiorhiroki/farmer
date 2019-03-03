# ImageAnalyzer
You can train Classification and Segmentation tasks


## Usage

### install requirements
we utilize ncc module
```bash
$ git clone git@github.com:NCC-AI/ncc.git
$ cd ncc
$ python setup.py install
```

to build segmentation models
```bash
$ pip install segmentation-models
```

### train models
`python train.py`


## Implementation
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
