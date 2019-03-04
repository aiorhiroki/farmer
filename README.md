# farmer
You can train Classification and Segmentation tasks as best practice




### Requirements: ncc
```bash
$ git clone git@github.com:NCC-AI/ncc.git
$ cd ncc
$ python setup.py install
```
### Installation
```bash
$ pip install git+https://github.com/NCC-AI/farmer
```

### Prepare Data set folder
- classification


    │
    ├── target_directory
          │ 
          ├── data_case_directory(dataA) # caseごとだったり、train/testだったり
          │      │ 
          │      ├── category_directory(Orange)  # クラスのフォルダー
          │      │       │      
          │      │       ├── image_file(jpg or png)
          │      │
          │      ├── category_directory(Apple)
          │              │      
          │              ├── image_file(jpg or png)
          │     
          ├── data_case_directory(dataB)

- segmentation


    │
    ├── target_directory
          │ 
          ├── data_case_directory(dataA) # caseごとだったり、train/testだったり
          │      │ 
          │      ├── input_image_directory  # 入力画像フォルダ
          │      │       │      
          │      │       ├── image_file(jpg or png)
          │      │
          │      ├── mask_image_directory  # マスク画像フォルダ
          │              │      
          │              ├── image_file(jpg or png)
          │     
          ├── data_case_directory(dataB)

### Training
`config.ini`に学習条件を書き込み、このファイルがある場所で以下のコマンドを実行 

```bash
$ ncc-cls  # classification
```

```bash
$ ncc-seg  # segmentation
```

`secret.ini`にスラックの情報を書き込めば、ログ画像を飛ばせる。

Result
------------
- 実行した日付時間で自動にフォルダを作成。結果を以下のディレクトリ構造で保存されます。


    │ 
    ├── result
          ├── 日付時間(結果A)
          │      ├── image  # 推論サンプル
          │      ├── info  # 設定ファイル/画像パス
          │      ├── learning  # 学習履歴 
          │      ├── model  # 最良モデル & 最終モデル
          │           
          ├── 日付時間(結果B)
