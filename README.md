# farmer
You can train Classification and Segmentation tasks as best practice


### Installation
```bash
$ pip install git+https://github.com/aiorhiroki/ncc
$ pip install git+https://github.com/aiorhiroki/farmer
```

Docker <= 18.09.0 (Don't use v19.03 or newer)

### Prepare Data set folder
classification


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

segmentation


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
`classification-config.ini` ファイルがある場所でコマンドを実行
```bash
$ ncc-cls  # classification
```

`segmentation-config.ini` ファイルがある場所でコマンドを実行
```bash
$ ncc-seg  # segmentation
```

`secret.ini`を作成し`config.ini`と同じ場所に配置すれば、Slackにログ画像を飛ばせる。
```buildoutcfg
[DEFAULT]
slack_token = xoxb-hogehoge
slack_channel = fugafuga
```

Result
------------
実行した日付時間で自動にフォルダを作成。結果を以下のディレクトリ構造で保存されます。


    │ 
    ├── result
          ├── 日付時間(結果A)
          │      ├── image  # 推論サンプル
          │      ├── info  # 設定ファイル/画像パス
          │      ├── learning  # 学習履歴 
          │      ├── model  # 最良モデル & 最終モデル
          │           
          ├── 日付時間(結果B)

Test and Format
------------
`$  pipenv run nox`