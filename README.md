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
`config.ini`ファイルを作成。学習条件を書き込む。 
`train_data`と`test_data`は半角スペースを開けてdata_case_directoryを列挙する。  
またはファイルパスとラベルがセットになったcsvファイルのパスを指定する。
```buildoutcfg
[project_settings]
target_dir = /mnt/hdd/data/Forceps/selected/data
train_data = 1 2 3 5 7 8 10 11 12 13 14 15 16 17 19 20 21 22 25 26 27 28 32 34 35 36
test_data = 4 6 9 18 23 24 29 33
nb_classes = 6

[default]
epoch = 100
batch_size = 4
optimizer = adam
augmentation = False

[classification_default]
model = Xception
width = 71
height = 71
backbone = xception

[segmentation_default]
model = WithOutOthers
width = 512
height = 256
backbone = resnet50
image_dir = images
label_dir = labels
class_names = Background Clippper Maryland Linear Point Grasper
```

`config.ini`ファイルがある場所でコマンドを実行 

```bash
$ ncc-cls  # classification
```

```bash
$ ncc-seg  # segmentation
```

`secret.ini`を作成し`config.ini`と同じ場所に配置すれば、Slackにログ画像を飛ばせる。
```buildoutcfg
[default]
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
