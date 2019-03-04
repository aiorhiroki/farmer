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

```buildoutcfg
[project_settings]
target_dir = /home/hiroki/ncc-dev/ncc/cifar10
train_dirs = train
test_dirs = test
nb_classes = 10

[default]
epoch = 50
batch_size = 1
optimizer = adam
augmentation = False

[classification_default]
model = Xception
width = 71
height = 71
backbone = xception

[segmentation_default]
model = Unet
width = 512
height = 256
backbone = resnet50
image_dir = images
label_dir = labels
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
