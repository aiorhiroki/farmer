# farmer

You can train Classification and Segmentation tasks semi-automatically

## Set Up

Docker >= 19.03

```bash
docker build -t tensorflow:v2 .
sh docker-start.sh
```

### Prepare Data set folder

classification

```tree
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
```

segmentation

```tree
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
```

## Training

`classification-config.ini`または`segmentation-config.ini` ファイルがある場所に
`run.ini`も置いて以下のコマンドを実行

```bash
Godfarmer
```

`secret.ini`を作成し`config.ini`と同じ場所に配置すれば、Slackにログ画像を飛ばせる。

```buildoutcfg
[DEFAULT]
slack_token = xoxb-hogehoge
slack_channel = fugafuga
```

## Result

実行した日付時間で自動にフォルダを作成。結果を以下のディレクトリ構造で保存されます。

```tree
    │
    ├── result
          ├── 日付時間(結果A)
          │      ├── image  # 推論サンプル
          │      ├── info  # 設定ファイル/画像パス
          │      ├── learning  # 学習履歴
          │      ├── model  # 最良モデル & 最終モデル
          │
          ├── 日付時間(結果B)
```

## Test and Format

`$  pipenv run nox`

## Trouble Shooting

If you can't use docker.

```bash
export DOCKER_API_VERSION=1.40
sudo systemctl start docker
```
