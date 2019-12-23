# farmer

You can train Classification and Segmentation tasks semi-automatically

## Set Up

Docker >= 19.03

```bash
docker build -t tensorflow:v2 .
sh docker-start.sh container_name port
```

### Prepare Data set folder

classification folder tree

e.g.)
- target_directory
  - data_case_directory(dataA) 
      - category_directory(Orange)
      - category_directory(Apple)
  - data_case_directory(dataB)


segmentation folder tree

e.g.)
- target_directory
  - data_case_directory(dataA)
    - input_image_directory
    - mask_image_directory
  - data_case_directory(dataB)


## Training

1. set param in `classification~~.ini` for classification
1. set param in `segmentation~~.ini` for segmentation
1. set param `run.ini`
1. set train, test and validation data folders in `data.json`
1. run `Godfarmer`

### Slack logging

set param in `secret.ini`
```buildoutcfg
[DEFAULT]
slack_token = xoxb-hogehoge
slack_channel = fugafuga
```

## Result

  - result_directory
    - image (sample image)
    - info (config param & image path)
    - learning (learning history)
    - model (best model and last model)


## Test and Format

### Unit Testing
```
pipenv run nox
```

### Integration Testing
```
cd exmaple
Godfarmer 
```

