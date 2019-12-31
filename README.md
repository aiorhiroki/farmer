# farmer

You can train Classification and Segmentation tasks semi-automatically

## Prerequisite

Docker >= 19.03

```bash
docker build -t tensorflow:v2 .
```

## Prepare Data set folder

classification folder tree

e.g.)

```yaml
- target_directory
  - data_case_directory(dataA)
    - category_directory(Orange)
    - category_directory(Apple)
  - data_case_directory(dataB)
```

segmentation folder tree

e.g.)

```yaml
- target_directory
  - data_case_directory(dataA)
    - input_image_directory
    - mask_image_directory
  - data_case_directory(dataB)
```

## Training

1. start docker  
  `$ sh docker-start.sh container_name port`
1. set param in `~.yaml` and `run.yaml`
1. run `$ Godfarmer`

### Slack logging

set param in `secret.yaml`

```yaml
slack_token: xoxb-hogehoge
slack_channel: fugafuga
```

## Result

```yaml
- result_directory
  - image (sample image)
  - info (config param & image path)
  - learning (learning history)
  - model (best model and last model)
```

## Test and Format

### Unit Testing

```bash
pipenv run nox
```

### Integration Testing

```bash
cd exmaple
Godfarmer
```
