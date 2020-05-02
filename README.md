# farmer

You can train Classification and Segmentation tasks semi-automatically

## Prerequisite

Docker >= 19.03

(python library) jinja2

build docker
```bash
docker build -t tensorflow:v2 .
```

run container in the same path with run.yaml
```
docker run \
    --gpus all \
    -it \
    --rm \
    -v $PWD:/tmp \
    -w /tmp \
    poetry:v1 \
    poetry run Godfarmer
    # poetry run python  # python shell with docker env
    # bash  # login


# options to develop farmer
# -v /Path_To_Farmer:/app
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

## Result

```yaml
- result_directory
  - image (sample image)
  - info (config param & image path)
  - learning (learning history)
  - model (best model and last model)
```

### Integration Test

cd example & run container

