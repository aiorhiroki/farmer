# farmer

You can train Classification and Segmentation tasks semi-automatically

## Prerequisite

Docker >= 19.03

(python library) jinja2

build docker
```bash
docker build -t tensorflow:v2 .
```

run container
```
docker run --gpus all --rm -v $PWD:/tmp -w /tmp tensorflow:v2 poetry run Godfarmer
```

other usages
```
docker run -it OPTIONS bash  # login
docker run -it OPTIONS poetry run python  # python with docker env
docker run -v Path_To_Farmer:/app OPTIONS  # develop farmer by local file change
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

```bash
cd example
docker run --gpus all --rm -v $PWD:/tmp -w /tmp tensorflow:v2 poetry run Godfarmer
```
