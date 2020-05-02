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
sh docker-start.sh farmer 5000
```

exec container
```
docker exec -it --user $USER farmer-dev bash --login
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

1. start docker `$ docker exec -it farmer bash`
1. set param in `~.yaml` and `run.yaml`
1. set param in `secret.yaml` (optional for slack logger)
1. run `$ Godfarmer`

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
Godfarmer
```

## For Developer
If you change files, run following command before *Godfarmer*
```bash
python setup.py develop --user
```
