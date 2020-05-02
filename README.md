# farmer

You can train Classification and Segmentation tasks semi-automatically

## Prerequisite

### `install docker`
- Docker >= 19.03

### `build docker`
```bash
docker build -t tensorflow:v2 .
```

### `register env & command`

```bash
# bash

# write farmer path in .bash_profile
echo "export FARMERPATH=$PWD" >> ~/.bash_profile

# ...or in .bashrc
echo "export FARMERPATH=$PWD" >> ~/.bashrc

# fish
echo "set -x FARMERPATH $PWD" >> ~/.config/fish/config.fish
```

#### **`~/.bash_aliases`**
```bash
dogrun () {
    docker run \
        --gpus all \
        -it \
        --rm \
        -v $PWD:/tmp -v $1 \
        -w /tmp \
        tensorflow:v2 \
        poetry run Godfarmer
}

dogdev () {
    docker run \
        --gpus all \
        -it \
        --rm \
        -v $PWD:/tmp -v $FARMERPATH:/app $1 \
        -w /tmp \
        tensorflow:v2 \
        poetry run Godfarmer
}

dogpy () {
    docker run \
        --gpus all \
        -it \
        --rm \
        -v $PWD:/tmp -v $FARMERPATH:/app $1 \
        -w /tmp \
        tensorflow:v2 \
        poetry run python
}

dogin () {
    docker run \
        --gpus all \
        -it \
        --rm \
        -v $PWD:/tmp \
        -v $FARMERPATH:/app $1 \
        -w /tmp \
        tensorflow:v2 \
        bash
}
```

#### **`~/.config/fish/config.fish`**
``` bash
function dogrun
    docker run \
        --gpus all \
        -it \
        --rm \
        -v $PWD:/tmp $argv \
        -w /tmp \
        tensorflow:v2 \
        poetry run Godfarmer
end

function dogdev
    docker run \
        --gpus all \
        -it \
        --rm \
        -v $PWD:/tmp -v $FARMERPATH:/app $argv \
        -w /tmp \
        tensorflow:v2 \
        poetry run Godfarmer
end

function dogpy
    docker run \
        --gpus all \
        -it \
        --rm \
        -v $PWD:/tmp -v $FARMERPATH:/app $argv \
        -w /tmp \
        tensorflow:v2 \
        poetry run python
end

function dogin
    docker run \
        --gpus all \
        -it \
        --rm \
        -v $PWD:/tmp \
        -v $FARMERPATH:/app $argv \
        -w /tmp \
        tensorflow:v2 \
        bash
end
```


```bash
# command list
dogrun  # run farmer @ master code
dogdev  # run farmer @ local code
dogpy   # python shell @ docker env
dogin   # login docker

# with additional mount
# bash 
dogin "-v /data:/data"
# fish
dogin -v /data:/data
```
* **dogrun** and **dogdev** needs run.yaml in the same path

## Prepare Data set folder

#### **`classification folder tree`**

```yaml
- target_directory
  - data_case_directory(dataA)
    - category_directory(Orange)
    - category_directory(Apple)
  - data_case_directory(dataB)
```

#### **`segmentation folder tree`**

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

## Integration Test

```
cd example
dogrun
```

