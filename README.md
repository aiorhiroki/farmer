# farmer

You can train Classification and Segmentation tasks semi-automatically

## Prerequisite

### `install docker`
- Docker >= 19.03

### `build docker`
```bash
docker build -t farmer:v2 .
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

# run docker container
```bash
docker run \
    --gpus all \
    -itd \
    -v $FARMERPATH:/app \
    -v /mnt/hdd2:/mnt/hdd2 \
    --name farmer \
    farmer:v2 \

docker exec farmer bash -c "poetry run python setup.py develop"
```

#### **`~/.bash_aliases`**
```bash
dogon () {
    docker exec -u (id -u):(id -g) farmer bash -c "cd $PWD && Godfarmer"
}

dogrun () {
    docker exec -it -u (id -u):(id -g) farmer bash -c "cd $PWD && $1"
}

dogin () {
    docker exec -it -u (id -u):(id -g) farmer bash
}
```

#### **`~/.config/fish/config.fish`**
``` bash
function dogon
    docker exec -u (id -u):(id -g) farmer bash -c "cd $PWD && Godfarmer"
end

function dogrun
    docker exec -it -u (id -u):(id -g) farmer bash -c "cd $PWD && $argv"
end

function dogin
    docker exec -it -u (id -u):(id -g) farmer bash
end
```


```bash
# command list
dogon  # run farmer
nohup dogon > out.log &  # run farmer in the background
dogrun COMMAND  # run command in docker
dogin   # login docker
```

* **dogon** needs run.yaml in the same path

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
dogon
```
