# farmer

You can train Classification and Segmentation tasks semi-automatically

## Prerequisite

### `install docker`
- Docker >= 19.03

### `build docker`
```bash
docker build -t farmer:v1.4 .
```

### `register env & command`

```bash
# for bash user

# write farmer path in .bashrc
echo "export FARMERPATH=$PWD" >> ~/.bashrc

# for fish user
echo "set -x FARMERPATH $PWD" >> ~/.config/fish/config.fish
```

#### **`~/.bash_aliases`**
```bash
dogrun () {
    docker exec -it -u (id -u):(id -g) farmer bash -c "cd $PWD && $1"
}

dogout () {
    nohup docker exec -u (id -u):(id -g) farmer bash -c "cd $PWD && Godfarmer" > $1 &
}

dogin () {
    docker exec -it -u (id -u):(id -g) farmer bash
}
```

#### **`~/.config/fish/config.fish`**
``` bash
function dogrun
    docker exec -it -u (id -u):(id -g) farmer bash -c "cd $PWD && $argv"
end

function dogout
    nohup docker exec -u (id -u):(id -g) farmer bash -c "cd $PWD && Godfarmer" > $argv &
end

function dogin
    docker exec -it -u (id -u):(id -g) farmer bash
end
```

```
source ~/.bashrc  # to activate bash aliases
source ~/.config/fish/config.fish  # to activate fish aliases
```

## Run docker container
```bash
docker run \
    --gpus all \
    -itd \
    -v $FARMERPATH:/app \
    -v /mnt/hdd2:/mnt/hdd2 \
    --name farmer \
    farmer:v1.4
```

```bash
docker exec farmer bash -c "poetry run python setup.py develop"
```


## COMMAND list
```bash
dogout log.out  # run farmer in the background
```

```bash
dogrun COMMAND  # run command in interactive docker
$ dogrun Godfarmer
$ dogrun python
```

```bash
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
dogrun Godfarmer
```
