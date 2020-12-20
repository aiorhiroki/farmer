# farmer

You can train Classification and Segmentation tasks semi-automatically

## Prerequisite

### `install docker`
- Docker == 19.03

- Add USER to docker group for using docker command of USER's authority
```bash
# (Optinal) create docker group if there is nothing
sudo groupadd docker

# add USER to docker group
sudo gpasswd -a $USER docker

# reflect setting if re-login
exit
```

### `build docker`
```bash
docker build -t farmer:v2.0 .
#when it fails, try with --no-cache
```

## Run docker container
You need to be in `WORKDIR` of this repository

check the current path
```bash
echo $PWD
$ /PATH/TO/farmer
```

### Start container for farmer
```bash
# you can change a directory for mount if you need
docker run \
    --gpus all \
    -itd \
    -v /mnt:/mnt \
    --name farmer \
    farmer:v2.0
```

### Install farmer in container
```bash
bash install_farmer.sh
```

### (Optional) Check farmer's path which is used in container
```bash
# show farmer path history
docker exec -it farmer bash -c "cat ~/.farmerpath.csv"
```


## COMMAND list

#### **`~/.bash_aliases`**
```bash
dogrun () {
    docker exec -it -u $(id -u):$(id -g) farmer bash -c "cd $PWD && $1"
}

dogout () {
    nohup docker exec -t -u $(id -u):$(id -g) farmer bash -c "cd $PWD && Godfarmer" > $1 &
}

dogin () {
    docker exec -it -u $(id -u):$(id -g) farmer bash
}
```

```bash
source ~/.bashrc  # to activate bash aliases
```

#### **`~/.config/fish/config.fish`**
``` bash
function dogrun
    docker exec -it -u (id -u):(id -g) farmer bash -c "cd $PWD && $argv"
end

function dogout
    nohup docker exec -t -u (id -u):(id -g) farmer bash -c "cd $PWD && Godfarmer" > $argv &
end

function dogin
    docker exec -it -u (id -u):(id -g) farmer bash
end
```

```bash
source ~/.config/fish/config.fish  # to activate fish aliases
```

### Example
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

## add package
```
# cd PATH/TO/farmer/
docker exec -it farmer bash -c "cd $PWD && poetry add pandas"
```
