docker run \
	--gpus all \
	-itd \
	--rm \
	-p $2:$2 \
	--name $1-$USER \
	--mount type=bind,source=/mnt,target=/mnt \
	--mount type=bind,source=/home/$USER/src,target=/home/docker/src \
	tensorflow:v2 \
	bash

docker exec -it $1-$USER bash
