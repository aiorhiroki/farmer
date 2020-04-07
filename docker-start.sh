docker run \
	--gpus all \
	-itd \
	--rm \
	-p $2:22 \
	--name $1 \
	--mount type=bind,source=/mnt,target=/mnt \
    --mount type=bind,source=/home/$USER/src,target=/home/$USER/src \
    tensorflow:v2
	bash

docker exec -it --user $USER $1 bash --login
