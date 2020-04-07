sudo docker run \
	--gpus all \
	-itd \
	--rm \
	-p $2:22 \
	--name $1 \
	--mount type=bind,source=/data,target=/data \
	--mount type=bind,source=/mnt,target=/mnt \
        --mount type=bind,source=/home/$USER/src,target=/home/$USER/src \
        tensorflow:v2 \
	bash

sudo docker exec -it --user $USER $1 bash --login
