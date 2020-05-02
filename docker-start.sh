sudo docker run \
	--gpus all \
	-itd \
	--rm \
	-p $2:22 \
	--name $1 \
	--mount type=bind,source=/mnt/hdd2,target=/mnt/hdd2 \
        tensorflow:v2 \
	bash

sudo docker exec -it --user $USER $1 bash --login
