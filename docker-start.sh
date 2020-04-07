sudo docker run \
	--gpus all \
	-itd \
	--rm \
	-p $2:22 \
	--name $1 \
	--mount type=bind,source=/data,target=/data \
	tensorflow:v2 \
	bash

sudo docker exec -it --user $USER $1 bash --login
