docker run \
	--gpus all \
	-itd \
	--rm \
	-p $2:22 \
	--name $1 \
	--mount type=bind,source=/mnt,target=/mnt \
	test:v1 \
	bash

docker exec -it --user $USER $1 bash --login
