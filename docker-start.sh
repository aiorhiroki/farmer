docker run \
	--gpus all \
	-itd \
	--rm \
	-p $2:$2 \
	--name $1 \
	--mount type=bind,source="$PWD",target=/app \
	--mount type=bind,source=/home,target=/home \
	--mount type=bind,source=/media,target=/media \
	--mount type=bind,source=/mnt,target=/mnt \
	tensorflow:v2 \
	bash

docker exec -it $1 bash
