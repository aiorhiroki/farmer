docker run \
	--gpus all \
	-it \
	--rm \
	-p 4000:4000 \
	--mount type=bind,source="$PWD",target=/app \
	--mount type=bind,source=/home,target=/home \
	--mount type=bind,source=/media,target=/media \
	--mount type=bind,source=/mnt,target=/mnt \
	tensorflow:v2 \
	bash
