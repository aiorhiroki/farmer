docker run \
	--gpus all \
	-it \
	--rm \
	-p 5000:5000 \
	--mount type=bind,source="$PWD",target=/app \
	--mount type=bind,source="$HOME",target="$HOME" \
	--mount type=bind,source=/media,target=/media \
	--mount type=bind,source=/mnt,target=/mnt \
	deep:v2 \
	bash
