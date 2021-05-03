docker exec -it $1 bash -c \
    "cd $PWD && \
    poetry run python setup.py develop && \
    echo $PWD, $(date), $(id -u -n) >> ~/.farmerpath.csv"
