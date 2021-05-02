# Set up your own self-hosted runners for github actions



- Build docker container and check if it works with GPU
```
docker run --gpus all dvcorg/cml-py3 nvidia-smi
```

- Run container connected with Github Repo
```
docker run --name myrunner -d --gpus all \
    -e RUNNER_IDLE_TIMEOUT=1800 \
    -e RUNNER_LABELS=cml,gpu \
    -e RUNNER_REPO=$my_repo_url \
    -e repo_token=$my_repo_token \
    dvcorg/cml-py3
```


https://dvc.org/blog/cml-self-hosted-runners-on-demand-with-gpus
