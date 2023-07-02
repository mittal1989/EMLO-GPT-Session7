# Experiment Tracking

## Training ViT Module on CIFAR10 dataset (multirun using joblib)

## Model Training and Testing

1. Run "docker-compose build" to build train and logger images.
2. Run "docker-compose run train" to start training your ViT model on the CIFAR10 dataset for three different patch sizes 4, 8, and 16.
Note: Training inside docker will be a multirun using joblib. There are five experiments conducted based on three different patch sizes. Training code running inside docker is:
"copper_train -m hydra/launcher=joblib hydra.launcher.n_jobs=3 experiment=cifar10 model.patch_size=4,8,16 data.num_workers=0".
3. Run "docker-compose run --service-ports logger" to start mlflow logging ui. Open "localhost:5000" inside your browser to view mlflow logs.

## DVC Setup
- Install dvc using "pip install dvc".
- Initialized DVC using `dvc init`. (you will find a .dvc folder created for your project)
- Add data to DVC : `dvc add docker_data`.
- Add logs & model to DVC : `dvc add docker_logs`.

- To track the changes with git, run: `git add docker_data.dvc .gitignore`
- To enable auto staging, run: `dvc config core.autostage true`

You will find two files created: `docker_data.dvc` & `docker_logs.dvc`.

## Integrating local storage with DVC
- Add data & logs to local folder dvc-data: `dvc remote add -d local /workspace/dvc-data`
- Check if a local remote storage has been added by using this command: `dvc remote list` (it will give you list of all remote storage for your project)
- Push the data using : `dvc push -r local`
- Pull data from local storage : `dvc pull -r local`

Group Members:
Aman Jaipuria, Anurag Mittal