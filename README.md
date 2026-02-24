### Install
```
conda create -n AutoMAX python=3.10
conda activate AutoMAX
conda install gxx_linux-64 gcc_linux-64 swig
pip install -r requirements.txt
```


### AutoTune AUROC on PneumoniaMNIST

```
python -m src.ui.auto_trainer --config_file ./recipes/automax/config_auc_PneumoniaMNIST.yaml
```

### AutoTune AUROC on RIP-Dataset

```
python -m src.ui.auto_transformers_trainer --config_file ./recipes/automax/config_auc_RIP.yaml
```

### Run single experiment (without autotune)

```
python -m src.ui.run_trainer --config_file ./recipes/libauc_trainer/config_auc.yaml
```
