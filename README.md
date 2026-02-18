### Install
```
bash setup.sh
```

### AutoTune AUROC on PneumoniaMNIST

```
python -m src.ui.run --config_file ./recipes/automax/config_auc_PneumoniaMNIST.yaml
```

### Run single experiment (without autotune)

```
python -m src.ui.run_trainer --config_file ./recipes/libauc_trainer/config_auc.yaml
```
