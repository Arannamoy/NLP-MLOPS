```txt
!python3 -m venv venv
python -m ipykernel install \
  --user \
  --name conda-env-gnn \
  --display-name "Python (conda-gnn)"
```

```txt
!pip uninstall -y torch torchvision torchaudio torch-scatter torch-cluster
```

```txt
!pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


```txt
!pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

```txt
!pip install laspy tqdm gdown scikit-learn mlflow open3d
```

```txt
!gdown --folder "https://drive.google.com/drive/folders/1fUCLmHNBI5gCP4w1Z-qA49Lkq4V_8xhC?usp=sharing"
```



# Computational Time:
```json
"num_classes"    : 2,                 # 0 = environment, 1 = wood powder
    "target_class"   : 1,                 # class we want to highlight (green)
    "val_ratio"      : 0.15,
    "test_ratio"     : 0.15,
    "seed"           : 42,
    # ── training ──
    "epochs"         : 100,
    "patience"       : 100,               # early stop if val mIoU doesn't improve
    "batch_size"     : 2,
    "num_points"     : 163840,             # points per training chunk
    "chunks_per_cloud": 10,
    "lr"             : 1e-3,
    "grad_accum"     : 4,    # gradient accumulation steps (effective batch = batch_size × grad_accum)
    "in_channels"    : 7, 
```
`5 min per epoch-4090`

```json
  "num_classes"    : 2,                 # 0 = environment, 1 = wood powder
    "target_class"   : 1,                 # class we want to highlight (green)
    "val_ratio"      : 0.15,
    "test_ratio"     : 0.15,
    "seed"           : 42,
    # ── training ──
    "epochs"         : 100,
    "patience"       : 100,               # early stop if val mIoU doesn't improve
    "batch_size"     : 4,
    "num_points"     : 89120,             # points per training chunk
    "chunks_per_cloud": 10,
    "lr"             : 2e-3,
    "grad_accum"     : 2,    # gradient accumulation steps (effective batch = batch_size × grad_accum)
    "in_channels"    : 7,   
```
`2 min per epoch-4090`
`7 min per epoch-3060`
