# Optional

## Setup via Podman Container Using Ubuntu Image

```bash
podman run -it --name Ubuntu -v path_of_local_machine:/home/ubuntu/folderName:z -p 8888:8888 ubuntu
```

```bash
apt update && apt install -y wget
```

```bash
wget collect-url-from-anaconda-download-page -O anaconda.sh
```

```bash
bash anaconda.sh -b -p /opt/anaconda
```

```bash
/opt/anaconda/bin/conda init
```

```bash
source ~/.bashrc
```

```bash
conda create -n my-env python=3.11 -y
conda activate my-env
conda env list
```

```bash
conda install jupyterlab
```

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```


### Requirements.txt without version 

```bash
pip freeze | python -c "for p in __import__('sys').stdin: print(p.split('=')[0])" > requirements.txt
```


### Update anaconda

```bash
conda update anaconda
```



### 
