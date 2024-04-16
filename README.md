# TianmoucAlg

**NEWS!** The USB module of TianMouC is released.

**NEWS!** The Python library of TianMouC sensor is available in [tianmoucv/Tianmoucv_preview](https://github.com/Tianmouc/Tianmoucv_preview)

The official version will be available at [tianmoucv/tianmocv](https://github.com/Tianmouc/tianmoucv)

## Arch of this project

```
├── data -> .../tianmouc_evaluation_data/
│   ├── ckpts
│   ├── demo_data
│   └── recon_data
├── datareader
├── demo
├── reconstruction
├── resources
└── tianmoucv
```


- reconstruction : The reconstruction algorithm for TianMouC raw data
- demo         : the automotive driving perception algorithm
- tianmoucv  : some basic algorithm for TianMouC raw data
- datareader : the raw data decoder and data reader

- tianmouc_evaluation_data : Part of the labeled data for demo evaluation
  - ckpts  : 3 Pytorch models used in this paper
  - demo_data : 5 samples used in Fig 4
  - recon_data : some clips for reconstruction


## requirement

```bash
git clone  
conda create -n tianmouc python=3.10
sh install.sh
```
## prepare Dataset

download the dataset in [zenodo](https://doi.org/10.5281/zenodo.10602822) and decompress it.

use the soft link to create easy data access for this repo:

```bash
cd [N_pub_code]
ln -s [your dataset path] data
```

## EASY START

The code is replicated using python with jupyter notebook

- Fig. 4 with mAP evaluation
```
  /code/demo/Evaluation_complex.ipynb
```
  ![fig4e](./resources/Evaluation_complex.png)
  
  ![fig4e](./resources/Evaluation_flash.png)
  
  *the OF data need to be calculated using raw data*
  
  ![fig4e](./resources/Evaluation_OF.png)
  
You can change the  **key(name) of dataset** to find more demos for Fig4 or automotive driving 
The labeled datasets are in /data/tianmouc_evaluation_data
  
- anti-aliasing reconstruction
```
  /code/reconstruction/reconstruction.ipynb
```
  ![fig4e](./resources/Reconstruction.png)
  


