# TianmoucAlg

## Arch of this project

├── code

│      └── reconstruction : The reconstruction algrithm for tianmouc raw data

│      └── demo       : the automotive driving perception algorithm

│      └── tianmoucv  : some basic algorithm for tianmouc raw data

│      └── datareader : the raw data decoder and data reader


├── data

│      ├── tianmouc_evaluation_data : Part of the labeled data for demo evaluation

│          └── ckpts  : 3 pytorch models used in this paper
  
│          └── demo_data : 5 samples used in Fig 4

│          └── recon_data : some clips for reconstruction


## EASY START

you can only directly run following notebook on code ocean for quick experience:

- Fig. 4 with mAP evaluation
```
  /code/demo/Evaluation_complex.ipynb
```
  ![fig4e](./resources/Evaluation_complex.png)
  ![fig4e](./resources/Evaluation_flash.png)
  *the OF data need to be calculated using raw data*
  ![fig4e](./resources/Evaluation_OF.png)
  
You can change the  **key(name) of dataset** to find more demo for Fig4 or automotive driving 
The labeled datasets are in /data/tianmouc_evaluation_data
  
- anti-aliasing reconstruction
```
  /code/reconstruction/reconstruction.ipynb
```
  ![fig4e](./resources/Reconstruction.png)
  

## requirement

```bash
git clone  
conda create -n tianmouc python=3.10
sh install.sh
```


