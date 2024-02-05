# TianmoucAlg

## Arch of this project

├── code

│      └── reconstruction : The reconstruction algrithm for tianmouc raw data

│      └── demo       : the automotive driving perception algorithm

│      └── tianmoucv  : some basic algorithm for tianmouc raw data

│      └── datareader : the raw data decoder and data reader

│      └── sample     : some simmple demos

├── data

│      ├── tianmouc_raw_data : some samples for testing the reconstruction NN

│      │  

│      ├── tianmouc_evaluation_data : Part of the labeled data for demo evaluation

│      │  

│      └── video_resource: the recorded and visualized data, can be played directly



## EASY START

you can only directly run following notebook on code ocean for quick experience:

- Fig. 4e with mAP evaluation
```
  /code/demo/Evaluation_complex.ipynb
```
  result in: /results/demo_complex.avi
  ![fig4e](./resources/Evaluation_complex.png)

- Fig. 4b with mAP evaluation
```
  /code/demo/Evaluation_flash.ipynb
```
  result in: /results/demo_flash.avi
  ![fig4e](./resources/Evaluation_flash.png)
  
- Fig. 4d
```
  /code/demo/realDataOF_only.ipynb
```
  *since the raw data is too large, you may only see the result we have generated offline*
  ![fig4e](./resources/Evaluation_OF.png)
  
And you can change the   **key(name) of dataset** to find more demo for Fig4 or automotive driving 
The labeled datasets are in /data/tianmouc_evaluation_data
  
- anti-aliasing reconstruction
```
  /code/reconstruction/reconstruction.ipynb
```
  result in: /results/demo_complex.avi
  ![fig4e](./resources/Reconstruction.png)
  
- some simple demo
```
  - /code/examples/Visualization.ipynb
  - /code/examples/dualMatch.ipynb
  - /code/examples/Basic_Poisson_Blend_Gray_reconstruction.ipynb
  - /code/examples/Gray_reconstruction_on_QRcode.ipynb
```
  

## requirement (no need on code ocean)

```bash
git clone https://git.codeocean.com/capsule-5450564.git
conda create -n tianmouc python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install tqdm scikit-image imageio 
conda install imageio
conda install tqdm
pip install -r requirements.txt
cd ./datareader/tools/rod_decode_pybind
sh compile_pybind.sh
```

## DMEO

The demo folder displays the Python version and testing methods of the algorithm used in the paper. These algorithms include the yolopv1 detector (running on COP), yolov5 detector (running on AOP), and a dense optical flow calculator ()

To run this demo, you can click on

```
./demo/Evaluation_xxx.ipynb
```

and run interactively. 

## Reconstruction


The reconstruction algorithm is encapsulated in /code/reconstruction, it is an offline reconstruction network. You can use

```
./reconstruction.ipynb
```
Direct operation, or

```
cd /Reconstruction/
Python dumpAll.py
```

Export all reconstruction result fragments (you can adjust the reconstruction length in this scripts)

## Examples

More examples of data uasage are presented in/ Using notebook under sample


