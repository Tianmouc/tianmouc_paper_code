conda install pytorch torchvision torchaudio pytorch-cuda=12.0 -c pytorch -c nvidia
python -m pip install -r requirements.txt
cd ./datareader/tools/rod_decode_pybind
sh compile_pybind.sh
cd ../../../