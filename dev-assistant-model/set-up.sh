# set up pip 
vim $HOME/.config/pip/pip.conf
# add these files
# [global]
# index-url = https://pypi.tuna.tsinghua.edu.cn/simple
# cache-dir=/home/uework/AiWeb/pwz/github/env/cache



# set deploy virtual environment
conda create -n deploy python=3.10 -y && conda activate deploy
pip install vllm
pip install -r ../luacoder-endpoint-server/requirements.txt


# set deepspeed train virtual environment
conda create -n train python=3.10 -y && conda activate train
pip install torch==2.0.1 torchvision torchaudio
pip install -r ./deepspeed/chat/requirements.txt