# DevAssistant VSCode Endpoint Server

DevAssistant server for devassistant-vscode endpoint.

**Can't handle distributed inference very well yet.**


## Version

**devassistant-endpoint beta 0.1.0**
- auto deploy scripts for pulling model
- data collection enable

**devassistant-endpoint beta 0.0.3**
- using 3B model for text generation 
- model version: devassistant-3B-beta2-1290


**devassistant-endpoint beta 0.0.2**

- enable autocomplete api preview
- disable chat api
- configure general model path
- model version:  devassistant-beta2-checkpoint-216

**devassistant-endpoint beta 0.0.1**

- chat api enable
- using vllm for fast output
- limit gpu memory usage to 25%, which is 10GB


## File Structure

```
devassistant-endpoint-server/
    doc/                    # api document
    models/                 # store the model 
    src/
        chat_server.py      # chat api server start point
        code_server.py      # code generation api server start point 
        generators.py       # define generating model
        gradio_interface.py # gradio interface for test the api
        sql.py              # data collection store in MySQL server
        tests.py            # unit tests
        util.py             # common tools
    scripts/
        create.sql          # sql scripts to create table in MySQL server
    .gitignore              # git ignore some file
    LICENSE
    README.md               # this file
    requirements.txt        # pip requirement
    deploy.sh               # one stop deploy server
```



## Usage


### developing
```shell
# install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

# need close ternimal


# set vllm and virtual environment
conda create -n deploy python=3.10 -y
conda activate deploy
pip install vllm
pip install -r requirements.txt

# run server
chmod +x deploy.sh
./deploy.sh
# select the server you want to deploy
```

Fill `http://<ip>:<port>/api/xxx/` into `DevAssistant > xxxEndpoint` in VSCode.

### deploy

```
# make sure python=3.10, cuda=11.7
# deploy scripts
chmod +x deploy.sh
./deploy.sh
```

## API Test

```shell
curl -X POST http://<ip>:<port>/api/generate/ -d '{"inputs": "How to detect if an object is monster?", "parameters": {"max_new_tokens": 128}}'
# response = {"generated_text": ""}
```
#### Or run in gradio

```
python src/gradio_interface.py
```