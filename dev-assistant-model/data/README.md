# Data processing

All data process function is defined in utils.py, you can reach the folder to use functions

## File Structure

```
data/
    json.ipynb          # json file processing playground
    lua.ipynb           # lua file processing playground
    utils.py            # defines all data processing functions
    dialogue.ipynb      # dialogue processing playground
    requirements.txt    # pip requirements
    README.md           # this file
    scripts/            # data processing basic piplines
        ......
```


## Usage

You can use the functions in utils.py to achieve your goal. Here are some examples to create datasets, processing data.

### using lua files to create dataset

```

from utils import *
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from datasets import DatasetDict,Dataset
from scripts.preprocessing import clean_dataset
#################################################################
#### generate dataset for luacoder

dataset_folder = '/home/uework/AiWeb/pwz/github/data/dataset'
dataset_name = 'test-lua'
model_path = '/home/uework/AiWeb/pwz/github/model/starcoderbase-3b'
files_directory = ['/home/uework/AiWeb/pwz/github/data/lua/api-doc-lua',
                   '/home/uework/AiWeb/pwz/github/data/lua/shengqu',
                   '/home/uework/AiWeb/pwz/github/data/lua/dialogue'
                   ]
generate_luacoder_dataset(dataset_folder,dataset_name,model_path,files_directory)


```




### search substring in lots lua files


```
from utils import search_files_for_substring
directory = '/home/uework/AiWeb/pwz/github/data/lua/shengqu'
substring = """g_mir.common.activityMgr"""

matches = search_files_for_substring(directory, substring)
for filename, lines in matches:
    print(f"Found in {filename} on lines: {', '.join(map(str, lines))}")

```


### find templates structure code in lots lua files

template
```
local function _refreshtab(player, _, param)--把调用函数的param写在这里
    xxxx --其他填写代码
    refresh_ssr_cmd(player,module.ssrID,param) --执行的函数，传入param
    xxxx --其他填写代码
end
GameEvent.register_ssr(refreshFuncName, _refreshtab) --注册
```

search code
```

from utils import *
# Get all Lua files in the directory
directory_path = '/home/uework/AiWeb/pwz/github/data/lua/shengqu'  # Replace with your directory path
results = find_all_samples(directory_path,mute=True)
# print(json.dumps(results, indent=4))
samples_to_lua_file(results,'output.lua')

```


### transform markdown api document to json format

```
from utils import *
file_path = '/home/uework/AiWeb/pwz/github/data/md/api-doc/06.玩家.md'
json_output = md_to_json(file_path)
print(json.dumps(json_output, indent=4, ensure_ascii=False))

```


### transform oppenassitent dataset to json format

```
from utils import dialogue_to_json
from tqdm import tqdm
from datasets import DatasetDict
from datasets import load_from_disk
openassistant = load_from_disk('/home/uework/AiWeb/pwz/github/data/dataset/openassistant')
train_json_list = []
# train data
for row in tqdm(openassistant['train']['text']):
    # Convert to JSON format
    json_data = dialogue_to_json(row)
    # Append to list
    train_json_list.append(json_data['messages'])
test_json_list = []
# test data
for row in tqdm(openassistant['test']['text']):
    # Convert to JSON format
    json_data = dialogue_to_json(row)
    # Append to list
    test_json_list.append(json_data['messages'])
# create dataset
train_dataset = openassistant['train'].add_column('messages', train_json_list)
test_dataset = openassistant['test'].add_column('messages', test_json_list)
# save dataset
openassistant_json = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})
openassistant_json.save_to_disk('/home/uework/AiWeb/pwz/github/data/dataset/openassistant_json')
openassistant_json

```


