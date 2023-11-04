import re
import os
import json
from transformers import pipeline
import torch
from tqdm import tqdm
import pandas as pd
from vllm import LLM, SamplingParams
import time
import fnmatch

import json

class Config(object):
    """
    A class for managing configuration settings.

    Attributes:
        my_dict (dict): A dictionary containing the configuration settings.

    Methods:
        save(path): Saves the configuration settings to a JSON file.
        load(path): Loads the configuration settings from a JSON file.
        has_key(key): Checks if a key exists in the configuration settings.
    """

    def __init__(self, my_dict):
        self.my_dict = my_dict
        for key in my_dict:
            setattr(self, key, my_dict[key])

    def save(self, path):
        """
        Saves the configuration settings to a JSON file.

        Args:
            path (str): The path to the JSON file.
        """
        with open(path, 'w') as f:
            json.dump(self.my_dict, f)

    def load(path):
        """
        Loads the configuration settings from a JSON file.

        Args:
            path (str): The path to the JSON file.

        Returns:
            Config: A Config object containing the loaded configuration settings.
        """
        with open(path, 'r') as f:
            my_dict = json.load(f)
        return Config(my_dict=my_dict)
    
    def has_key(self,key):
        """
        Checks if a key exists in the configuration settings.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return key in self.my_dict
    
    def __str__(self) -> str:
        return json.dumps(self.my_dict, indent=4, sort_keys=True)

def find_function_names(code):
    """
    Finds all function names in the given code.

    Args:
        code (str): The code to search for function names.

    Returns:
        list: A list of all function names found in the code.
    """
    pattern = r"edi\.\w+:(\w+)"
    matches = re.findall(pattern, code)
    return matches

def find_code_blocks(text):
    """
    Finds all code blocks in the given text that are enclosed in `lua` tags.

    Args:
        text (str): The text to search for code blocks.

    Returns:
        list: A list of strings, where each string is a code block found in the text.
    """
    pattern = r"`lua\n(.*?)\n`"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches
    
def find_function_names_in_code_blocks(text):
    """
    Finds all function names in the given text by searching through all code blocks.

    Args:
        text (str): The text to search for function names.

    Returns:
        list: A list of all function names found in the text.
    """
    blocks = find_code_blocks(text)
    names = []
    for block in blocks:
        names += find_function_names(block)
    return names

def get_json_files_from_directory(directory):
    """
    Returns a list of all JSON files in the specified directory and its subdirectories.

    Args:
        directory (str): The directory to search for JSON files.

    Returns:
        list: A list of all JSON files in the specified directory and its subdirectories.
    """
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def flatten(l):
    """
    Flattens a list of lists into a single list.

    Args:
        l (list): A list of lists.

    Returns:
        list: A flattened list.
    """
    return [item for sublist in l for item in sublist]


def find_all_apis(json_directory):
    """
    Finds all APIs in the JSON files located in the specified directory.

    Args:
        json_directory (str): The path to the directory containing the JSON files.

    Returns:
        list: A list of dictionaries, where each dictionary represents an API and contains the following keys:
            - 'function_name': The name of the API function.
            - 'description': A brief description of what the API does.
            - 'parameters': A list of dictionaries, where each dictionary represents a parameter of the API and contains the following keys:
                - 'name': The name of the parameter.
                - 'type': The data type of the parameter.
                - 'description': A brief description of what the parameter is used for.
            - 'returns': A dictionary representing the return value of the API, containing the following keys:
                - 'type': The data type of the return value.
                - 'description': A brief description of what the return value represents.
            - 'example': An example usage of the API.
    """
    results = []
    files_path = get_json_files_from_directory(json_directory)
    for file_path in files_path:
        with open(file_path, 'rb') as f:
            data = json.load(f)
            results.extend(data[list(data.keys())[0]])
    additional_functions = []
    for result in results:
        if result['example']:
            functions = find_function_names(result['example'])
            for function in functions:
                additional_functions.append({'function_name': function,'in_example':result['example']})
    results.extend(additional_functions)
    return results


def match_helper(function_name, all_apis):
    """
    Searches for a function name in a list of APIs and returns a dictionary with the function name and its match (if found).

    Args:
        function_name (str): The name of the function to search for.
        all_apis (list): A list of dictionaries containing information about all the available APIs.

    Returns:
        dict: A dictionary with the function name and its match (if found).
    """
    for api in all_apis:
        if function_name == api['function_name']:
            return {'target': function_name, 'match': api}
    return {'target': function_name, 'match': None}
def match_helper(function_name,all_apis):
    for api in all_apis:
        if function_name == api['function_name']:
            return {'target':function_name,'match':api}
    return {'target':function_name,'match':None}


def find_match_unmatch(function_names, all_apis):
    """
    Finds the matching and non-matching APIs for a given list of function names.

    Args:
        function_names (list): A list of function names to match against the APIs.
        all_apis (list): A list of all available APIs.

    Returns:
        tuple: A tuple containing three lists:
            - The first list contains the matching APIs.
            - The second list contains the non-matching APIs.
            - The third list contains all APIs, with the matching APIs listed first.
    """
    match = []
    unmatch = []
    for function_name in function_names:
        result = match_helper(function_name, all_apis)
        if result['match']:
            match.append(result)
        else:
            unmatch.append(result)
    return match, unmatch, unmatch + match
    
    



def generate_output(model, tokenizer, config, prompts):
    """
    Generates output sequences for a given list of prompts using a pre-trained text generation model.

    Args:
        model (transformers.PreTrainedModel): The pre-trained text generation model to use.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for encoding the prompts.
        config (Config): A configuration object containing various hyperparameters for the generation process.
        prompts (List[str]): A list of prompts to generate output sequences for.

    Returns:
        List[List[int]]: A list of generated output sequences, where each sequence is represented as a list of token IDs.
    """
    # Encode a list of prompts
    generated_sequences = []
    prompt_template = "<|system|>\n<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>"
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda:0")
    for index in tqdm(range(0, len(prompts))):
        prompt = prompt_template.format(query=prompts[index])
        generated_sequences += pipe(prompt, 
                                    max_new_tokens=config.max_length, 
                                    do_sample=True, 
                                    temperature=config.temperature, 
                                    top_k=config.top_k, 
                                    top_p=config.top_p, 
                                    pad_token_id=tokenizer.eos_token_id,
                                    eos_token_id=49155)
    return generated_sequences

def generate_output_with_batch(model, tokenizer, config, prompts, batch_size):
    """
    Generates output for a list of prompts using the given model and tokenizer in batches.

    Args:
        model (torch.nn.Module): The model to use for generating output.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for encoding the prompts.
        config (Config): The configuration object containing the generation parameters.
        prompts (List[str]): The list of prompts to generate output for.
        batch_size (int): The batch size to use for generating output.

    Returns:
        List[str]: The list of generated outputs for the given prompts.
    """
    # Encode a list of prompts
    generated_sequences = []
    prompt_template = "<|system|>\n<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>"
    for index in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[index:index + batch_size]
        batch_prompts = [prompt_template.format(query=prompt) for prompt in batch]
        batch_input = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to("cuda")
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
            # Generate sequences
            batch_output = model.generate(
                **batch_input,
                max_length=config.max_length,
                do_sample=True,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=49155,
                num_return_sequences=len(batch),
            )
        batch_output = tokenizer.batch_decode(batch_output)
        generated_sequences += batch_output
    return generated_sequences



def eval_output(df, config, all_apis, is_md=True):
    """
    Evaluates the generated output and returns a DataFrame with the results.

    Args:
        df (pandas.DataFrame): DataFrame containing the generated output.
        config (dict): Configuration dictionary.
        all_apis (list): List of all available APIs.
        is_md (bool, optional): Whether the generated output is in Markdown format. Defaults to True.

    Returns:
        pandas.DataFrame: DataFrame containing the evaluation results.
    """
    generated_sequences = df['generated'].tolist()
    results = []
    for output in tqdm(generated_sequences):
        if is_md:
            code_blocks = find_code_blocks(output)
        else:
            code_blocks = [output]
        function_names = flatten([find_function_names(code_block) for code_block in code_blocks])
        match,unmatch,all_result = find_match_unmatch(function_names,all_apis)

        # notation
        if len(match) == 0 and len(unmatch) == 0:
            notation = 'No function found'
        elif len(match) == 0 and len(unmatch) > 0:
            notation = 'No function matched: ' +" ".join([item['target'] for item in unmatch])
        elif len(match) > 0 and len(unmatch) == 0:
            notation = 'All functions matched: ' +" ".join([item['target'] for item in match])
        else:
            notation = 'Some functions matched: ' +" ".join([item['target'] for item in match]) + '  \nSome functions not matched: ' + " ".join([item['target'] for item in unmatch])
        results.append([all_result,match,unmatch,len(match),len(unmatch),notation])

    result_df = pd.DataFrame(results, columns=['all_result','match','unmatch','match_count','unmatch_count','note'])
    df = pd.concat([df,result_df],axis=1)
    return df




def generate_report(df, config, duration, is_md=True):
    """
    Generates a report based on the given DataFrame and configuration.

    Args:
        df (pandas.DataFrame): The DataFrame containing the test results.
        config (object): The configuration object used for the test.
        duration (float): The duration of the test in seconds.
        is_md (bool, optional): Whether to generate the report in Markdown format. Defaults to True.

    Returns:
        None
    """
    # function code here

    # 测试时间
    import datetime
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d %H:%M:%S")

    # 提取相关数据
    total_tests = len(df)
    matched = df['match_count'].sum()
    unmatched = df['unmatch_count'].sum()
    total = matched + unmatched
    match_rate_avg = matched/total

    #平均每条的api数量
    df['api_count'] = df['match_count'] + df['unmatch_count']
    api_count_avg = df['api_count'].mean()

    #生成的不含api的数量
    zero_api_count = df[df['api_count']==0].shape[0]
    zero_api_count_rate = zero_api_count/total_tests

    #生成的完整包含api的数量
    full_api_count = df[(df['match_count'] == df['api_count']) & (df['api_count'] > 0)].shape[0]
    full_api_count_rate = full_api_count/total_tests

    #部分包含api的数量
    partical_api_count = total_tests - zero_api_count - full_api_count
    partical_api_count_rate = partical_api_count/total_tests

    # unmatch function name
    unmatch_function_names = []
    for index, row in df.iterrows():
        if row['unmatch']:
            unmatch_function_names.extend([item['target'] for item in row['unmatch']])

    # 生成Markdown格式的测试报告
    report = f"""
# 测试报告

## 测试数据:


- **测试模型**: {config.model_path}
- **测试参数**:

```json
{config}
```

- **测试文件**: {config.test_data_path}
- **总测试次数**: {total_tests}
- **测试时间**: {now}
- **测试时长**: {duration}


## 测试结果数据:


- **匹配数总和**: {total}
- **不匹配数总和**: {unmatched}
- **总匹配率平均**: {match_rate_avg:.2%}
- **平均每条的api数量**: {api_count_avg:.2f}
- **生成的不含api的数量**: {zero_api_count}
- **生成的不含api的数量百分比**: {zero_api_count_rate:.2%}
- **生成的完整包含api的数量**: {full_api_count}
- **生成的完整包含api的数量百分比**: {full_api_count_rate:.2%}
- **部分包含api的数量**: {partical_api_count}
- **部分包含api的数量百分比**: {partical_api_count_rate:.2%}




## 备注:
    
 ### 未检测到的function:

"""

    for unmatch_function in unmatch_function_names:
        report += f"\n{unmatch_function}"

    report += "\n"
    for index, row in df.iterrows():
        if is_md:
            report += f"\n ---  \n  **Prompt: {row['prompt']}** \n\n **Output:**  \n {row['generated']}\n\n  **Note:**  {row['note']}\n\n"
        else:
            report += f"\n ---  \n  **Prompt: {row['prompt']}** \n\n **Output:**  \n ``` {row['generated']}\n```\n\n  **Note:**  {row['note']}\n\n"
    # 保存为 report.md
    with open(config.report_output_path, 'w+', encoding='utf-8') as file:
        file.write(report)

    print("Report saved to ",config.report_output_path)


def eval(config):
    """
    Evaluates a language model on a given test dataset and generates output based on the model's predictions.

    Args:
        config (Config): A configuration object containing the necessary parameters for evaluation.

    Returns:
        None
    """
    start_time = time.time()
    # load model
    if config.stop_word:
        sampling_params = SamplingParams(temperature=config.temperature, top_p=config.top_p, top_k=config.top_k, max_tokens=config.max_tokens,stop=[config.stop_word])
    else:
        sampling_params = SamplingParams(temperature=config.temperature, top_p=config.top_p, top_k=config.top_k, max_tokens=config.max_tokens)

    llm = LLM(model=config.model_path,gpu_memory_utilization=config.gpu_memory_utilization)

    # load test file
    df = pd.read_csv(config.test_data_path)

    print(f"Loaded {len(df)} prompts from {config.test_data_path}")
    # generate output
    prompts = df['prompt'].tolist()

    # edit prompt according to template
    prompt_template = config.prompt_template
    prompts = [prompt_template.format(query=prompt) for prompt in prompts]


    print(f"Generating output for {len(prompts)} prompts...")

    outputs = llm.generate(prompts, sampling_params,use_tqdm=True)
    generated_sequences = [output.outputs[0].text for output in outputs]
    # save output
    df['generated'] = generated_sequences
    print(f"Saving output to {config.output_path}...")
    df.to_csv(config.output_path,index=False)

    # get all apis
    print("Loading all apis...")
    all_apis = find_all_apis(config.api_path)

    # match function name in generated output
    print("Matching function names in generated output...")
    df = eval_output(df,config,all_apis,is_md=False)
    print(f"Saving output to {config.output_path}...")
    df.to_csv(config.output_path,index=False)


    # Stop the timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    # Convert elapsed time into minutes and seconds
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    # Print in the desired format
    duration_str = f'{minutes}:{seconds:02}'
    # generate report
    print("Generating report...")
    if config.report_type == 'markdown':
        generate_report(df,config,duration_str,is_md=True)
    else:
        generate_report(df,config,duration_str,is_md=False)

def get_matching_folders(path, pattern="checkpoint-*"):
    """
    Get all folder numbers in the specified path that match the given pattern.
    
    Args:
    - path (str): The directory path to search in.
    - pattern (str): The pattern to match folder names against.
    
    Returns:
    - List[int]: A list of numbers extracted from matching folder names.
    """
    all_folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    matching_folders = fnmatch.filter(all_folders, pattern)
    
    # Extract numbers from the matching folder names
    return matching_folders

def get_matching_files(path,pattern="test_result-*"):
    """
    Get all folder numbers in the specified path that match the given pattern.
    
    Args:
    - path (str): The directory path to search in.
    - pattern (str): The pattern to match folder names against.
    
    Returns:
    - List[int]: A list of numbers extracted from matching folder names.
    """
    all_files = [d for d in os.listdir(path) if os.path.isfile(os.path.join(path, d))]
    matching_files = fnmatch.filter(all_files, pattern)
    
    # Extract numbers from the matching folder names
    return matching_files