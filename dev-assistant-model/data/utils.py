import os
from bs4 import BeautifulSoup
import json
from marko.ext.gfm import gfm
import pandas as pd
from tqdm import tqdm
from datasets import DatasetDict,Dataset,load_from_disk
from scripts.preprocessing import clean_dataset
import re

###############################################################
### General functions
def get_files_from_directory(directory,file_type):
    """
    Returns a list of all files with the specified file type in the given directory and its subdirectories.

    Args:
        directory (str): The directory to search for files.
        file_type (str): The file extension to search for (e.g. '.lua').

    Returns:
        A list of file paths (str) for all files with the specified file type in the given directory and its subdirectories.
    """
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_type):
                json_files.append(os.path.join(root, file))
    return json_files



import re

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

###############################################################
### lua api document processing function

# Helper function to extract args, ret, and example
def extract_info(h3_tag):
    """
    Extracts information from a given h3 tag and returns a dictionary containing the extracted information.

    Args:
        h3_tag (bs4.element.Tag): The h3 tag to extract information from.

    Returns:
        dict: A dictionary containing the extracted information. The dictionary may contain the following keys:
            - 'args': A list of dictionaries containing information about the function's arguments. Each dictionary may contain the following keys:
                - 'name': The name of the argument.
                - 'type': The type of the argument.
                - 'comment': A comment about the argument.
            - 'return': A dictionary containing information about the function's return value. The dictionary may contain the following keys:
                - 'type': The type of the return value.
                - 'comment': A comment about the return value.
            - 'example': A string containing an example of how to use the function.
    """
    result = {}
    
    next_element = h3_tag.find_next_sibling()
    if next_element and next_element.name == 'ul':
        # Extracting args
        args_list = []
        args_table = next_element.find_next_sibling('table')
        if args_table:
            for row in args_table.tbody.find_all('tr'):
                columns = [col.get_text(strip=True) for col in row.find_all('td')]
                if len(columns) == 3 and columns[0] != '-':
                    arg = {'name': columns[0], 'type': columns[1], 'comment': columns[2]}
                    args_list.append(arg)
            result['args'] = args_list
            next_element = args_table.find_next_sibling('ul')

        # Extracting return value and example
        ret_tag = next_element.find('li')
        if ret_tag and 'ret:' in ret_tag.text:
            code_tag = ret_tag.find('code')
            if code_tag:
                result['return'] = {"type":code_tag.text.strip(),"comment":ret_tag.contents[2].strip()}

        ret_tag = next_element.find('p')
        if ret_tag and 'ret:' in ret_tag.text:
            code_tag = ret_tag.find('code')
            if code_tag:
                result['return'] = {"type":code_tag.text.strip(),"comment":ret_tag.contents[2].strip()}

        

    # Extracting  example
    next_h3 = h3_tag.find_next_sibling('h3')
    example_code = None
    for sibling in h3_tag.find_all_next():
        if sibling == next_h3:
            break
        if sibling.name == 'pre':
            example_code = sibling
            break

    if example_code and example_code.code:
        result['example'] = example_code.code.get_text(strip=True)
    else:
        result['example'] = None

    return result



def md_to_json(file_path):
    """
    Converts a markdown file to a JSON object containing information about events.

    Args:
        file_path (str): The path to the markdown file.

    Returns:
        dict: A dictionary containing information about events, with the following keys:
            - 'events': A list of dictionaries, where each dictionary contains information about a single event.
                Each event dictionary has the following keys:
                    - 'description': A string describing the event.
                    - 'function_name': The name of the function associated with the event.
                    - Additional keys with information about the event, such as 'parameters' and 'returns'.
    """
    with open(file_path, 'r') as f:
        md_file = f.read()

    html_content = gfm(md_file)

    soup = BeautifulSoup(html_content, 'html.parser')

    events = []

    for h3 in soup.find_all('h3'):
        event_dict = {}
        
        # Getting the event's description and code name
        event_dict['description'] = h3.contents[0].strip()
        event_dict['function_name'] = h3.code.text.strip()
        event_dict.update(extract_info(h3))
        
        events.append(event_dict)

    json_output = {'events': events}
    return json_output


# create lua code for common package
def common_package(data):
    """
    Generates Lua test code for a given package based on the provided data.

    Args:
        data (dict): A dictionary containing information about the package's functions and their parameters.

    Returns:
        str: A string containing the generated Lua test code.
    """
    package_name = list(data.keys())[0]
    file_content = ""
    for event in data[package_name]:
        # print json nicely
        # print(json.dumps(event, indent=4, ensure_ascii=False))

        # print example
        if event['example']:
            file_content += f'-- {event["description"]}的例子\n'
            file_content += f'{event["example"]}\n\n'

        # make content
        file_content += f'-- {event["description"]}\n'
        args_txt = ""
        for param in event['args']:
            file_content += f'---@param {param["name"]} {param["type"]} {param["comment"]}\n'
            args_txt += f'{param["name"]},'
        
        if 'return' in event.keys():
            file_content += f'---@return {event["return"]["type"]} #{event["return"]["comment"]}\n'
            file_content += f'function isc_test_{package_name}:{event["function_name"]}({args_txt[:-1]})\n\tlocal result = edi.{package_name}:{event["function_name"]}({args_txt[:-1]})\n\tLOGI("isc_test_{package_name} {event["function_name"]}: " .. tostring(result))\nend\n\n'
        
        else:
            file_content += f'function isc_test_{package_name}:{event["function_name"]}({args_txt[:-1]})\n\tedi.{package_name}:{event["function_name"]}({args_txt[:-1]})\nend\n\n'

    # write to file
    return file_content

# create lua code for table package
def table_package(data):
    """
    Converts a dictionary of Lua table data into a Lua file string.

    Args:
        data (dict): A dictionary containing Lua table data.

    Returns:
        str: A string containing the Lua code generated from the input data.
    """
    package_name = list(data.keys())[0]
    assert package_name == 'structures'
    file_content = ""
    for event in data[package_name]:
        # print json nicely
        # print(json.dumps(event, indent=4, ensure_ascii=False))

        # make content
        file_content += f'-- {event["description"]}\n'
        args_txt = ""
        for param in event['args']:
            file_content += f'-- 获取{event["description"]}的{param["comment"]}\n'
            file_content += f'---@field {param["name"]} {param["type"]} {param["comment"]}\n'
            args_txt += f'{param["name"]},'
            file_content += f'local {param["name"]} = {event["function_name"]}["{param["name"]}"]\n\n'
    # write to file
    return file_content

# create lua code for event package
def event_package(data):
    """
    Converts event data into a Lua file content string.

    Args:
        data (dict): A dictionary containing event data.

    Returns:
        str: A string containing the Lua file content.
    """
    package_name = list(data.keys())[0]
    assert package_name == 'events'
    file_content = ""
    for event in data[package_name]:
        # print json nicely
        # print(json.dumps(event, indent=4, ensure_ascii=False))

        # make content
        file_content += f'-- {event["description"]}\n'
        args_txt = ""
        for param in event['args']:
            file_content += f'---@param {param["name"]} {param["type"]} {param["comment"]}\n'
            args_txt += f'{param["name"]},'
        file_content += f'{event["function_name"]}({args_txt[:-1]})\n\n'
    # write to file
    return file_content

# change lua api document json to lua code
import json

def change_lua_api_to_code(json_directory, output_directory):
    """
    Converts JSON files containing Lua API definitions to Lua code and writes the output to the specified directory.

    Args:
        json_directory (str): The directory containing the JSON files to convert.
        output_directory (str): The directory to write the converted Lua code files to.
    """
    json_files = get_files_from_directory(json_directory,'.json')
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            if list(data.keys())[0] == 'events':
                file_content = event_package(data)
            elif list(data.keys())[0] == 'structures':
                file_content = table_package(data)
            else:
                file_content = common_package(data)
        # get file name
        file_name = json_file.split('/')[-1]
        output_filename = output_directory +'/'+ file_name[:-4] + '.lua'
        with open(output_filename, 'w+') as f:
            f.write(file_content)


###############################################################
### extract function from lua code
### related fucntions
import re

def find_comments_positions(lua_code):
    """
    Returns the positions of multi-line and single-line comments in the given Lua code.

    Args:
        lua_code (str): The Lua code to search for comments in.

    Returns:
        list: A list of tuples representing the start and end positions of each comment.
              Each tuple contains two integers: the start position and the end position.
              The list contains both single-line and multi-line comments.
    """
    single_line_comments = [(m.start(), m.end()) for m in re.finditer(r'--.*$', lua_code, re.MULTILINE)]
    multi_line_comments = [(m.start(), m.end()) for m in re.finditer(r'--\[\[[\s\S]*?\]\]', lua_code)]
    return single_line_comments + multi_line_comments

def is_within_comment(position, comments):
    """
    Check if a given position is within a comment.

    Args:
        position (int): The position to check.
        comments (list): A list of tuples representing the start and end positions of comments.

    Returns:
        bool: True if the position is within a comment, False otherwise.
    """
    for start, end in comments:
        if start <= position <= end:
            return True
    return False


def extract_function_names(lua_code):
    """
    Extracts the names of all functions defined in the given Lua code.

    Args:
        lua_code (str): The Lua code to extract function names from.

    Returns:
        list: A list of strings, where each string is the name of a function defined in the Lua code.
    """
    # This regex pattern matches both "function xxx(...)" and "local function xxx(...)"
    pattern = re.compile(r'\b(?:local\s+)?function\s+([\w_:.]+)\s*?\(')
    matches = pattern.findall(lua_code)
    return matches

def find_function_end(lua_code, start_index):
    """
    Finds the end of a Lua function starting from the given index in the code.

    Args:
        lua_code (str): The Lua code to search in.
        start_index (int): The index to start searching from.

    Returns:
        int: The index of the end of the function, or -1 if not found.
    """
    stack = []
    words = re.finditer(r'\b(function|if|for|while|end)\b', lua_code[start_index:])
    comments = find_comments_positions(lua_code)
    
    for match in words:
        if is_within_comment(match.start() + start_index, comments):
            continue
        
        word = match.group(0)
        if word in ["function","for","if","while"]:
            stack.append(word)
        elif word == "end":
            stack.pop()
        
        if not stack:
            return start_index + match.end()

    return -1

def extract_functions(lua_code):
    """
    Extracts all the functions from the given Lua code and returns them as a list of dictionaries.
    
    Each dictionary contains the following keys:
    - 'name': the name of the function
    - 'body': the body of the function
    - 'comment': any comments preceding the function
    
    Args:
    - lua_code (str): the Lua code to extract functions from
    
    Returns:
    - A dictionary with a single key 'functions' whose value is a list of dictionaries, each representing a function.
    """
    results = []
    comments = find_comments_positions(lua_code)
    
    for match in re.finditer(r'\b(?:local\s+)?function\b', lua_code):
        # If the function keyword is within a comment, skip it
        if is_within_comment(match.start(), comments):
            continue
        
        # Go backwards from the function start to capture any preceding comments
        
        comment_start = match.start()
        while comment_start > 0 and (lua_code[comment_start-1] == '\n' or is_within_comment(comment_start-1, comments)):
            comment_start -= 1
        
        # extract function comments
        if comment_start != match.start():
            function_comment_part = lua_code[comment_start:match.start()]
        else:
            function_comment_part = ''
            

        # extract function body
        start_index = match.start()
        end_index = find_function_end(lua_code, match.start())
        
        if end_index != -1:
            function_name = extract_function_names(lua_code[start_index:end_index])[0]
            results.append({'name':function_name,'body':lua_code[start_index:end_index],'comment':function_comment_part})
        else:
            print("Error: Matching end not found for function starting at index", start_index)
    return {'functions':results}




###############################################################
### generate luacoder dataset
def generate_luacoder_dataset(dataset_folder, dataset_name, model_path, files_directory, save_all=False):
    """
    Generates a LuaCoder dataset from Lua files in the specified directories.

    Args:
        dataset_folder (str): The folder to save the generated dataset in.
        dataset_name (str): The name to give the generated dataset.
        model_path (str): The path to the LuaCoder model to use for tokenization.
        files_directory (list): A list of directories containing Lua files to include in the dataset.
        save_all (bool, optional): Whether to save the intermediate and final datasets to disk. Defaults to False.

    Returns:
        None
    """
    # function code here
    # some example of parameters
    # dataset_folder = '/home/uework/AiWeb/pwz/github/data/dataset'
    # dataset_name = 'test-lua'
    # model_path = '/home/uework/AiWeb/pwz/github/model/starcoderbase-3b'
    # files_directory = ['/home/uework/AiWeb/pwz/github/data/lua/api-doc-lua',
    #                 '/home/uework/AiWeb/pwz/github/data/lua/shengqu',
    #                 '/home/uework/AiWeb/pwz/github/data/lua/dialogue'
    #                 ]

    # load lua files
    # lua file from api docs
    print(f'load lua files from {len(files_directory)} resources')
    lua_files = []
    for index, directory in enumerate(files_directory):
        print(f'{index+1}: {directory}')

    for directory in files_directory:
        lua_files.extend(get_files_from_directory(directory, '.lua'))
    content = []
    for file in tqdm(lua_files):
        with open(file, 'r') as f:
            content.append([file,f.read()])

    df = pd.DataFrame(content, columns=['filepath','content'])



    # save dataset and shuffle
    dataset = Dataset.from_pandas(df)
    dataset = dataset.shuffle()
    # dataset = dataset.train_test_split(test_size=0.05)
    print(f'generate dataset {dataset_name} with {len(dataset)} samples')

    if save_all:
        dataset.save_to_disk(dataset_folder+'/'+dataset_name)
        print(f'save dataset to {dataset_folder}/{dataset_name}')

    # clean dataset
    print(f'clean dataset {dataset_name}')
    args_dict = {
        'tokenizer': model_path,
        'dataset_name': dataset_folder+'/'+dataset_name,
        'output_dir': dataset_folder+'/'+dataset_name+'-clean'
    }
    if save_all:
        dataset = clean_dataset(args_dict,save_to_disk=True)
        print(f'save clean dataset to {dataset_folder}/{dataset_name}-clean')
    else:
        dataset = clean_dataset(args_dict,save_to_disk=False)
    
    # generate dataset for training
    print(f'generate clean dataset {dataset_name}-clean with {len(dataset)} samples')
    print(f'generate dataset for training')
    # sample random 5% for evaluation
    eval_dataset = dataset.train_test_split(test_size=0.05)['test']
    dataset = DatasetDict({
        'train': dataset,
        'test': eval_dataset,
    })
    dataset.save_to_disk(dataset_folder+'/'+dataset_name+'-train')
    print(f'save dataset to {dataset_folder}/{dataset_name}-train')



#################################################################
#### generate lua code from prompt and completion
def create_content(df):
    """
    Creates content for each row in the given DataFrame `df` based on the response and prompt columns.
    If a code block is found in the response column, it is included in the content with the corresponding prompt.
    If no code block is found, a default message is included in the content.
    
    Args:
    - df: pandas DataFrame with columns 'prompt', 'response', and 'content'
    
    Returns:
    - df: pandas DataFrame with updated 'content' column
    """
    for i in range(len(df)):
        code_blocks = find_code_blocks(str(df.loc[i, 'response']))
        notaion = df.loc[i, 'prompt']
        if len(code_blocks)>0:
            df.loc[i, 'content'] = f'-- {notaion}\n{code_blocks[0]}'
        else:
            df.loc[i, 'content'] = f'-- {notaion}\n-- 还没学会怎么写这个代码呢TAT'
    return df


def dialogue_to_lua(dialogue_df):
    """
    Converts a pandas DataFrame of dialogue content into a Lua file for use in a chatbot.

    Args:
        dialogue_df (pandas.DataFrame): A DataFrame containing dialogue content.

    Returns:
        None
    """
    # Load a dataset from the Hugging Face library
    dialogue_df = create_content(dialogue_df)
    # write df['content'] to a file
    with open('/home/uework/AiWeb/pwz/github/data/lua/dialogue/data-cn.lua', 'w+') as f:
        for i in range(len(dialogue_df)):
            f.write(dialogue_df.loc[i, 'content']+'\n\n')


#################################################################
#### transform openassitent dialogue to json code
import re

def dialogue_to_json(dialogue):
    """
    Converts a dialogue string into a JSON object.

    Args:
        dialogue (str): The dialogue string to convert.

    Returns:
        dict: A dictionary containing a list of messages, where each message is represented as a dictionary with a 'content' and 'role' field.
    """
    # Split the dialogue into individual messages using the '###' delimiter
    messages = re.split(r'###\s*', dialogue.strip())
    messages = [msg for msg in messages if msg]  # Remove empty strings

    # Transform each message into the desired dictionary format
    json_messages = []
    for msg in messages:
        if msg.startswith("Human:"):
            role = "user"
            content = msg[len("Human:"):].strip()
        elif msg.startswith("Assistant:"):
            role = "assistant"
            content = msg[len("Assistant:"):].strip()
        else:
            continue  # Skip any lines that don't match the expected format

        json_messages.append({
            "content": content,
            "role": role
        })

    return {
        "messages": json_messages
    }

#################################################################
#### find substring in file
import os
import concurrent.futures

def find_substring_in_file(filename, substring):
    """
    Search for a substring in a file and return the line numbers where it's found.

    Args:
        filename (str): The name of the file to search.
        substring (str): The substring to search for.

    Returns:
        tuple: A tuple containing the filename and a list of line numbers where the substring was found. 
               Returns None if the substring was not found in the file.
    """
    locations = []
    with open(filename, 'r') as file:
        for line_num, line in enumerate(file, 1):
            if substring in line:
                locations.append(line_num)
    return (filename, locations) if locations else None

def search_files_for_substring(directory, substring):
    """
    Search for a substring in all Lua files in a directory and its subdirectories using multithreading.

    Args:
        directory (str): The directory to search for Lua files.
        substring (str): The substring to search for in the Lua files.

    Returns:
        A list of file paths where the substring was found.
    """
    lua_files = [os.path.join(dirpath, filename)
                 for dirpath, dirnames, filenames in os.walk(directory)
                 for filename in filenames if filename.endswith('.lua')]
    
    # Limit the number of threads to 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(find_substring_in_file, lua_files, [substring]*len(lua_files)))

    return [result for result in results if result]


#################################################################
#### find code based on templetes
#### template samples
# -- notation describe the function
# local function action(param)
#     -- get or new some data
#     -- processing data
#     -- perform functionality
# end
# GameEvent.register_ssr(UEComponent, action)
# -- notation describe the function
# local function listenerAction(param)
#     -- get or new some data
#     -- processing data
#     -- perform functionality
# end
# GameEvent.add(eventDefAction, listenerAction, eventName)


def extract_functions(lua_code, mute=False):
    """
    Extracts all functions from the given Lua code and returns a dictionary containing the function names, bodies, comments, and any registered events.

    Args:
        lua_code (str): The Lua code to extract functions from.
        mute (bool, optional): If True, suppresses error messages. Defaults to False.

    Returns:
        dict: A dictionary containing the extracted functions, with each function represented as a dictionary with keys 'name', 'body', 'comment', and 'register' (if applicable).
    """
    results = []
    comments = find_comments_positions(lua_code)
    
    for match in re.finditer(r'\b(?:local\s+)?function\b', lua_code):
        # If the function keyword is within a comment, skip it
        if is_within_comment(match.start(), comments):
            continue
        
        # Go backwards from the function start to capture any preceding comments
        comment_start = match.start()
        while comment_start > 0 and (lua_code[comment_start-1] == '\n' or is_within_comment(comment_start-1, comments)):
            comment_start -= 1
        
        # extract function comments
        if comment_start != match.start():
            function_comment_part = lua_code[comment_start:match.start()]
        else:
            function_comment_part = ''
            

        # extract function body
        start_index = match.start()
        end_index = find_function_end(lua_code, match.start())
        
        if end_index != -1:
            result = extract_function_names(lua_code[start_index:end_index])
            if result:
                function_name = result[0]
            else:
                continue
            function_body = lua_code[start_index:end_index]
            
            # Check the lines after the function ends for the desired pattern
            next_lines_start = end_index
            while next_lines_start < len(lua_code) and lua_code[next_lines_start] not in ['\n', '\r']:
                next_lines_start += 1
            next_lines_end = next_lines_start + 1
            while next_lines_end < len(lua_code) and lua_code[next_lines_end] not in ['\n', '\r']:
                next_lines_end += 1
            next_line = lua_code[next_lines_start:next_lines_end].strip()
            register = None
            if re.match(r'((GameEvent|g_mir.ssr)\.(add|register_ssr|remove|removeByNameAndTag|removeByTag|removeAll|push|register|execute|addEvent)\(.+\))', next_line):
                register = next_line
            if register:
                results.append({'name': function_name, 'body': function_body, 'comment': function_comment_part, 'register': register})
            else:
                continue
        else:
            if not mute:
                print("Error: Matching end not found for function starting at index", start_index)
    return {'functions': results}

def find_all_samples(directory_path, mute=False):
    """
    Finds all Lua files in the given directory and its subdirectories, reads their content, and extracts all functions
    using the extract_functions function. Returns a dictionary containing the file paths and their extracted functions.

    Args:
        directory_path (str): The path of the directory to search for Lua files.
        mute (bool, optional): Whether to mute the output of the extract_functions function. Defaults to False.

    Returns:
        dict: A dictionary containing the file paths and their extracted functions.
    """
    lua_files = [os.path.join(root, file) for root, _, files in os.walk(directory_path) for file in files if file.endswith('.lua')]

    # Read the content of each Lua file and apply the find_template_samples function
    results = []
    for file_path in tqdm(lua_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            lua_code = f.read()
            functions = extract_functions(lua_code,mute)['functions']
            if functions:
                results.append({'file_path':file_path,'functions':functions})
    return {'results': results}

def samples_to_lua_file(results, file_path):
    """
    Write Lua code to a file from the given results.

    Args:
        results (dict): A dictionary containing the results to write to the file.
        file_path (str): The path to the file to write the Lua code to.

    Returns:
        None
    """
    with open(file_path, 'w+') as f:
        for sample in results['results']:
            file_path = sample['file_path']
            for function in sample['functions']:
                f.write(function['comment'])
                f.write(function['body'])
                f.write('\n')
                f.write(function['register'])