from utils import *
import multiprocessing

def create_configs(directory, api_path, test_data_path):
    """
    Creates a list of Config objects based on the given directory, API path, and test data path.

    Args:
        directory (str): The base directory to search for matching folders.
        api_path (str): The path to the API.
        test_data_path (str): The path to the test data.

    Returns:
        list: A list of Config objects.
    """
    base_folder = directory
    matching_folders = get_matching_folders(base_folder)
    numbers = [int(folder.split('-')[-1]) for folder in matching_folders]
    config_list = []
    for test_steps in numbers:
        test_steps = str(test_steps)
        config = {
            'model_path': base_folder +'/checkpoint-'+test_steps,
            'api_path': api_path,
            'test_data_path': test_data_path,
            'output_path': base_folder+'/test_result-'+test_steps+'.csv',
            'report_output_path':  base_folder+'/test_report-'+test_steps+'.md',
            'max_tokens': 128,
            'temperature': 0.2,
            'top_k': 4,
            'top_p': 0.95,
            'prompt_template':"<|system|>\n<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>",
            'gpu_memory_utilization':0.8,
            'stop_word':'',
            'report_type':'markdown',
        }
        config = Config(config)
        config_list.append(config)
    return config_list


if __name__ == "__main__":
    base_folder = '/home/uework/AiWeb/pwz/github/model/luachat-7b/beta2'
    api_path='/home/uework/AiWeb/pwz/github/data/json/api-doc'
    test_data_path='/home/uework/AiWeb/pwz/github/data/prompt/test_quary.csv'
    config_list = create_configs(base_folder,api_path,test_data_path)
    for index,config in enumerate(config_list):
        # Create a Process object
        process = multiprocessing.Process(target=eval, args=(config,))
        # Start the process
        process.start()
        # Optionally, wait for the process to complete
        process.join()
        print(f"Process {index} is done!")
    print("Main process is done!")