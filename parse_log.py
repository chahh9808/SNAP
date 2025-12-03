import os
import re
import pandas as pd
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm


def process_log_folders(parent_folder):
    date2parse = re.compile(r'^(510)')
    
    dataset2parse = ['cifar10log','cifar100log','INlog']
    
    log_folders = []

    for dataset in dataset2parse:
        dataset_folder = os.path.join(parent_folder, dataset)
        if os.path.isdir(dataset_folder):
            for root, dirs, files in os.walk(dataset_folder):
                for d in dirs:
                    if date2parse.match(d) and 'log' in d:
                        log_folders.append(os.path.join(root, d))

    total_data_dict = defaultdict(lambda: defaultdict(dict))
    

    for log_folder in tqdm(log_folders):
        corruption = log_folder.split('/')[-1].split('_')[0][3:]
        dataset = log_folder.split('/')[-1].split('_')[1].split('log')[0]

        filename_pattern = re.compile(
            r'(0|1|2)_(src|bn|tent|cotta|eata|sar|rotta)_([\d\.]+)_(all|high_conf|low_entr|basic)_(RAND|WASS_OPP|ENTR)_([\d\.]+)_([\d\.]+)_([\d\.e\-]+)_([\d]+)_([\w]+)\.txt$'
        )
        log_content_pattern = re.compile(r'\[0\] \w+@[\w_]+ Acc: (\d+\.\d+)%\s*')

        # Helper to parse each log file
        def process_file(filepath):
            filename = os.path.basename(filepath)
            match = filename_pattern.search(filename)
            if match:
                seed, alg, adaptrate, adst, rmst, iobmn_k, iobmn_s, lr, mem_size, filetype = match.groups()
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        acc_match = log_content_pattern.search(line)
                        if acc_match:
                            accuracy = float(acc_match.group(1))
                            key = (alg, adaptrate, adst, rmst, iobmn_k, iobmn_s, mem_size, corruption, dataset)
                            if seed not in total_data_dict[filetype][key]:
                                total_data_dict[filetype][key][seed] = []
                            total_data_dict[filetype][key][seed].append(accuracy)

        for root, dirs, files in os.walk(log_folder):
            for file in files:
                process_file(os.path.join(root, file))

    required_seeds = {'0', '1', '2'}
    rows = []
    
    for filetype, data in total_data_dict.items():
        missing_combinations = []
        
        for k, v in data.items():
            available_seeds = set(v.keys())
            missing_seeds = required_seeds - available_seeds
            if missing_seeds:
                missing_combinations.append((k, missing_seeds))
                
            for seed in available_seeds:
                rows.append({
                    'filetype': filetype, 'alg': k[0], 'adaptrate': k[1], 'adst': k[2], 'rmst': k[3], 'iobmn_k': k[4], 'iobmn_s': k[5], 'mem_size': k[6],
                    'corruption': k[7], 'dataset': k[8], 'seed': seed,
                    'accuracy': v[seed][0]
                })
        
        if missing_combinations:
            print(f"Missing combinations in filetype {filetype}:")
            for combo, missing in missing_combinations:
                print(f"Combination: {combo}, Missing seeds: {missing}")
        
    df = pd.DataFrame(rows)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'results_all_{current_time}.csv'
    path_to_save = './logs'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    df.to_csv(os.path.join(path_to_save,file_name), index=False)
    print(f'File saved as: {file_name}')
        

parent_txtlog_folder = './logs/txtlogs'
process_log_folders(parent_folder=parent_txtlog_folder)
