import os
import re
import json 
from tqdm import tqdm


def add_dem_group_values(responses):
    
    if 'age_group' not in responses.keys() or len(responses['age_group']) == 0:
        responses['age_group'] = []

    if 'gender' not in responses.keys() or len(responses['gender']) == 0:
        responses['gender'] = []

    if 'ethnicity' not in responses.keys() or len(responses['ethnicity']) == 0:
        responses['ethnicity'] = []

    if all([len(responses[group]) == len(responses['prompt']) for group in ['gender', 'ethnicity', 'age_group']]):
        return responses

    if 'demographic' not in responses.keys():
        responses['demographic'] = [None] * len(responses['prompt'])

    for i, dem in enumerate(responses['demographic']):
        if dem == 'gender':
            gender = re.search(r'(Male|Female|Non-binary)', responses['text'][i]).group(1)
            responses['age_group'].append(None)
            responses['gender'].append(gender)
            responses['ethnicity'].append(None)
        elif dem == 'ethnicity':
            ethnicity = re.search(r'(Black|White|Asian|Hispanic)', responses['text'][i]).group(1)
            responses['age_group'].append(None)
            responses['gender'].append(None)
            responses['ethnicity'].append(ethnicity)
        elif dem == 'age_group':
            age = re.search(r'([0-9]{2,2})', responses['text'][i]).group(1)
            age_group = '18-34' if 18 <= int(age) <= 34 else '35-54' if 35 <= int(age) <= 54 else '55+'
            responses['age_group'].append(age_group)
            responses['gender'].append(None)
            responses['ethnicity'].append(None)
        else:
            responses['age_group'].append(None)
            responses['gender'].append(None)
            responses['ethnicity'].append(None)

    return responses 


def merge_ib_response_batches(response_folder: str, n_batches: int = 3):

    os.makedirs(f'{response_folder}/partial', exist_ok=True)

    files = [x for x in os.listdir(response_folder) if '_ib_' in x]
    files_unique = [x.replace('_0', '') for x in os.listdir('responses') if '_ib_' in x and '_0' in x]
    for file in tqdm(files_unique):
        files_task = [x for x in files if file[:-5] in x and re.search(r'_[0-2]\.', x)]

        if len(files_task) == n_batches:
            with open(f'responses/{file[:-5]}_0.json', 'r') as f:
                responses = json.load(f)
            os.rename(f'responses/{file[:-5]}_0.json', f'responses/partial/{file[:-5]}_0.json')
            for i in range(1,3):        
                with open(f'responses/{file[:-5]}_{i}.json', 'r') as f:
                    responses_i = json.load(f)
                os.rename(f'responses/{file[:-5]}_{i}.json', f'responses/partial/{file[:-5]}_{i}.json')
                
                    
                for key, value in responses_i.items():
                    responses[key].extend(value)
            
            with open(f'responses/{file}', 'w') as f:
                json.dump(responses, f)
        else:
            print(file)


if __name__ == "__main__":

    files = [x for x in os.listdir('responses') if x.endswith('.json')]

    for file in tqdm(files): 
        try:
            with open(f'responses/{file}', 'r') as f:
                responses = json.load(f)
            
            responses = add_dem_group_values(responses)
            
            with open(f'responses/{file}', 'w') as f:
                json.dump(responses, f)
        except Exception as e:
            print(file)
    
    merge_ib_response_batches('responses', n_batches=3)