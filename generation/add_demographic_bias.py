
import json

from loguru import logger
import pandas as pd


def load_demographics(demographics_file:str='data/demographics_data.json'):
    with open(demographics_file, 'r') as f:
        demographics = json.load(f) 
    return demographics


def add_writing_style(data, demographics):
    updated_data = {}

    for eval_set, eval_data in data.items():
        df = pd.DataFrame()
        for demographic, dem_data in demographics.items():
                        
            dem_df = pd.DataFrame(dem_data)
            eval_df = pd.DataFrame(eval_data)
            df_tmp = dem_df.assign(_k=1).merge(eval_df.assign(_k=1), on="_k").drop(columns="_k")
            df_tmp['prompt'] = df_tmp.apply(lambda row: row['text'] + row['prompt'], axis=1)
            df_tmp['demographic'] = demographic
            df = pd.concat([df, df_tmp], ignore_index=True) 
        
        updated_data[eval_set] = df.to_dict(orient='list')

    return updated_data


def add_indicator(data, demographics, role: str):
    updated_data = {}

    for eval_set, eval_data in data.items():
        df = pd.DataFrame()
        for demographic, dem_data in demographics.items():
                        
            dem_df = pd.DataFrame(dem_data)
            eval_df = pd.DataFrame(eval_data)
            df_tmp = dem_df.assign(_k=1).merge(eval_df.assign(_k=1), on="_k").drop(columns="_k")
            if role == 'user':
                df_tmp['prompt'] = df_tmp.apply(
                    lambda row: [{'role': 'user', 'content': row['text']+ '\n' + row['prompt'][0]['content']}], axis=1
                )
            elif role == 'system':
                df_tmp['prompt'] = df_tmp.apply(
                    lambda row: [{'role': 'system', 'content': row['text']}] + row['prompt'], axis=1
                )
            df_tmp['demographic'] = demographic
            df = pd.concat([df, df_tmp], ignore_index=True) 
        
        updated_data[eval_set] = df.to_dict(orient='list')

    return updated_data


def add_demographic_bias(data, demographics, dem_bias_version):
    if dem_bias_version == 'name-system':
        data = add_indicator(data=data, demographics=demographics['name-system'], role='system')
    elif dem_bias_version == 'name-user':
        data = add_indicator(data=data, demographics=demographics['name-user'], role='user')
    elif dem_bias_version == 'explicit-mention-system':
        data = add_indicator(data=data, demographics=demographics['explicit-mention-system'], role='system')
    elif dem_bias_version == 'explicit-mention-user':
        data = add_indicator(data=data, demographics=demographics['explicit-mention-user'], role='user')
    elif dem_bias_version == 'writing-style-human':
        data = add_writing_style(data=data, demographics=demographics['writing-style-human'])
    elif dem_bias_version == 'writing-style-llm':
        data = add_writing_style(data=data, demographics=demographics['writing-style-llm'])
    elif dem_bias_version == 'no-demographics':
        pass
    else:
        raise ValueError(f'Unknown version to introduce demographic bias: {dem_bias_version}')
    return data