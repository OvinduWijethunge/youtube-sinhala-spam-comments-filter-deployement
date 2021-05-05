# -*- coding: utf-8 -*-
"""
Created on Wed May  5 22:18:41 2021

@author: Ovindu Wijethunge
"""
import pandas as pd
import numpy as np
import scipy.stats as stat

def data_modification(df):
    dfs = df
    dfs = dfs.drop('cid',axis=1)
    dfs['no_of_sentences'] = np.where(dfs['no_of_sentences']>3,3,dfs['no_of_sentences'])
    dfs['num_of_punctuations'] = np.where(dfs['num_of_punctuations']>6,6,dfs['num_of_punctuations'])
    
    dfs['sim_content'],parameters=stat.boxcox(dfs['sim_content']+0.000001)
    dfs['sim_comment'],parametes=stat.boxcox(dfs['sim_comment']+0.000001)
    dfs['word_count'],parameters=stat.boxcox(dfs['word_count']+0.000001)
    dfs['length_of_comment'],parameters=stat.boxcox(dfs['length_of_comment']+0.000001)
    dfs['post_coment_gap'],parameters=stat.boxcox(dfs['post_coment_gap']+0.000001)
    
    dfs['no_of_sentences'] = dfs['no_of_sentences'].astype(str)
    dfs['num_of_punctuations'] = dfs['num_of_punctuations'].astype(str)
    dfs['is_period_sequence'] = dfs['is_period_sequence'].astype(str)
    dfs['is_link'] = dfs['is_link'].astype(str)
    dfs['is_youtube_link'] = dfs['is_youtube_link'].astype(str)
    dfs['is_number'] = dfs['is_number'].astype(str)
    
    new_df = pd.get_dummies(dfs,drop_first=True)
    
    return new_df