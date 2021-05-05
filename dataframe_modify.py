# -*- coding: utf-8 -*-
"""
Created on Wed May  5 22:18:41 2021

@author: Ovindu Wijethunge
"""
import pandas as pd
import numpy as np
import scipy.stats as stat

def data_modification(df):
    
    df['no_of_sentences'] = np.where(df['no_of_sentences']>3,3,df['no_of_sentences'])
    df['num_of_punctuations'] = np.where(df['num_of_punctuations']>6,6,df['num_of_punctuations'])
    
    df['sim_content'],parameters=stat.boxcox(df['sim_content']+0.000001)
    df['sim_comment'],parametes=stat.boxcox(df['sim_comment']+0.000001)
    df['word_count'],parameters=stat.boxcox(df['word_count']+0.000001)
    df['length_of_comment'],parameters=stat.boxcox(df['length_of_comment']+0.000001)
    df['post_coment_gap'],parameters=stat.boxcox(df['post_coment_gap']+0.000001)
    
    df['no_of_sentences'] = df['no_of_sentences'].astype(str)
    df['num_of_punctuations'] = df['num_of_punctuations'].astype(str)
    df['is_period_sequence'] = df['is_period_sequence'].astype(str)
    df['is_link'] = df['is_link'].astype(str)
    df['is_youtube_link'] = df['is_youtube_link'].astype(str)
    df['is_number'] = df['is_number'].astype(str)
    
    new_df = pd.get_dummies(df,drop_first=True)
    
    return new_df