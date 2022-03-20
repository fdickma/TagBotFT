import pandas as pd
import sys
import sqlite3
import multiprocessing as mp
import numpy as np
import time
import datetime

import tagbotft_lib_file as tf
import tagbotft_lib as tl

# Write the text Ngrams to an SQLite database
def write_rel_text_db(ngrams):
    # Store Ngrams and corresponding Tags in SQLite Database
    detect_types=sqlite3.PARSE_DECLTYPES
    conn = sqlite3.connect('tagbotft.sqlite', \
            detect_types=sqlite3.PARSE_DECLTYPES)
    ngrams.to_sql('relNgrams_Text', conn, if_exists='replace', index=False)

# Write the other columns to an SQLite database
def write_other_db(other_df_set):
    # Open database connection
    conn = sqlite3.connect('tagbotft.sqlite', \
            detect_types=sqlite3.PARSE_DECLTYPES)
    for o in other_df_set:
        df = o[1].drop(o[1].columns.difference(['Tag', o[0]]), axis=1)
        df.to_sql('relNgrams_Col_'+o[0], conn, if_exists='replace', \
                index=False)

# Write the other columns to an SQLite database
def write_other_cols_db(df):
    # Open database connection
    conn = sqlite3.connect('tagbotft.sqlite', \
            detect_types=sqlite3.PARSE_DECLTYPES)
    df.to_sql('other_Col_Values', conn, if_exists='replace', index=False)

# Write the other columns to an SQLite database
def read_other_cols_db():
    # Open database connection
    conn = sqlite3.connect('tagbotft.sqlite', \
            detect_types=sqlite3.PARSE_DECLTYPES)
    df = pd.read_sql('select * from other_Col_Values', conn)
    return df['Col'].unique().tolist()

# Write the other columns to an SQLite database
def read_other_tag_vals_db():
    # Open database connection
    conn = sqlite3.connect('tagbotft.sqlite', \
            detect_types=sqlite3.PARSE_DECLTYPES)
    df = pd.read_sql('select * from other_Col_Values', conn)
    return df

# Write the text Ngrams to an SQLite database
def write_text_db(ngrams):
    # Store Ngrams and corresponding Tags in SQLite Database
    conn = sqlite3.connect('tagbotft.sqlite', \
            detect_types=sqlite3.PARSE_DECLTYPES)
    ngrams.to_sql('genNgrams_Text', conn, if_exists='replace', index=False)
    # Verifying data
    pd.read_sql('select * from genNgrams_Text', conn)

# Write the learn data to an SQLite database
def write_learn_db(learn_data):
    # First scan the lean data result for empty lines
    empty_rows = []
    learn_cols = len(learn_data.columns)
    for index, test_row in learn_data.iterrows():
        is_empty = 0
        for test_col in test_row:
            if (test_col == 'None') or (test_col == None) or (test_col == ''):
                is_empty += 1
        # Check if nearly all colums of a row have been empty
        if is_empty > (learn_cols - 3):
            empty_rows.append(index)
    
    # Filter out the empty rows from the dataframe
    learn_data = learn_data[~learn_data.index.isin(empty_rows)]
    print("Learned data lines:", len(learn_data))
    
    # Store Ngrams and corresponding Tags in SQLite Database
    conn = sqlite3.connect('tagbotft_learn.sqlite', \
            detect_types=sqlite3.PARSE_DECLTYPES)
    learn_data.to_sql('learn_data', conn, if_exists='replace', index=False)

# Read the learn data from an SQLite database
def read_learn_db():
    # Open database connection
    tl.message("Reading database with learned data")
    conn = sqlite3.connect('tagbotft_learn.sqlite', \
            detect_types=sqlite3.PARSE_DECLTYPES)
    df = pd.read_sql('select * from learn_data', conn)
    df = df.replace(r"^None$", '', regex=True)
    return df
