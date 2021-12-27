import pandas as pd
import sys
import sqlite3
import multiprocessing as mp
import numpy as np
import time
import datetime

import tagbotft_lib_file as tf

# Write the text Ngrams to an SQLite database
def write_rel_text_db(ngrams):
    # Store Ngrams and corresponding Tags in SQLite Database
    conn = sqlite3.connect('Ngrams.sqlite')
    ngrams.to_sql('relNgrams_Text', conn, if_exists='replace', index=False)
    # Verifying data
    pd.read_sql('select * from relNgrams_Text', conn)

# Write the other columns to an SQLite database
def write_other_db(other_df_set):
    # Open database connection
    conn = sqlite3.connect('Ngrams.sqlite')
    for o in other_df_set:
        df = o[1].drop(o[1].columns.difference(['Tag', o[0]]), axis=1)
        df.to_sql('relNgrams_Col_'+o[0], conn, if_exists='replace', \
                index=False)

# Write the text Ngrams to an SQLite database
def write_text_db(ngrams):
    # Store Ngrams and corresponding Tags in SQLite Database
    conn = sqlite3.connect('Ngrams.sqlite')
    ngrams.to_sql('genNgrams_Text', conn, if_exists='replace', index=False)
    # Verifying data
    pd.read_sql('select * from genNgrams_Text', conn)


