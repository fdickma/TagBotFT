#
#   TagBotFT - Tag Bot For Tables
#

import argparse
import os
import platform
import pandas as pd
import sys
import re
import multiprocessing as mp
import numpy as np
import time
import datetime
from itertools import repeat

import tagbotft_lib as tl
import tagbotft_file_lib as tf
    
#
# TagBotFT starts
#
if __name__ == '__main__':
    
    cores = os.cpu_count()
    use_os = platform.system()

    # Using parameter to enable Test mode for faster testing.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--test', help='Test run - less data', action='store_true')
    parser.add_argument(
        '-s', '--single', help='Use only one CPU core', action='store_true')
    parser.add_argument(
        '-l', '--logfile', help='Log console to file', action='store_true')
    parser.add_argument(
        '-n', '--newlearn', help='Process learning data', action='store_true')
    args = parser.parse_args()
    
    if args.test:
        max_lines = 1000
        max_input = 1000
        test = True
        print('Running in TEST mode! Using less data to speed things up...')
    
    # Check if learning data is to be processed
    if args.newlearn:
        newlearn = True
    else:
        newlearn = False

    if not args.test:
        max_lines = 100000000
        max_input = 0
        test = False

    if args.logfile:
        # Redirect output to files for easier debugging
        sys.stdout = open('tagbotft.out', 'w')
        sys.stderr = open('tagbotft.err', 'w')

    if args.single:
        cores = 1

    # Get the start time
    start_time = time.time()

    # Setting initial parameters
    org_data = []
    org_file = 'learn_data.xlsx'
    wsheet = "Komplett"
    non_relevant = "y"
    tag_col_txt = 'System'
    text_col_txt = 'Beschreibung'
    max_cols = 11

    # Read the SAP data into one list
    print("--------------------------------------")
    print("Loading data...")
    print("--------------------------------------")

    # Reading exclude file
    exclude_df = tf.readExclude("excludes.lst")
     
    newData = tf.read_SAP_1('*.TXT', max_input, test)
    newData = tl.filter_input(newData, exclude_df)

    # Reading data from Excel file
    work_data = tf.read_xl_learn(org_file, wsheet, max_lines, max_cols, \
                            tag_col_txt, text_col_txt)

    # Generate a dataframe from working data
    learn_df = tl.get_df(work_data, non_relevant)

    if newlearn:
        # Generate Ngrams for relevant/non-relevant from text column
        text_df, non_text_df = tl.get_text_df(learn_df)

        # Generate Ngrams for relevant/non-relevant from other columns
        other_df = tl.get_other_df(learn_df)

        # Generate Ngrams for learned tags from text column
        tags_df, non_tags_df = tl.get_tags_df(work_data, non_text_df)

    # Divide the new data into existing and new entries
    # Only new entries will be further processed 
    old_df, new_df = tl.get_existing(newData, work_data, cores, max_lines)
    tf.writeXLS('result_data_old.xlsx', old_df, tag_col_txt, text_col_txt)
    
    # Tagging the not relevant data first
    tagged_non_relevant, untagged = tl.tag_non_relevant(new_df)
    tagged_relevant, untagged = tl.tag_relevant(untagged)
    tagged_similar = tl.tag_lev_df(untagged, work_data, cores, max_lines)

    tagged = tagged_non_relevant.append(tagged_relevant)
    tagged = tagged.append(tagged_similar)

    tf.writeXLS('result_data.xlsx', tagged, tag_col_txt, text_col_txt)

    print('Execution took {} (h:min:s, wall clock time).' \
        .format(datetime.timedelta(seconds=round(time.time() - start_time))))
