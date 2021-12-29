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
import tagbotft_lib_file as tf
    
#
# TagBotFT starts
#
if __name__ == '__main__':

    # Get the number of available CPU cores    
    cores = os.cpu_count()

    # Using parameter to enable Test mode for faster testing.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--test', help='Test run - less data', action='store_true')
    parser.add_argument(
        '-s', '--single', help='Use only one CPU core', action='store_true')
    parser.add_argument(
        '-l', '--logfile', help='Log console to file', action='store_true')
    parser.add_argument(
        '-m', '--maxcol', help='Use maximum length in a columns', action='store_true')
    parser.add_argument(
        '-n', '--newlearn', help='Process learning data', action='store_true')
    args = parser.parse_args()
    
    # Test run limits the amounts of input data
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

    # If not in Test mode, use as many input data as possible
    if not args.test:
        max_lines = 100000000
        max_input = 0
        test = False

    # Redirect output to files for easier debugging
    if args.logfile:
        sys.stdout = open('tagbotft.out', 'w')
        sys.stderr = open('tagbotft.err', 'w')

    # Set use to only one CPU core 
    if args.single:
        cores = 1

    # Get the start time
    start_time = time.time()

    # Setting initial parameters
    org_data = []
    dbf = "tagbotft.sqlite"
    org_file = 'learn_data.xlsx'
    wsheet = "Komplett"
    non_relevant = "y"
    tag_col_txt = 'System'
    text_col_txt = 'Beschreibung'
    max_cols = 11

    # Reading exclude file
    exclude_df = tf.readExclude("excludes.lst")

    # Read the SAP data into one list
    newData = tf.read_SAP_1('*.TXT', max_input, test, exclude_df)

    # Reading data from Excel file
    work_data = tf.read_xl_learn(org_file, wsheet, max_lines, max_cols, \
                            tag_col_txt, text_col_txt)

    if args.maxcol == True:
        newData = tl.maxcol(newData, work_data)

    # Generate a dataframe from working data
    learn_df = tl.get_df(work_data, non_relevant)

    # Start learning from already tagged data
    if newlearn:
        # Generate Ngrams for relevant/non-relevant from text column
        text_df, non_text_df = tl.get_text_df(learn_df)

        # Generate Ngrams for relevant/non-relevant from other columns
        other_df, other_cols = tl.get_other_df(learn_df)

        # Generate Ngrams for learned tags from text column
        tags_df, non_tags_df = tl.get_tags_df(work_data, non_text_df)

        # Analyze relationship between important columns and tags
        tl.tag_to_other(other_cols, work_data, newData)

    # Divide the new data into existing and new entries
    # Only new entries will be further processed 
    old_df, new_df = tl.get_existing(newData, work_data, cores, max_lines)
   
    # Writing existing entries to Excel file
    tf.writeXLS('result_data_old.xlsx', old_df, tag_col_txt, text_col_txt)
    
    # Stage 1 - tagging the not relevant data first
    tagged_non_relevant, untagged = tl.tag_non_relevant(new_df)

    # Stage 2 - tagging relevant data from other than text column
    tagged_other, untagged = tl.tag_other(untagged)

    # Stage 3 - tagging relevant data with unique Ngrams 
    tagged_relevant, untagged = tl.tag_relevant(untagged)

    # Stage 4 - tagging relevant data by Levenshtein distance analysis
    tagged_similar = tl.tag_lev_df(untagged, work_data, cores, max_lines)

    # Concatenating results from the four stages
    tagged = tagged_non_relevant.append(tagged_other)
    tagged = tagged.append(tagged_relevant)
    tagged = tagged.append(tagged_similar)

    print('Tagged data: ' + str(len(tagged)))

    # Writing tagging results to Excel file
    tf.writeXLS('result_data.xlsx', tagged, tag_col_txt, text_col_txt)

    print('Execution took {} (h:min:s, wall clock time).' \
        .format(datetime.timedelta(seconds=round(time.time() - start_time))))
