#
#   TagBotFT - Tag Bot For Tables
#

import argparse
import os
import platform
import pandas as pd
import sys
import numpy as np
import time
import datetime
import configparser
from itertools import repeat

import tagbotft_lib as tl
import tagbotft_lib_file as tf
import tagbotft_lib_db as td

#
# TagBotFT starts
#
if __name__ == '__main__':

    # Get the number of available CPU cores    
    cores = os.cpu_count()

    tl.message('\nStarting TagBotFT -- Tag Bot for Tables\n')

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
    parser.add_argument(
        '-r', '--rebuild', help='Reload learning data', action='store_true')
    parser.add_argument(
        '--inifile', default='tagbotft.ini', type=str,
        dest='inifile', help='Different INI file')
    args = parser.parse_args()
    
    # Test run limits the amounts of input data
    if args.test:
        max_lines = 10000000
        max_input = 1000
        test = True
        print('Running in TEST mode! Using less data to speed things up...')
    
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

    # Use a different INI file
    if args.inifile:
        ini_file = args.inifile

    # Setting initial parameters
    # Set default variables
    org_data = []
    myDir = os.path.abspath(os.path.dirname(__file__)) + '/'
    config = configparser.ConfigParser()
    config.sections()
    config.read(myDir + ini_file)
    dbf = config['Settings']['dbf']
    dbl = config['Settings']['dbl']
    org_file = config['Settings']['org_file']
    wsheet = config['Settings']['wsheet']
    non_relevant_tag = config['Settings']['non_relevant_tag']
    exclude_file = config['Settings']['exclude_file']
    tag_col = config['Settings']['tag_col_txt']
    text_col = config['Settings']['text_col_txt']
    input_files = config['Settings']['input_files']
    max_cols = config.getint('Settings','max_cols')

    # Only rebuild data when the parameter is set
    if args.rebuild:
        # Reading data from Excel file
        work_data = tf.read_xl_learn(org_file, max_lines)
        td.write_learn_db(work_data)
        
    # Reading the data from database ensures clear column formats
    work_data = td.read_learn_db()

    # Read the input data into dataframe
    if input_files[-3:] == "TXT":
        newData = tf.read_SAP_1(input_files, max_input, exclude_file, work_data)
    elif input_files[-3:] == "csv":
        newData = tf.read_CSV(input_files, max_input, exclude_file, work_data)
    else:
        exit()

    # Generate a dataframe from working data
    learn_df = tl.get_df(work_data, non_relevant_tag)

    # Start learning from already tagged data
    if args.newlearn:
        # Generate Ngrams for relevant/non-relevant from text column
        text_df, non_text_df = tl.get_text_df(learn_df)

        # Generate Ngrams for relevant/non-relevant from other columns
        other_df, other_cols = tl.get_other_df(learn_df)

        # Generate Ngrams for learned tags from text column
        tags_df, non_tags_df = tl.get_tags_df(work_data, non_text_df)

        # Analyze relationship between important columns and tags
        tl.tag_to_other(other_cols, work_data, newData)

    # Divide the new data into existing and new entries
    # Only new entries will be further processed after this step
    old_df, new_df = tl.get_existing(newData, work_data, max_lines)
   
    # Writing existing entries to Excel file
    tf.writeXLS('result_data_old.xlsx', old_df)

    # Stage: tagging the not relevant data first
    tagged_non_relevant, untagged = tl.tag_non_relevant(new_df)

    # Stage: tagging relevant data from other than text column
    #tagged_other, untagged = tl.tag_other(untagged)

    # Stage: tagging relevant data with unique Ngrams 
    tagged_relevant, untagged = tl.tag_relevant(untagged)

    # Stage: tagging relevant data by Levenshtein distance analysis
    tagged_similar = tl.tag_lev_df(untagged, work_data, max_lines)

    # Concatenating results from the stages
    tagged = tagged_non_relevant
    tagged = pd.concat([tagged, tagged_relevant])
    tagged = pd.concat([tagged, tagged_similar])

    print('Tagged data: ' + str(len(tagged)))

    # Writing tagging results to Excel file
    tf.writeXLS('result_data.xlsx', tagged, tag_col_txt, text_col_txt)

    # End of TagBotFT
    tl.message('TagBotFT has finished')
    print('Execution took {} (h:min:s, wall clock time).' \
        .format(datetime.timedelta(seconds=round(time.time() - start_time))))
