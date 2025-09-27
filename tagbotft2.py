import argparse
import os
import platform
import pandas as pd
import sys
import time
import datetime
import configparser
import pandas as pd
import re
import tagbotft2_proc as tp
import tagbotft2_data as td

# Main routine
if __name__ == "__main__":

    # Using parameter to enable Test mode for faster testing.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--test', help='Test run - less data', action='store_true')
    parser.add_argument(
        '-s', '--single', help='Use only one CPU core', action='store_true')
    parser.add_argument(
        '-l', '--logfile', help='Log console to file', action='store_true')
    parser.add_argument(
        '-r', '--rebuild', help='Reload learning data', action='store_true')
    parser.add_argument(
        '-p', '--progress', help='Displays progress information', action='store_true')
    parser.add_argument(
        '--inifile', default='tagbotft2.ini', type=str,
        dest='inifile', help='Different INI file')
    parser.add_argument(
        '--tag', default='Tag', type=str, dest='tag',
        help='Define tag column name')
    args = parser.parse_args()

    # Test run limits the amounts of input data
    if args.test:
        max_lines = 18000
        testrun = True
    # Set to 0 to read all initial data
    else:
        max_lines = 0
        testrun = False

    # Redirect output to files for easier debugging
    if args.logfile:
        sys.stdout = open('tagbotft.out', 'w')
        sys.stderr = open('tagbotft.err', 'w')
        
    # Set the number of CPU cores to be used
    if args.single:
        # Single core only
        cores = 1
    else:
        # Get the number of CPU cores available
        cores = os.cpu_count()

    # Set database and database path
    database_name = 'tagbotft2.sqlite'
    check_db = os.path.isfile('./' + database_name)

    # Use a different INI file
    if args.inifile:
        ini_file = args.inifile
    
    # Loading config    
    myDir = os.path.abspath(os.path.dirname(__file__)) + '/'
    config = configparser.ConfigParser()
    config.sections()
    config.read(myDir + ini_file)

    # Use config settings
    if config['Settings']['init_files']:
        initial_files = config['Settings']['init_files']
    else:
        initial_files = 'learn_data*.xlsx'
    
    if config['Settings']['new_files']:
        new_files = config['Settings']['new_files']
    else:
        new_files = 'input*.csv'
    
    wsheet = config['Settings']['wsheet']
    non_relevant_tag = config['Settings']['non_relevant_tag']
    exclude_file = config['Settings']['exclude_file']

    # Set the tag column name
    if config['Settings']['tag_cols']:
        tag_cols = config['Settings']['tag_cols'].split(",")
    else:
        tag_cols = args.tag

    # Maximum of non-tag columns to process
    max_cols = config.getint('Settings','max_cols')

    # Initialize initial_data Dataframe
    initial_data = pd.DataFrame()

    # Start message
    print()
    print("TagBot for Tables")
    print()
    print("Tag columns:\t\t", tag_cols)

    # If the database exists skip creating a new one and load from storage
    print("Checking database:\t", end="")

    # In case the database doesn't exist, start intial routine
    # Or set the rebuild parameter to force initial routine
    if not check_db or args.rebuild:
        print(" not found")
        print('Creating new database...')

        # Read initial tagged data from file
        initial_data = td.file_read(initial_files)

        # Exit if not data is available
        if len(initial_data) < 1:
            exit()

        # Process initial data
        processed_df = tp.initial_process(initial_data, tag_cols)

        # Save the initial tagged data to database
        td.save_data(processed_df, database_name, 'initial_data')

    # Otherwise just read the data from the database
    else:
        print(" found")
        processed_df = td.read_data(database_name, 'initial_data')    

    # Check if the data weights table already exists or 
    # rebuilding is forced
    if not td.check_table(database_name, 'data_column_names') or args.rebuild:

        # Read initial tagged data from file 
        if len(initial_data) < 1:
            initial_data = td.file_read(initial_files)

        data_col_names = pd.DataFrame(initial_data.columns.values, \
                        columns=['data_col_names'])
        td.save_data(data_col_names, database_name, 'data_column_names')
    else:
        data_col_names = td.read_data(database_name, 'data_column_names')
        
    # Check if the data weights table already exists or 
    # rebuilding is forced
    if not td.check_table(database_name, 'data_weights') or args.rebuild:

        # Calculate weights of unique words to tags
        weights_df = tp.generate_weights(processed_df)

        # Save the weights to database
        td.save_data(weights_df, database_name, 'data_weights')

    # Otherwise just read the data from the database
    else:
        weights_df = td.read_data(database_name, 'data_weights')

    # Check if the data probabilities table already exists or 
    # rebuilding is forced
    if not td.check_table(database_name, 'data_probabilities') or args.rebuild:
        probability_df = tp.generate_probabilities(weights_df)    
        td.save_data(probability_df, database_name, 'data_probabilities')

    # Otherwise just read the data from the database
    else:
        probability_df = td.read_data(database_name, 'data_probabilities')

    # Check if the data probabilities table already exists or 
    # rebuilding is forced
    if not td.check_table(database_name, 'data_multi_probabilities') or args.rebuild:
        multi_probability_df = probability_df[\
                                probability_df.duplicated(subset=['word', 'tag_col']) == True]
        td.save_data(multi_probability_df, database_name, 'data_multi_probabilities')
        
    # Otherwise just read the data from the database
    else:
        multi_probability_df = td.read_data(database_name, 'data_multi_probabilities')

    # Check if the data probabilities table already exists or 
    # rebuilding is forced
    if not td.check_table(database_name, 'data_unique_probabilities') or args.rebuild:
        unique_probability_df = tp.unique_probabilities(probability_df)    
        td.save_data(unique_probability_df, database_name, 'data_unique_probabilities')

    # Otherwise just read the data from the database
    else:
        unique_probability_df = td.read_data(database_name, 'data_unique_probabilities')
    
    unique_records = len(unique_probability_df[unique_probability_df['probability'] > 99.9])

    # Read to be tagged data from file 
    new_data = td.file_read(new_files)
    
    print("Cleaned duplicates:\t", len(tp.get_wrong_probabilities(probability_df)))                            
    print("Unique records:\t\t", unique_records)

    result_data = tp.process_new_data(new_data, len(tag_cols))

    td.write_results(result_data)

    print()
