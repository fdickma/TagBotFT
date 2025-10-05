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

    # Set inistial start time
    start_time = time.time()

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
        '-i', '--inifile', default='tagbotft2.ini', type=str,
        dest='inifile', help='Different INI file')
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

    # Set data sources/directories
    init_dir = config['Settings']['init_dir']
    input_dir = config['Settings']['input_dir']
    output_dir = config['Settings']['output_dir']

    # Retrieve home path for ~ replacement
    from pathlib import Path
    home = str(Path.home())

    # Replace ~ and add final /
    if len(init_dir) > 0:
        init_dir = init_dir.replace("~", home) + "/"
    if len(input_dir) > 0:
        input_dir = input_dir.replace("~", home) + "/"
    if len(output_dir) > 0:
        output_dir = output_dir.replace("~", home) + "/"
    initial_files = init_dir + initial_files
    new_files = input_dir + new_files

    # Set database and database path
    database_name = init_dir + 'tagbotft2.sqlite'
    check_db = os.path.isfile(database_name)

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

        # Store initial data in plain form without tag columns
        plain_data = initial_data.copy()
        td.save_data(plain_data, database_name, 'plain_initial_data')

        # Exit if not data is available
        if len(initial_data) < 1:
            print("No input data found...")
            exit(1)

        # Process initial data
        processed_df = tp.initial_process(initial_data, tag_cols)

        # Save the initial tagged data to database
        td.save_data(processed_df, database_name, 'initial_data')

    # Otherwise just read the data from the database
    else:
        print(" found")
        processed_df = td.read_data(database_name, 'initial_data')    

    #  Create the plain data table if it doesn't exist  
    if not td.check_table(database_name, 'plain_initial_data'):

        if len(initial_data) < 1:
            # Read initial tagged data from file
            initial_data = td.file_read(initial_files).astype(str)

        plain_data = initial_data.copy()
        td.save_data(plain_data, database_name, 'plain_initial_data')
    else:
        plain_data = td.read_data(database_name, 'plain_initial_data')

    # Exit if the tag columns don't exist in the initial data
    if not set(tag_cols).issubset(plain_data.columns):
        print("Tagging columns are missing in initial data...")
        exit(1)

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
    
    print("Cleaned duplicates:\t", len(tp.get_wrong_probabilities(probability_df)))                            
    print("Unique records:\t\t", unique_records)

    # Read new to be tagged data from file 
    new_data = td.file_read(new_files)

    # Remove existing rows from new data, as they are already processed
    old_data, new_data = tp.get_existing(new_data, plain_data)

    # Write old data to file
    td.write_results(old_data, output_dir + "results_old.xlsx")

    # Process new data
    result_data = tp.process_new_data(new_data, len(tag_cols))

    # Write results to file
    td.write_results(result_data, output_dir + "results_new.xlsx")

    # Print the total runtime
    print('Execution took:\t\t {} (h:min:s, wall clock time).' \
        .format(datetime.timedelta(seconds=round(time.time() - start_time))))

    print()
