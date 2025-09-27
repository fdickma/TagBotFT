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
import pandas as pd
import re
import multiprocessing as mp
from itertools import repeat
import tagbotft2_data as td
import __main__

# Check if string is a number
def is_valid_number(value):
    # Pattern for decimals or formatted numbers, including signed numbers
    pattern = re.compile(r'^[-]?[\d,.]+')
    # Return True of False  
    return bool(pattern.fullmatch(value))

def split_string(input_string):
    # Handle float numbers first
    if is_valid_number(input_string):
        words = [input_string]
    else:
        # Split the string using the specified delimiters
        words = re.split(r'[ ,;:.]+', str(input_string))
    # Return the words from splitting
    return words

# Check if string is a number
def is_valid_number(value):
    # Pattern for decimals or formatted numbers, including signed numbers
    pattern = re.compile(r'^[-]?[\d,.]+')
    # Return True of False  
    return bool(pattern.fullmatch(value))

def split_string(input_string):
    # Handle float numbers first
    if is_valid_number(input_string):
        words = [input_string]
    else:
        # Split the string using the specified delimiters
        words = re.split(r'[ ,;:.]+', str(input_string))
    # Return the words from splitting
    return words

def initial_split(split_df, proc_num, tag_cols):

    # Set the max_lines to the number of read lines from initial data
    max_lines = len(split_df)

    # Create data for the DataFrame
    entries = []
    tags = []
    cols = []
    datas = []

    # Create a dictionary with the columns and their corresponding values
    data_dict = {
        'entry': entries,
        'tag': tags,
        'tag_col': cols,
        'data': datas
    }

    # Initialize an empty Dataframe to store processed data
    processed_data = pd.DataFrame(data_dict)
   
    # Align the printout of the process number to equal length
    proc_digits = 3 - len(str(proc_num))
    proc_string = " " * proc_digits + str(proc_num)

    # Exit in case no tagging columns are provided
    if tag_cols == None:
        return

    # Initialize progress variable
    progress_old = 0
    progress_max = len(tag_cols) * max_lines
    progress_count = 0
    
    # Calculate a delay factor for printing the progress
    delay_factor = int(round((3 / progress_max * 777 * (proc_num / 2)), 0))
    if delay_factor > 11:
        delay_factor = 11
    delay_factor = delay_factor * 0.97711

    # Iterate over all given tags
    for tag_col in tag_cols:

        # Start counting with 0 lines
        line = 0

        # Iterate over each row in the DataFrame
        for index, row in split_df.iterrows():

            tag = row[tag_col]
    
            # Count lines until max_lines new_filesare reached
            line += 1
            if line > max_lines:
                continue

            # Iterate the tag counter for progress calculation
            progress_count += 1

            # Process all columns except the tagging column
            # split_strings = {}
            for col_name in split_df.columns:
                # Only process non-tag columns 
                if col_name not in tag_cols:

                    # Read the column data as string
                    original_string = str(row[col_name])

                    # Split the string into single words
                    split_words = split_string(original_string)
                    
                    # Clear the tags 
                    temp_tags = []
                    temp_tag_col = []
                    temp_strings = []

                    # Iterate over the single words list and combine word with tag
                    for word in split_words:
                        # Use the tag as is
                        temp_tags.append(tag)
                        # Use the tag column as is
                        temp_tag_col.append(tag_col)
                        # Convert the words to lower case for easier comparison later
                        temp_strings.append(str(word).lower())
                                    
                    # In the DataFrame also add the original index for clear distinction
                    temp_data = {
                        'entry': index,
                        'tag': temp_tags,
                        'tag_col': temp_tag_col,
                        'data': temp_strings
                    }
                    temp_df = pd.DataFrame(temp_data)
                    
                    # Append the data from the initial data line to the processed DataFrame
                    processed_data = pd.concat([processed_data, temp_df], ignore_index=True)

            # Print progress
            if __main__.args.progress:
                # Calculate progress and progress bar
                progress = int(round(progress_count / progress_max * 100, 0))
                bars = int(round(progress / 10, 0))
                spaces = 10 - bars

                # Only print when update limit is exceeded
                if progress > (progress_old + delay_factor):
                    progress_old = progress
                    
                    # First clear the previous printout
                    out_string = "\r" + (" " * 48) + "\r"
                    sys.stdout.write(out_string)
                    sys.stdout.flush()

                    # Print the progress
                    out_string = "\rProcess: " + str(proc_string) + " |  Progress: " + \
                        str(progress) + " %  [" + ("#" * bars) + (" " * spaces) + "]\r"
                    sys.stdout.write(out_string)
                    sys.stdout.flush()

    return processed_data

def initial_process(initial_data, tag_cols=None):
    
    # Create data for the processing DataFrame
    entries = []
    tags = []
    cols = []
    datas = []

    # Create a dictionary with the columns and their corresponding values
    data_dict = {
        'entry': entries,
        'tag': tags,
        'tag_col': cols,
        'data': datas
    }

    # Initialize an empty Dataframe to store processed data
    processed_data = pd.DataFrame(data_dict)

    # Building test data, the data to tagged
    # One process gets all data
    if __main__.cores < 2:
        # One process means all data for that process and one process only
        processed_data = initial_split(initial_data, 1, tag_cols)

    else:
        # Chunk it in equal parts
        chunk_size = round(len(initial_data) / __main__.cores)
        chunks = [initial_data.iloc[i:i+chunk_size] \
            for i in range(0, len(initial_data), chunk_size)]

        # Define a list of processes form a range
        proc_num = [*range(1, __main__.cores + 1)]
    
        # Run tagging as Pool parallel processes;
        pool = mp.Pool(processes = __main__.cores)
    
        # Define the processing queues with function to call and data together
        pqueue = pool.starmap(initial_split, zip(chunks, proc_num, \
                            repeat(tag_cols)))
        pool.close()
        pool.join()
    
        # Iterate the Pool segments for results to build the complete results
        for q in pqueue:
            try:
                processed_data = pd.concat([processed_data, q], ignore_index=True)
            except:
                processed_data = q

    if tag_cols == None:
        processed_data = processed_data.drop('tag', axis=1)

    print()

    return processed_data

# Actually doing the counting of words/tags with one process 
def calculate_weights(unique_words, unique_tags, unique_data, proc_num):
    # Create data for the DataFrame
    tags = []
    cols = []
    words = []
    counts = []

    # Create a dictionary with the columns and their corresponding values
    count_dict = {
        'tag': tags,
        'tag_col': cols,
        'word': words,
        'count': counts
    }
    weights_df = pd.DataFrame(count_dict)

    # Start counting with 0 lines
    line = 0

    # Align the printout of the process number to equal length
    proc_digits = 3 - len(str(proc_num))
    proc_string = " " * proc_digits + str(proc_num)

    # Initialize progress variable
    progress_old = 0
    
    # Calculate the possible combinations
    total_combinations = len(unique_tags) * len(unique_words)

    # Calculating a delay for printing the progress
    delay_factor = int(round((3 / total_combinations * 777 * (proc_num / 2)), 0))
    if delay_factor > 11:
        delay_factor = 11
    delay_factor = delay_factor * 0.97711
    
    # Iterate over the list of words
    for word in unique_words:

        # For each word iterate over all given tags
        for index, tag_row in unique_tags.iterrows():

            # Extract the tag and its column name
            tag = tag_row['tag']
            tag_col =tag_row['tag_col']

            # Count the number of rows with word and tag 
            count = len(unique_data[(unique_data["tag"] == tag) & \
                        (unique_data["tag_col"] == tag_col) & \
                        (unique_data["data"] == word)])
            
            # If there is at least one row found
            if count > 0 and len(word) > 0:

                # Add a new row to the weights Dataframe with 
                # word, tag and the counted number
                weights_df.loc[len(weights_df)] = {
                    'tag': tag,
                    'tag_col': tag_col,
                    'word': word,
                    'count': count
                }
            line += 1

        # Print progress
        if __main__.args.progress:
            # Calculate the progress and progress bar
            progress = int(round(line / total_combinations * 100, 0))
            bars = int(round(progress / 10, 0))
            spaces = 10 - bars

            # Only print when update limit is exceeded
            if progress > (progress_old + delay_factor):
                progress_old = progress
                
                # First clear the previous printout
                out_string = "\r" + (" " * 48) + "\r"
                sys.stdout.write(out_string)
                sys.stdout.flush()

                # Print the progress
                out_string = "\rProcess: " + str(proc_string) + " |  Progress: " + \
                     str(progress) + " %  [" + ("#" * bars) + (" " * spaces) + "]\r"
                sys.stdout.write(out_string)
                sys.stdout.flush()

    # Return the chunk of weights
    return weights_df

# Generate weights by counting occurrences of word/tag combinations
def generate_weights(unique_data):
    print("Generating weights:")

    # Create data for the DataFrame
    tags = []
    cols = []
    words = []
    counts = []

    # Create a dictionary with the columns and their corresponding values
    count_dict = {
        'tag': tags,
        'tag_col': cols,
        'word': words,
        'count': counts
    }

    copy_unique = unique_data[['tag', 'tag_col']].copy()
    unique_tags = copy_unique[['tag', 'tag_col']].drop_duplicates()
    unique_words = unique_data['data'].unique()
    weights_df = pd.DataFrame(count_dict)

    # Building test data, the data to tagged
    # One process gets all data
    if __main__.cores < 2:
        # One process means all data for that process and one process only
        weights_df = calculate_weights(unique_words, unique_tags, unique_data, 1)

    else:
        # Split the list of words into equal chunks according to the number 
        # of CPU cores available        
        chunks = np.array_split(unique_words, __main__.cores)

        # Define a list of processes form a range
        proc_num = [*range(1, __main__.cores + 1)]
    
        # Run tagging as Pool parallel processes;
        pool = mp.Pool(processes = __main__.cores)
    
        # Define the processing queues with function to call and data together
        pqueue = pool.starmap(calculate_weights, zip(chunks, repeat(unique_tags), \
                    repeat(unique_data), proc_num))
        pool.close()
        pool.join()
    
        # Iterate the Pool segments for results to build the complete results
        for q in pqueue:
            try:
                weights_df = pd.concat([weights_df, q], ignore_index=True)
            except:
                weights_df = q

    print()

    # Return non-duplicate weight results
    return weights_df.drop_duplicates() 

# Calculating probabilities from weights as absolute and relative measures
# Absolute measures in case of a 100% relation between a word and a tag,
# relative measures in case of relations of a word to several tags. In the
# latter case a relative probability is being calculated.
def generate_probabilities(weights_df):

    # Make sure not to work on the original weights Dataframe
    probability_df = weights_df.copy()

    # Add the probability column with a value of 100 for all rows
    probability_df['probability'] = 100

    # Identify all rows with words that occur at least twice
    duplicated_df = weights_df[weights_df[['word']].duplicated() == True]

    # Iterate over all words
    for word in duplicated_df['word'].unique():

        # Get a temporary Dataframe with all entries of the word
        single_df = duplicated_df[duplicated_df['word'] == word]

        # Only investigate further if there is more than one entry
        if len(single_df) > 1:
            
            # Get a list of the tags for the word 
            temp_tags = single_df['tag']
            
            # Calculate the total sum of all counted occurrences
            temp_max = single_df['count'].sum()

            # Iterate over the tags
            for temp_tag in temp_tags:

                # For each tag get the individual count
                temp_val = single_df[single_df['tag'] == temp_tag]['count'].iloc[0]

                # And calculate the relative occurrences to the total count sum
                temp_prob = int(round(temp_val / temp_max * 100, 0))

                # Change the probebilities for the rows in the probability Dataframe
                # based on the index reference
                probability_df.at[single_df[single_df['tag'] == temp_tag].index[0], 
                        'probability'] = temp_prob
    
    # Return the Dataframe with the probabilities
    return probability_df

# Identify not unique probabilities which would lead to unclear decisions
def get_wrong_probabilities(probability_df):
    # Group the Dataframe column with words and count the occurrencies in a Series
    wrong_indicators = probability_df.groupby('word')['word'].count()
    # Return the Series entries with a greater value than 1
    return wrong_indicators[wrong_indicators > 1]

# Clean the probabilities Dataframe and don't keep the duplicate values at all
def unique_probabilities(probability_df):
    return probability_df.drop_duplicates(subset=['word', 'tag_col'], keep=False)

# Process Dataframe with new data
def process_new_data(new_data_df, tag_count):
    
    # Initialize line number
    line = 0

    # Set column list
    new_cols = __main__.tag_cols
    new_cols.append("TB_qual")

    # Initialize the Dataframe for tag and tag_col
    new_tagged_df = pd.DataFrame(columns=new_cols)

    # Initialize temporary Dataframe for tagging columns
    temp_new_data = pd.DataFrame(columns=new_data_df.columns.values)

    # Initialize maximum number of rows to process in test mode
    if __main__.max_lines > 1000:
        max_lines = __main__.max_lines / 100
    else:
        max_lines = 100

    max_lines = 1000000

    # Initialize progress variable
    proc_num = 0
    progress_old = 0
    progress_max = len(new_data_df)
    progress_count = 0
    
    # Align the printout of the process number to equal length
    proc_digits = 3 - len(str(proc_num))
    proc_string = " " * proc_digits + str(proc_num)
    
    # Calculate a delay factor for printing the progress
    delay_factor = int(round((3 / progress_max * 777 * (proc_num / 2)), 0))
    if delay_factor > 11:
        delay_factor = 11
    delay_factor = delay_factor * 0.97711

    # Iterate over the Dataframe
    for index, new_row in new_data_df.iterrows():

        # Increase the line number
        line += 1

        # If in test mode reduce the number of lines to process
        if __main__.args.test == True:
            if line > max_lines:
                continue

        # Generate the row as one string
        new_line = ""
        for new_item in new_row:
            if len(new_line) < 1:
                new_line = str(new_item)
            else:
                new_line = new_line + " " + str(new_item)

        # Split the row into a list of words like in the training data
        list_words = re.split(r'[ ,;:.]+', str(new_line).lower())

        # Check if trained unique words are in the list of words
        filtered_df = __main__.unique_probability_df[\
            __main__.unique_probability_df['word'].isin(list_words)].copy()
        filtered_len = len(filtered_df)
        tag_quality = 100

        if filtered_len == 0:
            filtered_df = __main__.multi_probability_df[\
                __main__.multi_probability_df['word'].isin(list_words)].copy()
            filtered_df = filtered_df.sort_values(by=['probability', 'word'], \
                ascending=False)
            filtered_len = len(filtered_df)
            tag_quality = filtered_df['probability'].values[:1]

        # Append the new data to the temporary Dataframe
        temp_new_data.loc[index] = new_data_df.loc[index]

        # In case of exactly the number of results as tag columns given,
        # it is assumed that there is an exact result
        # In case of more results than tag columns given,
        # it is assumed that there is not an exact result
        if filtered_len >= tag_count:
            filtered_df = filtered_df[["tag", "tag_col"]].drop_duplicates()
            for i, t in filtered_df.iterrows():
                new_tagged_df.loc[index, t["tag_col"]] = t["tag"]
            new_tagged_df.loc[index, "TB_qual"] = int(tag_quality)
        # In case of zero results, it is assumed that there is no result
        elif filtered_len == 0:
            for t in __main__.tag_cols:
                new_tagged_df.loc[index, t] = np.nan
            new_tagged_df.loc[index, "TB_qual"] = 0
        # In other cases, it is assumed that there is no result,
        # and leave a particular quality marker of 1
        else:
            for t in __main__.tag_cols:
                new_tagged_df.loc[index, t] = np.nan
            new_tagged_df.loc[index, "TB_qual"] = 1

        # Print progress
        if __main__.args.progress:
            # Calculate progress and progress bar
            progress = int(round(line / progress_max * 100, 0))
            bars = int(round(progress / 10, 0))
            spaces = 10 - bars

            # Only print when update limit is exceeded
            if progress > (progress_old + delay_factor):
                progress_old = progress
                
                # First clear the previous printout
                out_string = "\r" + (" " * 48) + "\r"
                sys.stdout.write(out_string)
                sys.stdout.flush()

                # Print the progress
                out_string = "\rProcess: " + str(proc_string) + " |  Progress: " + \
                    str(progress) + " %  [" + ("#" * bars) + (" " * spaces) + "]\r"
                sys.stdout.write(out_string)
                sys.stdout.flush()

    for t in __main__.tag_cols:
        temp_new_data[t] = new_tagged_df[t]

    temp_new_data = temp_new_data[list(__main__.data_col_names['data_col_names'])]
    temp_new_data["TB_qual"] = new_tagged_df["TB_qual"]

    print()

    return temp_new_data