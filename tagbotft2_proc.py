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
from difflib import SequenceMatcher
import tagbotft2_data as td
import __main__

# Check if string is a number
def is_valid_number(value):
    # Pattern for decimals or formatted numbers, including signed numbers
    pattern = re.compile(r'^[-]?[\d,.]+')
    # Return True of False  
    return bool(pattern.fullmatch(value))

# Check if string is a number
def is_valid_number(value):
    # Pattern for decimals or formatted numbers, including signed numbers
    pattern = re.compile(r'^[-]?[\d,.]+')
    # Return True of False  
    return bool(pattern.fullmatch(value))

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
    progress_max = len(unique_words) * len(unique_tags)
    
    # Calculate the possible combinations
    total_combinations = len(unique_tags) * len(unique_words)
    if total_combinations < 0:
        return weights_df

    # Set inistial start time
    existing_start = time.time()

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
        # Calculate the progress and progress bar
        progress = int(round(line / total_combinations * 100, 0))

        # Only print when update limit is exceeded
        if progress > (progress_old + delay_factor):
            # Time difference from start of process to now
            timediff = datetime.timedelta(seconds=round(time.time() \
                                                        - existing_start))
            
            # Calculate the remaining seconds for tagging to finish
            timeremain = datetime.timedelta(\
                                            seconds=round(((time.time() - \
                                            existing_start) / line) \
                                            * (progress_max - line)))

            progress_old = progress
            
            # First clear the previous printout
            out_string = "\r" + (" " * 40) + "\r"
            sys.stdout.write(out_string)
            sys.stdout.flush()

            # Print the progress
            out_string = "\rProcess: " + str(proc_string) + " |  Progress: " + \
                str(progress) + " %  |  time: " + str(timediff) \
                + ' elapsed, ' + str(timeremain) + ' remaining\r'
            sys.stdout.write(out_string)
            sys.stdout.flush()

    # Return the chunk of weights
    return weights_df

# Process initial data with DataFrame much faster than iterating rows
# However, it delivers slightly different substrings
def initial_process(plain_data):

    # List of data column names, exlude tag columns
    data_cols = plain_data.columns.tolist()
    only_data_cols = [x for x in data_cols if x not in __main__.tag_cols]

    # Initialize DataFrame 
    init_df = pd.DataFrame()

    # Combine all data columns to one column with a space separator
    init_df["word"] = plain_data[only_data_cols].agg(' '.join, axis=1)
    
    # Initialize tagging columns in the DataFrame
    for col in __main__.tag_cols:
        init_df[col] = plain_data[col]
    
    # Add the entry ID
    init_df['entry'] = init_df.index
    
    # Split the words in the word column by the seperator symbols
    init_df['word'] = init_df['word'].str.split(r'[ ,;:]+')
    
    # Explode the words into one line per word and lower case for all characters
    init_df = init_df.explode('word')
    init_df['word'] = init_df['word'].str.lower()

    # Removing duplicate entries
    init_df = init_df.drop_duplicates()

    # Change the tags and tag column names from rows to standard columns
    init_df = init_df.melt(id_vars=["entry","word"],var_name="tag_col",
                value_name="tag").sort_values(['word']).reset_index(drop=True)
    
    # Remove rows with empty or only one character words
    init_df = init_df.drop(init_df[init_df['word'].str.len() < 2].index)
    
    # Remove words that contain only numbers like small prices
    regex = '^[-]{0,1}[0-9.]{2,5}$'
    init_df = init_df.drop(init_df[init_df['word'].str.match(regex)].index)

    # Remove words that contains only special characters
    regex = '^["!()#\'?=$&/.%§-]{2,}$'
    init_df = init_df.drop(init_df[init_df['word'].str.match(regex)].index)

    # Remove long sequences of special characters
    regex = '["!()#\'?=$&/.%§-]{2,}'
    init_df['word'] = init_df['word'].replace(to_replace=regex, value='', regex=True)

    # Remove beginning and ending characters
    init_df['word'] = init_df['word'].str.strip('&"()#*+»«')
    init_df['word'] = init_df['word'].str.rstrip('-.')
    init_df['word'] = init_df['word'].str.strip()

    # Remove rows with empty or only one character words after cleaning
    init_df = init_df.drop(init_df[init_df['word'].str.len() < 2].index)

    # Removing duplicate entries
    init_df = init_df.drop_duplicates()

    # Set the right data format for the word column
    init_df['word'] = init_df['word'].astype(str)

    return init_df

# Generate weights by counting occurrences of word/tag combinations
# Faster version
def generate_weights(unique_data):
    weights_df = unique_data.groupby(by=["tag","tag_col","word"], \
        as_index=False)["entry"].count()
    weights_df = weights_df.rename(columns={"entry": "count"})
    return weights_df

# Split a list of values into equally sized chunks
def chunk_list(data_list, chunks):
    for i in range(0, len(data_list), chunks):
        yield data_list[i:i + chunks]

# Initial data is being split into single words in a separate table
def generate_probabilities(weights_df):

    print("Generating probabilities...")

    # Make sure not to work on the original weights Dataframe
    probability_df = weights_df.copy()

    # Identify all rows with words that occur at least twice
    # duplicates_df = weights_df[weights_df[['word']].duplicated() == True]
    duplicates_df = probability_df[probability_df.duplicated(\
                    subset=['tag_col','word'], \
                    keep=False) == True].copy()

    # Adding probabilities to results with multiple tags
    duplicates_df['probability'] = 0
    duplicates_df['temp'] = duplicates_df.groupby(['tag_col', 'word'])['count'].transform('sum')
    if len(duplicates_df['temp']) < 1:
        return
    duplicates_df['probability'] = round(duplicates_df['count'] / duplicates_df['temp'] * 100, 0)
    duplicates_df = duplicates_df.drop(['temp'], axis=1)

    # The probability of unique results is 100%
    single_df = probability_df[probability_df.duplicated(\
                    subset=['tag_col','word'], \
                    keep=False) == False].copy()
    single_df['probability'] = 100

    # Set the right data format for the word column
    duplicates_df['word'] = duplicates_df['word'].astype(str)
    single_df['word'] = single_df['word'].astype(str)

    return single_df, duplicates_df

# Identify not unique probabilities which would lead to unclear decisions
def get_wrong_probabilities(probability_df):
    # Group the Dataframe column with words and count the occurrencies in a Series
    wrong_indicators = probability_df.groupby('word')['word'].count()
    # Return the Series entries with a greater value than 1
    return wrong_indicators[wrong_indicators > 1]

# Clean the probabilities Dataframe and don't keep the duplicate values at all
def unique_probabilities(probability_df):
    return probability_df.drop_duplicates(subset=['word', 'tag_col'], keep=False)

def generate_matrix(plain_data):

    # First make a copy of the plain data
    plain_str_df = plain_data.copy()

    # Create data for the DataFrame
    words = []

    # Create a dictionary with the columns and their corresponding values
    matrix_dict = {
        'word': words
    }

    # Create the DataFrame
    matrix_df = pd.DataFrame(matrix_dict)

    # Define lists with names of tagging columns and data columns 
    plain_data_cols = list(__main__.data_col_names['data_col_names'])
    tag_cols = __main__.tag_cols
    data_cols = [x for x in plain_data_cols if not x in tag_cols]

    # Iterate over the data colums and add them to the matrix DataFrame. 
    # All columns are integrated into one single column
    for data_col in data_cols:
        if pd.api.types.is_string_dtype(plain_str_df[data_col]):
            if len(matrix_df) < 1:
                matrix_df['word'] = plain_str_df[data_col]
            else:
                matrix_df['word'] = matrix_df['word'] + \
                                        " " + plain_str_df[data_col]
    
    # Finally add the tagging columns
    for tag_col in tag_cols:
        matrix_df[tag_col] = plain_str_df[tag_col]

    # Make a copy of the index column to a column named row for later
    # accessing the rows after exploding the data column
    matrix_df['row'] = matrix_df.index

    # Split the strings in the data column
    matrix_df['word'] = matrix_df['word'].str.split(r'[ ,;:]+')
    
    # Explode and lower case the data column 
    matrix_df = matrix_df.explode('word')
    matrix_df['word'] = matrix_df['word'].str.lower()
    
    # Finally reset the index to make each line clearly accessible
    matrix_df = matrix_df.reset_index()

    # Remove words that contains only special characters
    regex = '^["!()#\'?=$&/*.§-]{2,}$'
    matrix_df = matrix_df.drop(matrix_df[matrix_df['word'].str.match(regex)].index)

    # Remove long sequences of special characters
    regex = '["!()#\'?=$&/*.§-]{2,}'
    matrix_df['word'] = matrix_df['word'].replace(to_replace=regex, value='', regex=True)

    # Remove beginning and ending characters
    matrix_df['word'] = matrix_df['word'].str.strip('"()#*&+»«')
    matrix_df['word'] = matrix_df['word'].str.rstrip('-.')

    # Add the subsequent word to a separate column for each initial
    # data row
    matrix_df['next_word'] = matrix_df['word'].shift(-1)

    # Generate a temporary Dataframe to identify the last row of a set of words
    # from the initial data rows 
    rows_df = pd.DataFrame()
    rows_df['row'] = matrix_df['row'].copy()
    rows_df['idx'] = rows_df.index

    # Identify the last rows of each initial row
    last_rows = rows_df.groupby('row').last()

    # Empty the next word for the last rows since they do not have one in
    # the initial data set
    matrix_df.loc[last_rows['idx'], 'next_word'] = ""

    # Clean the matrix Dataframe from index, row columns, and short words
    matrix_prob_df = matrix_df.copy()
    matrix_prob_df = matrix_prob_df.drop(columns=['index'])
    matrix_prob_df = matrix_prob_df[matrix_prob_df['next_word'] != ""]
    matrix_prob_df = matrix_prob_df[matrix_prob_df['word'].str.len() > 1]
    matrix_prob_df = matrix_prob_df[matrix_prob_df['next_word'].str.len() > 1]

    # Change the tags and tag column names from rows to standard columns
    matrix_prob_df = matrix_prob_df.melt(id_vars=["word","next_word","row"], var_name="tag_col",
                value_name="tag").sort_values(['word']).reset_index(drop=True)

    # Count the unique occurrences in the row column and rename that column
    matrix_prob_df = matrix_prob_df.groupby(by=["tag","tag_col","word", \
        "next_word"], as_index=False)["row"].count()
    matrix_prob_df = matrix_prob_df.rename(columns={"row": "count"})

    # Compute the probabilities of each unique entry
    matrix_prob_df['temp'] = matrix_prob_df.groupby(['tag_col', 'word', 'next_word'])\
        ['count'].transform('sum')
    matrix_prob_df['probability'] = round(matrix_prob_df['count'] / matrix_prob_df['temp']\
         * 100, 0)
    matrix_prob_df = matrix_prob_df.drop(['temp','count'], axis=1)

    return matrix_prob_df

# Process a wordlist against trained data from a Dataframe
def calculate_similarity(word_list, tag_col):

    # Create data for the DataFrame
    tags = []
    cols = []
    words = []
    probability = []

    # Create a dictionary with the columns and their corresponding values
    similar_dict = {
        'tag': tags,
        'tag_col': cols,
        'word': words,
        'probability': probability
    }
    similar_df = pd.DataFrame(similar_dict)

    # In case the word_list is empty, return an empty Dataframe
    if len(word_list) < 1:
        return similar_df

    # Create a testing dataframe to check against
    test_df = __main__.unique_probability_df[
        __main__.unique_probability_df['tag_col'] == tag_col].copy()

    # Iterate over the word_list
    for word in word_list:
        
        # Generate a DataFrame that has only words of possible similar length
        test_len_df = test_df.loc[(test_df['word'].str.len() > len(word) - 2) &\
            (test_df['word'].str.len() < len(word) + 2)]

        # Generate a DataFrame that has a best matching selection
        preselect_df = pd.DataFrame()
        for i in range(80, 10, -10):
            if len(preselect_df) < 1:
                test_rate = i / 100
                preselect_df = test_len_df.loc[test_len_df.apply(lambda \
                    x: SequenceMatcher(\
                    None, word, x.word).ratio() > test_rate, axis=1)]    

        # Iterate over the trained data from a Dataframe
        #for index, df_word in __main__.unique_probability_df[\
        #    __main__.unique_probability_df['tag_col'] == tag_col].iterrows():
        for index, df_word in preselect_df.iterrows():
            
            # Make sure both variables are strings
            str_df_word = str(df_word['word'])
            str_word = str(word)

            # In case the word length is zero, don't use it
            if len(str_word) < 1:
                continue

            # Compare the length of both variables
            length_comp = len(str_df_word) / len(str_word)

            # Initialize similarity to 0
            s = 0

            # Only process in case the comparison is between 95% and 105% and both
            # variables are at least 5 characters long.
            # In that way to short and not similar results are takin into account.
            if length_comp > 0.95 and length_comp < 1.05 and \
            len(str_word) > 4 and len(str_df_word) > 4:

                # Calculate a Levenshtein distance for both variables as ratio
                s = SequenceMatcher(None, str_df_word, str_word, autojunk=True).ratio()
            
            # Only append if the Levenshtein distance ratio is greate than 0.3
            if s > 0.3:
                similar_df.loc[len(similar_df)] = {
                    'tag': df_word['tag'],
                    'tag_col': df_word['tag_col'],
                    'word': df_word['word'],
                    'probability': int(s * 100)
                }

            if index % 2 == 0:
                activity_sign = "-" 
            else:
                activity_sign = "|" 

            # Print indicator of activity
            out_string = "\b" + activity_sign
            sys.stdout.write(out_string)
            sys.stdout.flush()


    if len(similar_df) > 0:
        similar_df = similar_df.sort_values(by=['probability'], ascending=False)
        #print(similar_df)

    # Return the complete Dataframe with all similarities
    return similar_df

# Function to get the most similar word from a Dataframe column 
def get_most_similar(word_list, tag_col):

    # Financial numbers
    pattern = re.compile(r'^[-]{0,1}[0-9]{1,20}[.]{1}[0-9]{1,2}$')
    temp_list = list(filter(lambda s: not pattern.search(s), word_list))
    if len(temp_list) > 0:
        word_list = temp_list

    # One process gets all data
    if __main__.cores < 2:
        # One process means all data for that process and one process only
        processed_data = calculate_similarity(word_list, tag_col)

    else:
        # Split the list of words into equal chunks according to the number 
        # of CPU cores available        
        chunks = np.array_split(word_list, __main__.cores)
    
        # Run tagging as Pool parallel processes;
        pool = mp.Pool(processes = __main__.cores)
    
        # Define the processing queues with function to call and data together
        pqueue = pool.starmap(calculate_similarity, zip(chunks, repeat(tag_col)))
        pool.close()
        pool.join()
    
        # Iterate the Pool segments for results to build the complete results
        for q in pqueue:
            try:
                processed_data = pd.concat([processed_data, q], ignore_index=True)
            except:
                processed_data = q

    processed_data = processed_data.sort_values(by=['probability', 'word'], \
                ascending=False)

    # Return the the most similar result
    processed_data = processed_data.head(1)
    return processed_data

# Process Dataframe with new data
def process_new_data(new_data_df, tag_count):
    
    print("Tagging new data...")

    # Test if there is any new data
    if len(new_data_df) < 1:
        return pd.DataFrame()

    # Initialize line number
    line = 0

    # Set column list
    new_cols = __main__.tag_cols
    # new_cols.append("TB_qual")

    # Initialize the Dataframe for tag and tag_col
    new_tagged_df = pd.DataFrame(columns=new_cols)

    # Initialize temporary Dataframe for tagging columns
    temp_new_data = pd.DataFrame(columns=new_data_df.columns.values)

    # Set inistial start time
    existing_start = time.time()

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

    # Create a dictionary with the columns and their corresponding values
    next_df_dict = {
        'tag': [],
        'tag_col': [],
        'word': [],
        'next_word': [],
        'probability': []
    }
    
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
        list_words = re.split(r'[ ,;:]+', str(new_line).lower())

        # Remove beginning and ending characters 
        cleaned_word_list = [x for x in list_words if not re.search(r'^["!()#\'?=$&/*.§-]{2,}$', x)]

        # Remove long sequences of special characters
        pattern = re.compile(r'["!()#\'?=$&/*.§-]{2,}')
        cleaned_word_list = [pattern.sub('', string) for string in list_words]

        # Remove beginning and ending characters
        cleaned_word_list = [x.strip('"()#*&+»«') for x in cleaned_word_list]
        cleaned_word_list = [x.rstrip('-.') for x in cleaned_word_list]
        cleaned_word_list = [x.strip() for x in cleaned_word_list]
        if len(cleaned_word_list) > 0:
            delta = list(set(list_words) - set(cleaned_word_list))
            list_words = cleaned_word_list
            if len(delta) > 0 and __main__.args.debug:
                print("Cleaned words:", delta)

        # Filter less ideal entries
        # Empty strings
        list_words = [x for x in list_words if x.strip()]

        # Strings with only one character 
        list_words = [x for x in list_words if len(str(x)) > 1]
        
        # Financial numbers
        pattern = re.compile(r'^[-]{0,1}[0-9]{1,4}[.]{0,1}[0-9]{0,2}$')
        filtered_word_list = list(filter(lambda s: not pattern.search(s), list_words))
        if len(filtered_word_list) > 0:
            list_words = filtered_word_list

        # Date numbers
        pattern = re.compile(r'^[0-9]{1,4}[/]{1}[0-9]{1,4}$')
        filtered_word_list = list(filter(lambda s: not pattern.search(s), list_words))
        if len(filtered_word_list) > 0:
            list_words = filtered_word_list

        # Date numbers
        pattern = re.compile(r'^[0-9]{1,4}[-.]{1}[0-9]{1,4}[-.]{1}[0-9]{1,4}$')
        filtered_word_list = list(filter(lambda s: not pattern.search(s), list_words))
        if len(filtered_word_list) > 0:
            list_words = filtered_word_list

        # Remove words with less than 3 characters, but only if the remainder
        # has data left. This list is only used for testing against multi
        # result data
        multi_words = [x for x in list_words if len(x) > 2]
        if len(multi_words) < 1:
            multi_words = list_words

        pattern = re.compile(r'^[-]{0,1}[0-9]{1,10}[,.]{0,1}[0-9]{1,2}$')
        filtered_multi_list = list(filter(lambda s: not pattern.search(s), multi_words))
        if len(filtered_word_list) > 0:
            multi_words = filtered_multi_list

        # Initialize tag quality for current row
        tag_quality = 0

        # Get the filtered number of words in the row
        word_list_len = len(list_words)

        if __main__.args.debug:
            print("Word list:", list_words)

        # For each row iterate over the tagging columns
        for tag_col in __main__.tag_cols:

            tags_len = 0
            filtered_len = 0
            tags_df = pd.DataFrame()
            tags_df_uni = pd.DataFrame()
            tags_df_matrix = pd.DataFrame()
            tags_df_mult = pd.DataFrame()
            tags_df_simil = pd.DataFrame()
            tags_prob = 0

            # Check if trained unique words are in the list of words
            filtered_df = __main__.unique_probability_df[\
                __main__.unique_probability_df['word'].isin(list_words)].copy()
            filtered_df = filtered_df[filtered_df['tag_col'] == tag_col]

            # Omit single numbers as the sole basis for tagging
            if len(filtered_df[filtered_df['word'].str.isnumeric() == True]) < 2 and \
                len(filtered_df[filtered_df['word'].str.isnumeric() == False]) <= 1:
                filtered_df = filtered_df[filtered_df['word'].str.isnumeric() == False]

            # Only process further if there are elements to process
            if len(filtered_df) > 0:
                filtered_df = filtered_df.sort_values(by=['probability', 'count', 'word'], \
                    ascending=False)
                tags_df_uni = filtered_df[["tag", "tag_col", "probability"]].drop_duplicates(\
                    subset=["tag", "tag_col"])
                tags_df_uni = tags_df_uni.sort_values(by=['probability'], ascending=False)

            # Append to the all tag columns spanning variable if unique elements
            # have been identified 
            if len(tags_df_uni) > 0:
                tags_df = pd.concat([tags_df, tags_df_uni], ignore_index=True)
                if __main__.args.debug:
                    print("Uni:", tags_df_uni)
                    print()

            # Check if trained words and subsequent words tags are in the list of words
            # by using the matrix data
            if len(tags_df_uni) < 1:
                next_df = pd.DataFrame(next_df_dict)
                filtered_df = __main__.matrix_probability_df[\
                    __main__.matrix_probability_df['word'].isin(list_words) & \
                    __main__.matrix_probability_df['next_word'].isin(list_words)].copy()
                filtered_df = filtered_df[filtered_df['tag_col'] == tag_col]
                for w in range(0, word_list_len):
                    if w + 1 < word_list_len:
                        for i, filter_row in filtered_df.iterrows():
                            if (list_words[w] == filter_row['word']) & \
                                (list_words[w + 1] == filter_row['next_word']) & \
                                (str(list_words[w]).isnumeric() == False):
                                next_df.loc[len(next_df)] = {
                                        'tag': filter_row['tag'],
                                        'tag_col': filter_row['tag_col'],
                                        'word': filter_row['word'],
                                        'next_word': filter_row['next_word'],
                                        'probability': filter_row['probability']
                                        }
                if len(next_df) > 0:
                    tags_df_matrix = next_df.sort_values(by=['probability'], \
                        ascending=False)
                    tags_df_matrix = tags_df_matrix.drop(columns=['word','next_word'])
 
            # Append to the all tag columns spanning variable if matrix elements
            # have been identified 
            if len(tags_df_matrix) > 0:
                tags_df = pd.concat([tags_df, tags_df_matrix], ignore_index=True)
                if __main__.args.debug:
                    print("Matrix:", tags_df_matrix)
                    print() 

            # Check if trained words with multiple tags are in the list of words
            if len(tags_df_uni) < 1 & len(tags_df_matrix) < 1:
                filtered_df = __main__.multi_probability_df[\
                    __main__.multi_probability_df['word'].isin(multi_words)].copy()
                filtered_df = filtered_df[filtered_df['tag_col'] == tag_col]

                if len(filtered_df[filtered_df['word'].str.isnumeric() == False]) > 0:
                    filtered_df = filtered_df[filtered_df['word'].str.isnumeric() == False]

                if len(filtered_df) > 0:
                    filtered_df = filtered_df.sort_values(by=['probability', 'count', 'word'], \
                        ascending=False)
                    tags_df_mult = filtered_df[["tag", "tag_col", "probability"]].drop_duplicates(\
                        subset=["tag", "tag_col"])
                    tags_df_mult = tags_df_mult.sort_values(by=['probability'], \
                        ascending=False)
                    if len(tags_df_mult[tags_df_mult['probability'] >= 5]) > 0:
                        tags_df_mult = tags_df_mult[tags_df_mult['probability'] >= 5]

            # Append to the all tag columns spanning variable if multiple elements
            # have been identified 
            if len(tags_df_mult) > 0:
                tags_df = pd.concat([tags_df, tags_df_mult], ignore_index=True)
                if __main__.args.debug:
                    print("Multi:", tags_df_mult)
                    print()

            if len(tags_df) > 0:
                tags_df = tags_df.sort_values(by=['probability'], ascending=False)
            if len(tags_df) > 0:
                tags_prob = int(tags_df["probability"].iloc[0])

            # If there are no results from both tests, test for options within the list of 
            # unique words by using Levenshtein distance 
            if tags_prob < 60:
                filtered_df = get_most_similar(list_words, tag_col)
                if len(filtered_df) > 0:
                    filtered_df = filtered_df.sort_values(by=['probability', 'word'], \
                        ascending=False)
                    tags_df_simil = filtered_df[["tag", "tag_col", "probability"]].drop_duplicates(\
                        subset=["tag", "tag_col"])
                    tags_df_simil = tags_df_simil.sort_values(by=['probability'], ascending=False)
                
            if len(tags_df_simil) > 0:
                tags_df = pd.concat([tags_df, tags_df_simil], ignore_index=True)
                if __main__.args.debug:
                    print("Similar:", tags_df_simil)
                    print()
            if len(tags_df) > 0:
                tags_df = tags_df.sort_values(by=['probability'], ascending=False)

            # In case of exactly the number of results as tag columns given,
            # it is assumed that there is an exact result
            # In case of more results than tag columns given,
            # it is assumed that there is not an exact result            
            if len(tags_df) > 0:
                new_tagged_df.loc[index, tag_col] = tags_df["tag"].iloc[0]
                #tag_quality += int(tags_df["probability"].iloc[0])
                if tag_quality == 0 or int(tags_df["probability"].iloc[0]) < tag_quality:
                    tag_quality = int(tags_df["probability"].iloc[0])

            # In other cases, it is assumed that there is no result,
            # and leave a particular quality marker of 1
            else:
                new_tagged_df.loc[index, tag_col] = np.nan

        new_tagged_df.loc[index, "TB_qual"] = int(tag_quality)

        # Append the new data to the temporary Dataframe
        temp_new_data.loc[index] = new_data_df.loc[index]

        # Print progress
        # Calculate progress and progress bar
        progress = int(round(line / progress_max * 100, 0))

        # Only print when update limit is exceeded
        if progress > (progress_old):
            # Time difference from start of process to now
            timediff = datetime.timedelta(seconds=round(time.time() \
                                                        - existing_start))
            
            # Calculate the remaining seconds for tagging to finish
            timeremain = datetime.timedelta(\
                                            seconds=round(((time.time() - \
                                            existing_start) / line) \
                                            * (progress_max - line)))
            progress_old = progress
            
            # First clear the previous printout
            out_string = "\r" + (" " * 40) + "\r"
            sys.stdout.write(out_string)
            sys.stdout.flush()

            # Print the progress
            out_string = "\rProgress: " + \
                str(progress) + " %  |  time: " + str(timediff) \
                + ' elapsed, ' + str(timeremain) + ' remaining\r'
            sys.stdout.write(out_string)
            sys.stdout.flush()
    
    for t in __main__.tag_cols:
        temp_new_data[t] = new_tagged_df[t]

    # Catch the error in case no temp data has been processed
    try:
        temp_new_data = temp_new_data[list(__main__.data_col_names['data_col_names'])]
        temp_new_data["TB_qual"] = new_tagged_df["TB_qual"]
    except:
        pass

    print()

    return temp_new_data

def get_existing(new_data, existing_data):

    print("Finding existing data...")

    # Make the new Dataframe similar
    exist_col_list = existing_data.columns.to_list()
    new_col_list =[]

    for exist_col in existing_data.columns:
        if exist_col in new_data:
            new_col_list.append(exist_col)

    new_df = new_data.merge(existing_data, on=list(new_col_list), \
        how='left', indicator=True)\
        .assign(**{'Equal': lambda d: d['_merge']\
        .eq('left_only').map({True: True, False: False})})\
        .drop(columns='_merge')

    existing_df = new_data.merge(existing_data, on=list(new_col_list), \
        how='left', indicator=True)\
        .assign(**{'Equal': lambda d: d['_merge']\
        .eq('both').map({True: True, False: False})})\
        .drop(columns='_merge')

    existing_df = existing_df[existing_df['Equal'] == True][exist_col_list]
    new_df = new_df[new_df['Equal'] == True][new_col_list]
    
    # Return the existing and not existing data
    return existing_df, new_df

