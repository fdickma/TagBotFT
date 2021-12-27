import pandas as pd
import sys
import re
import sqlite3
import multiprocessing as mp
import numpy as np
import time
import datetime
import difflib
from itertools import repeat
# partial partitions data for the Pool version of tagging on Windows
from functools import partial
from difflib import SequenceMatcher

import tagbotft_lib_file as tf
import tagbotft_lib_db as td

def message(msg_text):
    print()
    print("--------------------------------------")
    print(msg_text+"...")
    print("--------------------------------------")
    return
    
# Convert input array to dataframe and assign "relevant"/"non_relevant" as tag
def get_df(df, non_relevant):
    
    print('Preparing Dataframe for Ngrams')
        
    # First get rid of all empty Text rows; no input means nothing to work on
    df = df[df['Text'].notna()]
    print(f'Kept {len(df.index)} lines.')
    
    # Replace all Tags containing the value of non_relevant with "not_relevant" 
    # and all other with "relevant"
    df.loc[df["Tag"] == non_relevant, "Tag"] = "not_relevant"
    df.loc[df["Tag"] != "not_relevant", "Tag"] = "relevant"
    
    # Change text column to uppercase 
    df["Text"] = df["Text"].str.upper()
    
    return df

# Return new dataframe with Ngrams from input dataframe
def get_ngrams_df(df):
    
    print('Extracting Ngrams from dataframe')

    # The number of columns depends on the input data
    # The maximum of Ngrams is the number of columns
    df_ngrams = df["Text"].str.split(expand = True)
    
    # Adding Tag column from original data as first column
    df_ngrams.insert(0, "Tag", df["Tag"], True)
    
    # Generate a dataframe with only the Tag and Ngram column
    not_list_cols = [col for col in df_ngrams.columns if col not in ['Tag']] 
    for num_col in not_list_cols:
        df_tmp = df_ngrams[['Tag', num_col]]
        df_tmp = df_tmp.rename(columns={"Tag" : "ngramTag", num_col : "Ngram"})
        try:
            exploded_df = exploded_df.append(df_tmp, ignore_index=True)
        except:
            exploded_df = df_tmp
    
    # Finally returning the dataframe by removing None values first
    return exploded_df.replace(to_replace='None', value=np.nan).dropna()

# Count occurences of tags    
def count_df(df_ngrams, max_lines):
    
    length = len(df_ngrams.index)
    p = 0
    progress_old = 0

    # Iterating Ngrams DataFrame to extract the DataFrame for analysis
    # Count is added to later count the number of occurrences of a Tag
    df_ngram_all = pd.DataFrame(columns=['Tag', 'Ngram', 'Count'])
    for i, j in df_ngrams.iterrows(): 
        for k in j:
            if k != None and k != "not_relevant" and k != "relevant":
                df_ngram_all = df_ngram_all.append({"Tag" : j["Tag"], \
                                        "Ngram" : k, "Count" : 1}, \
                                        ignore_index=True)
        p += 1
        progress = round(p/length*100)
        if progress > progress_old:
            progress_old = progress
            print('\rCount progress: ' + str(progress) + str(' % '), \
                  end="", flush=True)
        if i > max_lines:
            break
    return df_ngram_all

def clean_ngrams_df(ngrams_df, df, max_lines):
    # Delete all Ngrams with only one character
    ngrams_df = ngrams_df[ngrams_df['Ngram'].map(len) > 1]
    
    # Counting the Ngram occurences for tags as a new DataFrame
    df_ngramcount = ngrams_df.groupby(['Ngram', 'Tag'], as_index=False)\
                    ['Count'].count()

    # Delete all Ngram/tag combinations with less than 5 occurences
    # The more occurences the stronger a correlation exists between
    # Ngram and tag
    dropRows = df_ngramcount[(df_ngramcount['Count'] < 5)].index
    df_ngramcount.drop(dropRows, inplace=True)
    
    # Remove all not unique Ngrams leading towards both relevant or 
    # non_relevant
    # If there are duplicates they have a count on relevant and non relevant,
    # and thus need to be removed
    df_ngram_uni = df_ngramcount.drop_duplicates(subset=['Ngram'], keep=False)
    df_ngram_uni = df_ngram_uni.reset_index(drop=True)

    return df_ngram_uni

def discard_df(ngrams_uni_df, df, max_lines):
    # Copy DataFrame without escape characters and add the columns Test and 
    # Check where Test contains the relevant or non_relevant result and Check
    # the string leading to the Test result
    n = 0
    l = 0
    discarded = []
    unique_tags = []
    for a, b in (ngrams_uni_df.iterrows()):
        # Remove the escape characters
        d = df[df['Text'].str.contains(re.escape(b[0]))].copy()
        # Add the original Tag as Test to re-check later
        d['ngramTag'] = b[1]
        # Add the Ngram being tested
        d['Ngram'] = b[0]
    
        # Get a data from the above data set where the Tag equals the tested 
        # Tag
        chk = d[d['Tag'].str.contains(re.escape(b[1]))]
        # If the number of found Tags is not equal to the original data set, 
        # the Ngram is not a distinct identifier
        if len(chk) != len(d):
            n += 1
            print('\r' + 'Discarded: ' + str(n), end="", flush=True)
            discarded.append(b[0])
        # Otherwise the Tag is a good identifier
        else:
            # If the data set is empty set c to the original data set        
            if l == 0:
                unique_tags = d
            # Otherwise append the original data set entries
            else:
                unique_tags = unique_tags.append(d)
            l += 1    
        if l > (max_lines):
            break

    return unique_tags
    
def filter_df(unique_tags, max_lines, proc_num):
    # Iterate through the de-escaped unique Tags
    # Skip if the Test result and the original value/Tag are not equal
    # In that case the Ngram is not a distinct identifier
    n = 0
    m = 0
    p = 0
    progress_old = 0
    errors = []
    length = len(unique_tags.index)
        
    for a, b in unique_tags.iterrows():
        # Check if the Tag (2nd item)is the same as the Test (last -2 items)
        if b[2] != b[len(b)-2]:
            n += 1
            errors.append(b[len(b)-1])
        # Append the usable data line to the data set
        else:
            if m == 0:
                usable_lines = pd.DataFrame(b).transpose()
            else:
                usable_lines = usable_lines.append(pd.DataFrame(b).transpose())
            m += 1
    
        if a > max_lines * 10:
            break
        p += 1
        progress = round(p/length*100)
        if progress > progress_old:
            progress_old = progress
            print('\rFilter progress: ' + str(progress) + str(' % '), \
                  end="", flush=True)
        
    return usable_lines

# Return columns where more than 10% of the entries are unique
def get_uni_cols(df):
    col_list = []
    # Iterate over all columns
    for col in df:
        # Skip the standard columns for text and tag
        if col == 'Text' or col == 'Tag':
            continue
        # Calculate the degree of uniqueness based on the number of entries
        uni_degree = len(df[col].unique())/len(df.index)*100
        # Only consider the column as relevant when degree over 10 percent
        if uni_degree > 10:
            col_list.append(col)
            print(col + ":", uni_degree)
    print()
    return col_list

# Return unique identifiers for tags in a given column
def get_col_tags(df, col):
    
    # First drop all other columns except tag and col
    df = df.drop(df.columns.difference(['Tag', col]), axis=1)
    
    # Get the data type of col
    print("Data type:", df[col].dtypes)
    col_type = df[col].dtypes
    if col_type == object:
        # Remove multiple space characters
        dictionary = {'   ':'', '  ':''}
        df.replace(dictionary, regex=True, inplace=True)

        # Remove values with only one space character
        df = df[df[col] != " "]

        # Delete all entries with zero length
        df = df[df[col].map(len) > 0]
    # Remove all residual NaN
    df = df[df[col].notna()]
    # Counting the occurences of tags and col entries as a new DataFrame
    df_uni_count = df.groupby(['Tag', col], as_index=False).size()

    # Delete all Ngram/tag combinations with less than 5 occurences
    # The more occurences the stronger a correlation exists between
    # Ngram and tag
    if 'size' in df_uni_count: 
        dropRows = df_uni_count[(df_uni_count['size'] < 2)].index
        df_uni_count.drop(dropRows, inplace=True)
    
    # Remove douplettes which can only exist if the result is not unique
    for a, b in df_uni_count[col].value_counts().iteritems():
        # Check if the tag exists more than one time
        if b > 1:
            # Remove the tag from the results
            indexNames = df_uni_count[df_uni_count[col] == a].index
            df_uni_count.drop(indexNames, inplace=True)
    
    # Remove all colums except the tag and the content column
    df = df_uni_count.drop(df.columns.difference(['Tag', col]), axis=1)

    return col, df

# Return list of tags for non standard text columns
def get_other_df(df):

    message("Identifying other columns Ngrams")

    # First get all relevant columns
    uni_cols = get_uni_cols(df)
    other_df_set = []
    for c in uni_cols:
        print("Adding column:", c)
        # Append the list with column and dataframe
        other_df_set.append(get_col_tags(df, c))
    # Returning list with column and dataframe with tags and ID column
    # including size (the number of occurences per ID item)
    # But before returning the data will be saved in the database
    td.write_other_db(other_df_set)
    return other_df_set

def get_text_df(df):

    message("Identifying text column Ngrams")

    # Split dataframe in a new dataframe with ngrams
    ngrams_raw_df = get_ngrams_df(df)
    ngrams_count = ngrams_raw_df.groupby(['Ngram', 'ngramTag'], as_index=False).size()

    ngrams = ngrams_count.groupby(['Ngram']).filter(lambda x : len(x)<2)
    non_ngrams = ngrams_count.groupby(['Ngram']).filter(lambda x : len(x)>1)

    print(f'Total usable: {len(ngrams.index)}')
    td.write_rel_text_db(ngrams)

    return ngrams, non_ngrams

# Assign the non relevant input data based on distinct Ngrams
def tag_non_relevant(input_df):

    message('Tagging not relevant data first')
    
    # Loading Ngrams and corresponding Tags from SQLite Database
    conn = sqlite3.connect('Ngrams.sqlite')
    ngram_df = pd.read_sql('select * from relNgrams_Text', conn)
        
    # Split the Text into Ngrams / words
    df_ngrams = input_df["Text"].str.split(expand = True)
    
    counter = 0
    allerrors = 0
    progress_old = 0
    p = 0
    length = len(input_df.index)
    existing_start = time.time()
    
    # Iterate the Ngrams
    for a, b in df_ngrams.iterrows():
        n = 0
        result = None
        error = False
        
        # The Ngrams are in b
        for m in b:
            
            # If the input is not empty
            if m is not None:

                # Check if the Input is a non relevant distinct Ngram
                test = ngram_df.loc[ngram_df['Ngram'] == m]
                                
                # It is non relevant if the Ngram is found at all
                if len(test) > 0:
                    if result != ngram_df.loc[test.index[0], "ngramTag"]:
                        n += 1
                    if n > 1:
                        error = True
                    result = ngram_df.loc[test.index[0], "ngramTag"]

        p += 1

        # Time difference from start of process to now
        timediff = datetime.timedelta(seconds=round(time.time() \
                                                    - existing_start))
        
        # Calculate the remaining seconds for tagging to finish
        timeremain = datetime.timedelta(\
                                        seconds=round(((time.time() - \
                                        existing_start) / p) \
                                        * (length - p)))
      
        print("\r" + "Found existing: " + str(p) + " from " 
              + str(length) + '  |  time: ' + str(timediff) 
              + ' elapsed, ' + str(timeremain) 
              + ' remaining             ', end="")

        # If there is a result
        if result is not None:
            
            # Check if no error has been detected and then assign the Tag
            # and an Edit column indicator
            if result == "not_relevant" and error is False:
                input_df.loc[a, "Tag"] = "y"
                input_df.loc[a, "Edit"] = ""
                input_df.loc[a, "Quality"] = 1
                counter += 1
            # If there has been an error it is not a clear non relevant Tag
            # and therefore it is new, which is indicated in the Edit column
            if error is True:
                input_df.loc[a, "Edit"] = "NEW"
                input_df.loc[a, "Quality"] = 0
                allerrors += 1

    print()
    print('New Ngram data: ' + str(allerrors))
    tagged = input_df[(input_df['Tag'] == 'y')]
    untagged = input_df[(input_df['Tag'] == '')]
    print('Tagged data: ' + str(len(tagged)))
    print('Untagged data: ' + str(len(untagged)))

    #print(tagged)    
    return tagged, untagged

def get_existing_proc(in_df, learn_df):
    from functools import reduce

    count = 0
    total_count = len(in_df)

    in_df_copy = in_df.copy()
    in_df_testcopy = in_df.copy()
    in_df_testcopy = in_df_testcopy.astype(str)
    learn_df_copy = learn_df.copy()
    #learn_df_copy = learn_df_copy.astype(str)

    for col in in_df_copy.columns:
        col_type = in_df_copy[col].dtypes
        if col_type == object:
            in_df_copy[col] = in_df_copy[col].str.replace(' ', '')
    
    for col in learn_df_copy.columns:
        col_type = learn_df_copy[col].dtypes
        if col_type == object:
            learn_df_copy[col] = learn_df_copy[col].str.replace(' ', '')

    df_length = len(in_df_copy.index)
    found_lst = []
    not_found_lst = []

    existing_start = time.time()
    total_count = 0    
    comp_cols = []

    for ncol in list(learn_df_copy.columns.values):
        if len(in_df_testcopy[in_df_testcopy[ncol].str.len()>0]) == df_length:
            try:
                comp_cols.append(ncol)
            except:
                comp_cols = ncol

    for a, b in in_df_copy.iterrows():
        # Define empty conditions list
        comparisons = []

        for head in comp_cols:
            if head == "Tag":
                continue
            # If not a tag column add the comparison to conditions list
            # In case of string comparison apply uppercase on the dataframe
            # and the comparing string
            col_type = learn_df_copy[head].dtypes
            if col_type == np.object:
                check = b[head].upper()
                comparisons.append(learn_df_copy[head].str.upper() \
                                    == check)
            else:
                comparisons.append(learn_df_copy[head] == b[head])

        # Check if all conditions are met by applying numpy.logical
        test = learn_df_copy.loc[np.logical_and.reduce(comparisons)]

        if len(test) > 0:
            found_lst.append(in_df.loc[a])
            count += 1

        if len(test) == 0:
            not_found_lst.append(in_df.loc[a])

        total_count += 1

        # Time difference from start of process to now
        timediff = datetime.timedelta(seconds=round(time.time() \
                                                    - existing_start))
        
        # Calculate the remaining seconds for tagging to finish
        timeremain = datetime.timedelta(\
                                        seconds=round(((time.time() - \
                                        existing_start) / total_count) \
                                        * (df_length - total_count)))
      
        print("\r" + "Found existing: " + str(count) + " from " 
              + str(total_count) + '  |  time: ' + str(timediff) 
              + ' elapsed, ' + str(timeremain) 
              + ' remaining             ', end="")

    # Build the dataframe with existing elements from the new data
    found_df = pd.DataFrame(found_lst, \
                    columns=list(in_df.columns.values))
    # Add an edit remark, that this data is existing data
    found_df['Edit'] = 'OLD'
    # Because it is identical data an assignment quality of 1.0 is set
    found_df['Quality'] = 1
    
    # Build the dataframe for not existing data to be further processed
    not_found_df = pd.DataFrame(not_found_lst, \
                    columns=list(in_df.columns.values))
    # In this case the edit remark is empty
    not_found_df['Edit'] = ''

    # Return the existing and not existing data
    return found_df, not_found_df

def get_existing(in_df, learn_df, cores, max_lines):

    message("Identifying existing data")

    start = 0
    proc_num = []
    end = len(in_df.index)

    # One process gets all data
    if cores < 2:
        # One process means all data for that process and one process only
        found_df, not_found_df = get_existing_proc(in_df, learn_df)
        
    # multiple processes require data partitioned 
    else:
        # Define a list of processes form a range
        proc_num = [*range(1, cores + 1)]
        
        part_ngrams = np.array_split(in_df, cores)

        #partialCount = partial(count_df, ngrams=ngrams)
        pool = mp.Pool(processes=cores)
    
        # Define the processing queues with function to call and data together
        # with process number
        pqueue = pool.starmap(get_existing_proc, zip(part_ngrams, \
                            repeat(learn_df)))
        pool.close()
        pool.join()
        # Iterate the Pool segments for results to build the complete results
        for q in pqueue:
            try:
                found_df = found_df.append(q[0], ignore_index=True)
            except:
                found_df = q[0]
            try:
                not_found_df = not_found_df.append(q[1], ignore_index=True)
            except:
                not_found_df = q[1]

    print()
    if len(found_df) > 0:
        print("Found existing data:", len(found_df))
    if len(not_found_df) > 0:
        print("Found new data:", len(not_found_df))
    # Return the existing and not existing data
    return found_df, not_found_df

def count_tags_df(df_ngrams, max_lines):
    
    length = len(df_ngrams.index)
    p = 0
    progress_old = 0

    # Iterating Ngrams DataFrame to extract the DataFrame for analysis
    # Count is added to later count the number of occurrences of a Tag
    df_ngram_all = pd.DataFrame(columns=['Tag', 'Ngram', 'Count'])
    for i, j in df_ngrams.iterrows(): 
        for k in j:
            if k != None and k != j["Tag"]:
                df_ngram_all = df_ngram_all.append({"Tag" : j["Tag"], \
                                        "Ngram" : k, "Count" : 1}, \
                                        ignore_index=True)
        p += 1
        progress = round(p/length*100)
        if progress > progress_old:
            progress_old = progress
            print('\rCount progress: ' + str(progress) + str(' % '), \
                  end="", flush=True)
        if i > max_lines:
            break
    return df_ngram_all

def get_tags_df(df, non_ngrams):

    message("Identifying tag column Ngrams")
   
    # First get rid of all empty Text rows; no input means nothing to work on
    df = df[df['Text'].notna()]
    print(f'Kept {len(df.index)} lines.')
    
    # Get rid of all non-relevant rows
    df = df[df['Tag'] != 'y']

    # Change text column to uppercase 
    df["Text"] = df["Text"].str.upper()
    
    # Split dataframe in a new dataframe with ngrams and non-ngrams
    ngrams_raw_df = get_ngrams_df(df)

    print('Ngram maximum columns:', len(ngrams_raw_df.columns))
    
    tmp_ngrams = ngrams_raw_df.groupby(['Ngram']).filter(lambda x : len(x)<2)

    ngrams = tmp_ngrams[~tmp_ngrams[['Ngram']]\
        .isin(non_ngrams.Ngram.tolist()).any(axis=1)]

    print(f'Total usable: {len(ngrams.index)}')
    td.write_text_db(ngrams)

    return ngrams, non_ngrams

# Assign the relevant input data based on distinct Ngrams
def tag_relevant(input_df):

    pd.set_option('mode.chained_assignment', None)

    message('Tagging relevant data')
    
    # Loading Ngrams and corresponding Tags from SQLite Database
    conn = sqlite3.connect('Ngrams.sqlite')
    ngram_df = pd.read_sql('select * from genNgrams_Text', conn)
        
    # Split the Text into Ngrams / words
    df_ngrams = input_df["Text"].str.split(expand = True)
    
    counter = 0
    allerrors = 0
    progress_old = 0
    p = 0
    length = len(input_df.index)
    existing_start = time.time()
    
    # Iterate the Ngrams
    for a, b in df_ngrams.iterrows():
        n = 0
        result = None
        error = False
        
        # The Ngrams are in b
        for m in b:
            
            # If the input is not empty
            if m is not None:
                # Check if the Input is a non relevant distinct Ngram
                test = ngram_df.loc[ngram_df['Ngram'] == m]
                
                # It is non relevant if the Ngram is found at all
                if len(test) > 0:
                    if result != ngram_df["ngramTag"].values[test.index[0]]:
                        n += 1
                    if n > 1:
                        error = True
                    result = ngram_df["ngramTag"].values[test.index[0]]
                    v = 1
  
        p += 1
        # Time difference from start of process to now
        timediff = datetime.timedelta(seconds=round(time.time() \
                                                    - existing_start))
        
        # Calculate the remaining seconds for tagging to finish
        timeremain = datetime.timedelta(\
                                        seconds=round(((time.time() - \
                                        existing_start) / p) \
                                        * (length - p)))

        progress = round(p/length*100)
        if progress > progress_old:
            progress_old = progress
            print('\rFilter progress: ' + str(progress) + str(' % ') 
            + '  |  time: ' + str(timediff) + ' elapsed, ' + str(timeremain)
            + ' remaining                  ', end="")

        # If there is a result
        if result is not None:
            # Check if no error has been detected and then assign the Tag
            # and an Edit column indicator
            if error is False:
                input_df.loc[a, "Tag"] = result
                input_df.loc[a, "Edit"] = ""
                input_df.loc[a, "Quality"] = 1
                counter += 1
            # If there has been an error it is not a clear non relevant Tag
            # and therefore it is new, which is indicated in the Edit column
            if error is True:
                input_df.loc[a, "Edit"] = "NEW"
                input_df.loc[a, "Quality"] = 0
                allerrors += 1

    print()
    print('New Ngram data: ' + str(allerrors))
    tagged = input_df[(input_df['Tag'] != '')]
    untagged = input_df[(input_df['Tag'] == '')]
    print('Tagged data: ' + str(len(tagged)))
    print('Untagged data: ' + str(len(untagged)))
    pd.reset_option('mode.chained_assignment')
    return tagged, untagged

# Assign the relevant input data based on distinct Ngrams
def tag_other(input_df):

    pd.set_option('mode.chained_assignment', None)

    message('Tagging other columns data')
    
    # Loading Ngrams and corresponding Tags from SQLite Database
    tables = []
    conn = sqlite3.connect('Ngrams.sqlite')
    tmp_tab = pd.read_sql('SELECT name FROM sqlite_master WHERE type="table";', conn)
    for tab in tmp_tab['name']:
        if '_Col_' in tab:
            try:
                tables.append(tab)
            except:
                tables = tab

    for t in tables:

        col_name = t.replace("relNgrams_Col_","")
        ngram_df = pd.read_sql('select * from ' + t, conn)
    
        counter = 0
        progress_old = 0
        p = 0
        length = len(input_df.index)
        existing_start = time.time()
    
        # Iterate the Ngrams
        for a, b in input_df[col_name].to_frame().iterrows():
            n = 0
            result = None
            error = False
            
            # The Ngrams are in b
            for m in b:
                
                # If the input is not empty
                if m is not None:
                    # Check if the Input is a non relevant distinct Ngram
                    test = ngram_df.loc[ngram_df[col_name] == m]
                    
                    # It is non relevant if the Ngram is found at all
                    if len(test) > 0:
                        result = test['Tag'].item()
    
            p += 1
            # Time difference from start of process to now
            timediff = datetime.timedelta(seconds=round(time.time() \
                                                        - existing_start))

            # Calculate the remaining seconds for tagging to finish
            timeremain = datetime.timedelta(\
                                            seconds=round(((time.time() - \
                                            existing_start) / p) \
                                            * (length - p)))

            progress = round(p/length*100)
            if progress > progress_old:
                progress_old = progress
                print('\rFilter progress: ' + str(progress) + str(' % ') 
                + '  |  time: ' + str(timediff) + ' elapsed, ' + str(timeremain)
                + ' remaining                  ', end="")

            # If there is a result
            if result is not None:
                # Check if no error has been detected and then assign the Tag
                # and an Edit column indicator
                if result == "not_relevant":
                    input_df.loc[a, "Tag"] = "y"
                    input_df.loc[a, "Edit"] = ""
                    input_df.loc[a, "Quality"] = 1
                    counter += 1

    print()
    tagged = input_df[(input_df['Tag'] != '')]
    untagged = input_df[(input_df['Tag'] == '')]
    print('Tagged data: ' + str(len(tagged)))
    print('Untagged data: ' + str(len(untagged)))
    pd.reset_option('mode.chained_assignment')
    return tagged, untagged

# Find similar entries in a given dataframe with Levenshtein distance
def find_similar(w, df):
    w = w.upper()
    quality = 0.0
    result = ''
    match = difflib.get_close_matches(w, df['Text'].astype(str), 1, 0.25)
    if len(match) > 0:
        str_match = match[0]
        s = difflib.SequenceMatcher(None, str_match, w, autojunk=True).ratio()
        result_df = df[(df['Text'] == str_match)]
        result = result_df['Tag'].values[0]
        quality = float(format(s, '0.02f'))
    return result, quality

# Assign tags to input data by applying Levenshtein distance
def lev_tagging(in_df, learn_df, proc_num):
    progress_old = 0
    p = 0
    length = len(in_df.index)
    existing_start = time.time()
    for a, b in in_df.iterrows():
        # Call the Levenshtein distance calculation function
        result, quality = find_similar(b.Text, learn_df)
        in_df.loc[a, 'Tag'] = result
        in_df.loc[a, 'Quality'] = quality

        p += 1
        
        # Time difference from start of process to now
        timediff = datetime.timedelta(seconds=round(time.time() \
                                                    - existing_start))
        
        # Calculate the remaining seconds for tagging to finish
        timeremain = datetime.timedelta(\
                                        seconds=round(((time.time() - \
                                        existing_start) / p) \
                                        * (length - p)))

        progress = round(p/length*100)
        if progress > progress_old:
            progress_old = progress
            print('\rFilter progress: ' + str(progress) + str(' % ') 
              + '  |  time: ' + str(timediff) + ' elapsed, ' + str(timeremain)
              + ' remaining                  ', end="", flush=True)
    return in_df

# Main tagging function with Levenshtein distance
def tag_lev_df(in_df, learn_df, cores, max_lines):

    message('Tagging similar data with Levenshtein distance')
    
    # Building test data, the data to tagged
    # One process gets all data
    if cores < 2:
        # One process means all data for that process and one process only
        tag_data = lev_tagging(in_df.copy(), learn_df, 1)
        
    # multiple processes need the data to be separated
    else:
        tag_data = np.array_split(in_df.copy(), cores)

        # Define a list of processes form a range
        proc_num = [*range(1, cores + 1)]
    
        # Run tagging as Pool parallel processes; indata and tags are fixed input
        #partial_lev_tagging = partial(lev_tagging, ldata=learn_df)
        pool = mp.Pool(processes=cores)
    
        # Define the processing queues with function to call and data together
        #with process number
        pqueue = pool.starmap(lev_tagging, zip(tag_data, repeat(learn_df), \
                                proc_num))
        pool.close()
        pool.join()
    
        # Iterate the Pool segments for results to build the complete results
        for q in pqueue:
            try:
                found_df = found_df.append(q, ignore_index=True)
            except:
                found_df = q
    print()
    return found_df
