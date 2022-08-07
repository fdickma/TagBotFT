#
#   TagBotFT - Tag Bot For Tables
#   Convert SAP data
#

import os
import platform
import pandas as pd
import sys
import numpy as np
import time
import datetime
import configparser
from itertools import repeat

import tagbotft_lib_file as tf
import tagbotft_lib as tl

#
# Convertig data starts
#
if __name__ == '__main__':

    tl.message('\nStarting SAP data conversion\n')

    # Setting initial parameters
    # Set default variables
    ini_file = 'sap_convert.ini'
    test = False
    myDir = os.path.abspath(os.path.dirname(__file__)) + '/'
    config = configparser.ConfigParser()
    config.sections()
    config.read(myDir + ini_file)
    tag_col = config['Settings']['tag_col_txt']
    text_col = config['Settings']['text_col_txt']
    input_files = config['Settings']['input_files']
    max_cols = config.getint('Settings','max_cols')

    # Read the input data into dataframe
    convData = tf.read_SAP_1("*.TXT", '', '')

    # Writing tagging results to Excel file
    tf.writeXLS('result_conv.xlsx', convData)

    # End of TagBotFT
    tl.message('SAP data conversion has finished')
