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

    test = False
    tag_col = 'System'
    text_col = 'Beschreibung'

    # Read the input data into dataframe
    convData = tf.read_SAP_1("*.TXT", '', '')

    # Writing tagging results to Excel file
    tf.writeXLS('result_conv.xlsx', convData)

    # End of TagBotFT
    tl.message('SAP data conversion has finished')
