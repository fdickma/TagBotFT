### **TagBotFT - TagBot For Tables v2**

New data is being tagged based on already existing data. Results are provided as a results data file in XLSX format.

Practically, TagBotFT works like a generic data tagging automation tool for data organized in tables. Due to optimized automated tagging it hels improving data quality.

Use cases can be accounting data with non-natural language like containing just nouns and/or product names, abbreviations and other product or accounting related text pieces. This kind of data cannot be processed by NLP algorithms to be categorized because it usually has no sentence structure. In addition, there are often typing errors in accounting data when using manual data entry or even optical character recognition. Therefore, the final step of processing in TagBotFT is Levenshtein distance ([https://en.wikipedia.org/wiki/Levenshtein\\\_distance](https://en.wikipedia.org/wiki/Levenshtein%5C_distance)) to get good results by similarity of strings.

The input data has to be organized in tables of the same columns and column order. The existing data is to be provided in XLSX file format. For new data two different import filter exist: one for SAP report TXT files with a certain column order, and one for more generic XLSX files. Further import filters can be added individually to the code. The current version uses the SAP TXT filter for all files with a TXT suffix.

The results are split in two files: results.xlsx contains the new tagged rows while result_data_old.xlsx contains already tagged rows. This is useful in case if one usually extracts the whole data of a year on a monthly basis while adding the tagged results to the existing data file. However, due to filtering identical data out into the old data results new data with duplicates is not identified as new data to be tagged. Due to the background of TagBotFT in accounting it counts on information columns like dates for uniqueness. In the end, it is up to the user to assure the necessary input data quality.

At least one column must be defined to contain the tags. This is done in the column names and the corresponding setting in the INI file.

#### Requirements

All can be met by most Linux distributions like Debian 12.

* Python 3.9+
* Pandas 1.0+
* Openpyxl 3.0.3+

TagBotFT employs all CPU cores by default. Therefore, the more cores are available, the faster the processing speed of the main operations.

#### Parameters

```
usage: python tagbotft.py [-t] [-s] [-l]
                          [-p] [-r]

-t, --test      start in test mode with fewer data for faster processing 
-s, --single    use only one thread
-l, --logfile   log console output to file "tagbotft.log"
-p, --progress  show progress information
-r, --rebuild   re-process the existing data by deleting previously processed data

The usual use case is to tag data based on trained/existing data first. Then a new training might be required to encompass the new data.
```

#### SAP converter

```
usage: sap_convert.py [no parameters / check ini file]

The usual use case is to convert SAP exports in a certain column format to an XLSX file which can be processed by TagBotFT. It is individually used and not necessary to use TagBotFT in general.
```

#### Provided example data

The data attached to TagBotFT is compiled from real open government data of the United States of America state of Oklahoma ([https://data.ok.gov](https://data.ok.gov/)). In order to make it more compact not all data from the respective years has been kept. The data is only provided to demonstrate the operation of TagBotFT.

120 lines of the learn_data_pub1.xlsx is also included in input_2021.xlsx in order to test matching existing data. These are the 120 lines at the bottom of the table.
