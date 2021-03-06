### **TagBotFT - TagBot For Tables**

New data is being tagged based on already existing data. Results are provided in a result_data.xlsx file.

Practically, TagBotFT works like a SPAM filter but not only on SPAM vs non-SPAM. In order to better detect non-relevant data that data is labelled with a "y" tag. Except that tag there is an infinite number of other tags possible.

Use cases can be accounting data with non-natural language like containing just nouns and/or product names, abbreviations and other product or accounting related text pieces. This kind of data cannot be processed by NLP algorithms to be categorized because it usually has no sentence structure. In addition, there are often typing errors in accounting data when using manual data entry or even optical character recognition. Therefore, the final step of processing in TagBotFT is Levenshtein distance ([https://en.wikipedia.org/wiki/Levenshtein\\\_distance](https://en.wikipedia.org/wiki/Levenshtein%5C_distance)) to get good results by similarity of strings.

The input data is to be organized in tables of the same columns and column order. The existing data is to be provided in XLSX file format. For new data two different import filter exist: one for SAP report TXT files with a certain column order, and one for more generic XLSX files. Further import filters can be added individually to the code. The current version uses the SAP TXT filter for all files with a TXT suffix.

The results are split in two files: results.xlsx contains the new tagged rows while result_data_old.xlsx contains already tagged rows. This is useful in case if one usually extracts the whole data of a year on a monthly basis while adding the tagged results to the existing data file. However, due to filtering identical data out into the old data results new data with duplicates is not identified as new data to be tagged. Due to the background of TagBotFT in accounting it counts on information columns like dates for uniqueness. In the end, it is up to the user to assure the necessary input data quality.

At least one column must be defined to contain the tags. This is done in the column names and the corresponding setting in the INI file.

#### Requirements

All can be met by Linux distributions like Debian 11 or Arch Linux.

* Python 3.9+
* Pandas 1.0+
* Openpyxl 3.0.3+

TagBotFT employs all CPU cores by default. Therefore, the more cores are available, the faster the processing speed of the main operations.

#### Parameters

```
usage: python tagbotft.py [-t] [-s] [-l]
                          [-n] [-r]

-t, --test      start in test mode with fewer data for faster processing 
-s, --single    use only one thread
-l, --logfile   log console output to file "tagbotft.log"
-n, --newlearn  process the database to learning data
-r, --rebuild   process the existing data to generate a database

A combination of [-n] and [-r] makes sense to completely reprocess the existing data. This is necessary if new data has been processed and added to the existing data.
```

#### Provided example data

The data attached to TagBotFT is compiled from real open government data of the state of Oklahoma ([https://data.ok.gov](https://data.ok.gov/)). In order to make it more compact not all data from the respective years has been kept. The data is only provided to demonstrate the operation of TagBotFT.