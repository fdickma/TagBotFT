import configparser
import __main__

# Function to save data to a database in a certain table
def save_data(to_store, database_name, table_name):
    print("Saving to database:\t", table_name)

    # Define database connection
    engine = create_engine('sqlite:///' + database_name, echo=False)
    sqlite_connection = engine.connect()

    # Save data to database
    to_store.to_sql(name=table_name, con=engine, if_exists='replace')

    # Close database connection
    sqlite_connection.close()

# Function to read data from a database from a certain table
def read_data(database_name, table_name):
    print("Reading from database:\t", table_name)

    # Define database connection
    engine = create_engine('sqlite:///' + database_name, echo=False)
    sqlite_connection = engine.connect()

    # Read data from database
    temp_df = pd.read_sql('SELECT * FROM ' + table_name, con=engine, index_col='index')
    
    # Close database connection
    sqlite_connection.close()
    return temp_df

