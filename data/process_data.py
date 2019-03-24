# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
  """
    Function to load messages and categories data
    Input:
        message_filepath (path): path to the messages csv file
        categories_filepath (path): path to the categories csv file
    Output:
        A pandas dataframe with merged messages and categories csv files
  """
  # load messages dataset
  messages = pd.read_csv(messages_filepath)
  categories = pd.read_csv(categories_filepath)
  df = pd.merge(messages, categories, on='id')
  return df

def clean_data(df):
    """
      Function to clean the input data
      Input:
          df (pandas dataframe): dataframe that needs to be cleaned
      Output:
          A cleaned pandas dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.str.split('-', expand=True)[0].values
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
      # set each value to be the last character of the string
      categories[column] = categories[column].str.split('-', expand=True)[1]
      # convert column from string to numeric
      categories[column] = categories[column].astype(int)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(df.reset_index(), categories.reset_index()).drop('index', axis=1).drop('categories', axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df



def save_data(df, database_filename):
    """
      Function to save the input dataframe into a sql database
      Inputs:
          df (pandas dataframe): contains the data that needs to be stored into the sql database
          database_filename (filename): a path to the database into which data needs to be inserted
      Output:
          True if data is successfully inserted else False
    """ 
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('dataset', engine, index=False, if_exists='replace')

    nrows = engine.execute('select count(*) from dataset').fetchall()[0][0]
    return (df.shape[0] == nrows) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        success = save_data(df, database_filepath)
        
        if success:
          print('Cleaned data saved to database!')
        else:
          print('Process failed')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()