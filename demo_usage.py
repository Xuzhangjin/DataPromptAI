# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:28:53 2024

@author: zhang
"""
#E:/Existing Files/data_demo/Iris.csv


'''
Titanic Dataset: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

Iris Dataset: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv

Housing Prices Dataset: https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv

Wine Quality Dataset: https://raw.githubusercontent.com/akmand/datasets/master/winequality-red.csv

Air Quality Dataset: https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv

'''

import pandas as pd
from DataPromptAI import (
    load_dataset,
    create_table_definition_prompt,
    combine_prompts,
    generate_code_snippet,
    remove_specific_lines
)

def main():
    # Prompt the user for the dataset location
    location = input("Enter the location of your dataset (local path or URL): ")
    chunk_size = input("Enter the chunk size (0 for full load): ")

    if chunk_size.isdigit():
        chunk_size = int(chunk_size)
    else:
        print("Invalid chunk size. Defaulting to full load.")
        chunk_size = 0

    # Load the dataset
    if chunk_size > 0:
        df_iterator = load_dataset(location, chunksize=chunk_size)
        if df_iterator is not None:
            if isinstance(df_iterator, pd.io.parsers.TextFileReader):
                print("Data loaded in chunks:")
                for chunk in df_iterator:
                    print(chunk.head())  # Display the first few rows of each chunk
            else:
                print("Chunking requested but returned full dataset.")
        else:
            print("Failed to load dataset in chunks.")
    else:
        df = load_dataset(location)
        if df is not None:
            print("Full dataset loaded:")
            print(df.head())  # Display the first few rows
        else:
            print("Failed to load dataset.")
            return

    # If the dataset is loaded in chunks, you can concatenate them into a single DataFrame for the next steps
    if chunk_size > 0 and isinstance(df_iterator, pd.io.parsers.TextFileReader):
        df = pd.concat(df_iterator)
    
    # Test create_table_definition_prompt function
    prompt = create_table_definition_prompt(df)
    print("create_table_definition_prompt output:")
    print(prompt)

    # Test combine_prompts function with a dummy natural language query
    query_prompt = input("What operation do you want to perform on the DataFrame? ")
    combined_prompt = combine_prompts(df, query_prompt)
    print("combine_prompts output:")
    print(combined_prompt)

    # Test generate_code_snippet function (requires a valid API key)
    try:
        API_KEY = "your OpenAI API key"   # Replace with your OpenAI API key
        code_snippet = generate_code_snippet(API_KEY, combined_prompt)
        print("generate_code_snippet output:")
        print(code_snippet)

        # Ask user if they want to run the code
        user_input = input("Do you want to run the code? (Yes/No): ").strip().lower()
        
        if user_input == 'yes':
            print("Running the code...")
            start_strings = ["df = pd.DataFrame"]
            exe_code = remove_specific_lines(code_snippet, start_strings)
            exec(exe_code)
        elif user_input == 'no':
            print("Code execution skipped.")
        else:
            print("Invalid input. Please type 'Yes' or 'No'.")
    except Exception as e:
        print(f"Error generating code snippet: {e}")

if __name__ == "__main__":
    main()
