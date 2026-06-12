import pandas as pd
import os

def split_csv_by_batch(input_filepath, output_dir, batch_size=1000, prefix="batch_"):
    """
    Splits a large CSV file into smaller CSV files, each containing a fixed
    number of entries (rows).
    """
    # 1. Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 2. Read the entire CSV file into a Pandas DataFrame
        data_summary = pd.read_csv(input_filepath, sep='\t').to_dict(orient='records')
        total_rows = len(data_summary)
        
        print(f"Total rows found: {total_rows}")
        print(f"Splitting into batches of {batch_size}...")

        batch_number = 1
        
        # 3. Iterate over the DataFrame in chunks (slices)
        for start_index in range(0, total_rows, batch_size):
            end_index = start_index + batch_size
            
            # Select the current batch slice
            batch_df = data_summary[start_index:end_index]
            
            # Define the output file name
            output_filename = os.path.join(
                output_dir, 
                f"{prefix}{batch_number}.csv"
            )
            
            # 4. Save the batch to a new CSV file
            # index=False prevents Pandas from writing the DataFrame index as a column
            pd.DataFrame(batch_df).to_csv(output_filename, index=False)
            
            print(f"Saved {len(batch_df)} rows to: {output_filename}")
            batch_number += 1
            
        print("\n✅ Splitting complete.")

    except FileNotFoundError:
        print(f"ERROR: Input file not found at {input_filepath}")
    except Exception as e:
        print(f"An error occurred during processing: {e}")

def split_remaining_tasks(todo_filepath, processed_filepath, output_dir, batch_size=1000, prefix="batch_"):
    """
    Finds entries in 'todo' that are not in 'processed' and splits them into batches.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Load the To-do list (the full table)
        # We assume 'Key' is the column name based on your previous messages
        df_todo = pd.read_csv(todo_filepath, sep='\t')
        if 'Key' in df_todo.columns:
            df_todo = df_todo.rename(columns={'Key': 'material_id'})
        # 2. Load the Processed list
        # If this is just a list of IDs, we read it as a simple series
        if os.path.exists(processed_filepath):
            # Read first column as the keys already finished
            df_processed = pd.read_csv(processed_filepath, sep='\t')
            p_col = 'material_id' if 'material_id' in df_processed.columns else 'Key'
            processed_keys = set(df_processed[p_col].astype(str))
        else:
            print("Processed file not found. Starting from scratch.")
            processed_keys = set()

        # 3. Filter: Find rows where 'Key' is NOT in processed_keys
        # This is the "Remaining" work
        df_remaining = df_todo[~df_todo['material_id'].astype(str).isin(processed_keys)]
        
        total_remaining = len(df_remaining)
        print(f"Total entries in To-do: {len(df_todo)}")
        print(f"Already processed: {len(processed_keys)}")
        print(f"Remaining tasks to split: {total_remaining}")

        if total_remaining == 0:
            print("Everything is already processed! No new batches created.")
            return

        # 4. Split the remaining dataframe into batches
        batch_number = 1
        for start_index in range(0, total_remaining, batch_size):
            end_index = start_index + batch_size
            batch_df = df_remaining.iloc[start_index:end_index]
            
            output_filename = os.path.join(output_dir, f"batch_{batch_number}.csv")
            
            # Save as tab-separated txt file
            batch_df.to_csv(output_filename, index=False, sep=',')
            
            print(f"Saved {len(batch_df)} rows to: {output_filename}")
            batch_number += 1
            
        print("\n✅ Batching complete.")

    except KeyError:
        print("ERROR: Could not find a column named 'Key' in one of the files.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Configuration ---
if __name__ == "__main__":
    # Ensure you replace 'path/to/your_large_file.csv' with the actual path
    #INPUT_FILE = "../data/MP_properties.csv" 
    
    # The directory where the new, smaller files will be saved
    #OUTPUT_DIRECTORY = "../data/batches" 
    
    # Run the splitting function
    #split_csv_by_batch(INPUT_FILE, OUTPUT_DIRECTORY, batch_size=1000)

    TODO_FILE = "../data/materials_with_bandstructure.txt"        # The full list of materials
    PROCESSED_FILE = "../data/processed_materials.txt" # The list of things you already finished
    
    OUTPUT_DIRECTORY = "../data/batches_2" 
    
    split_remaining_tasks(TODO_FILE, PROCESSED_FILE, OUTPUT_DIRECTORY, batch_size=1000)