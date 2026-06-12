
import os
import pandas as pd
from mp_api.client import MPRester


def get_mp_api_key():
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("Set MP_API_KEY before running download_mp_summary.py.")
    return api_key

def download_and_save_summary(api_key, summary_path):
    """
    Downloads material summary data from Materials Project and saves it to a CSV file.
    Only runs if the file does not already exist.
    """
    if os.path.exists(summary_path):
        print(f"Summary file already exists at: {summary_path}")
        print("Skipping download.")
        return

    print("Summary file not found. downloading data from Materials Project...")
    try:
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(
                fields=["material_id", "formula_pretty", "is_stable", "energy_above_hull"]
            )
        
        # Convert to DataFrame
        # Assuming docs are Pydantic models with .dict() method as per original script
        df_complete = pd.DataFrame([doc.dict() for doc in docs])
        
        # Save to CSV
        output_dir = os.path.dirname(summary_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        df_complete.to_csv(summary_path, index=False)
        print(f"Successfully saved {len(df_complete)} records to {summary_path}")
        
    except Exception as e:
        print(f"An error occurred during download or saving: {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    SUMMARY_PATH = os.path.join(script_dir, '../data/mp_complete_summary.csv')
    
    download_and_save_summary(get_mp_api_key(), SUMMARY_PATH)
