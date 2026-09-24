import pandas as pd
import duckdb
import os

def convert_parquet_to_json_robust(input_path, output_path):
    print(f"Reading: {input_path}")
    print("Using the DuckDB read engine (handles complex nested and irregular data blocks)...")

    try:
        # Use DuckDB to pull directly into a Pandas DataFrame
        df = duckdb.read_parquet(input_path).df()

        print(f"Read succeeded! Data shape: {df.shape}")

        if 'prompt' in df.columns and not df['prompt'].isnull().all():
            sample = df['prompt'].iloc[0]
            print(f"Prompt column sample (type: {type(sample)}): {str(sample)[:100]}...")
        else:
            print("Warning: the Prompt column is empty or all Null; please check the source file content.")

        print("Converting to JSON...")
        df.to_json(output_path, orient='records', force_ascii=False, indent=4)
        print(f"Conversion done! Saved to: {output_path}")

    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    DATA_ROOT = os.environ.get("DATA_ROOT", os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(DATA_ROOT, "AIME24", "AIME24_verl_val.parquet")
    output_file = os.path.join(DATA_ROOT, "AIME24", "AIME24_verl_val_view.json")

    if os.path.exists(input_file):
        convert_parquet_to_json_robust(input_file, output_file)
    else:
        print(f"Error: input file not found {input_file}")