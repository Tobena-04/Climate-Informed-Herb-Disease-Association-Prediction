import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HERB_KERNEL_DIR = DATA_DIR / "herb_kernel"
DISEASE_KERNEL_DIR = DATA_DIR / "disease_kernel"
ASSOC_DIR = DATA_DIR / "disease_herb"

def get_unique_entries_pandas(filename, column_name):
    """
    Get unique entries from a CSV column using pandas.
    
    Args:
        filename (str): Path to the CSV file
        column_name (str): Name of the column to extract unique values from
    
    Returns:
        list: Unique entries from the column
    """
    # Read the CSV file
    df = pd.read_csv(filename)
    
    # Get unique values from the specified column
    unique_values = df[column_name].dropna().unique().tolist()
    
    return unique_values

# Usage
unique_entries = get_unique_entries_pandas(PROJECT_ROOT / DATA_DIR / HERB_KERNEL_DIR / 'herb_climate_data.csv', 'HerbID')
print(len(unique_entries))