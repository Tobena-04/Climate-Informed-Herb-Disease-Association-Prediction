import pandas as pd
import numpy as np
import networkx as nx # visualize graph
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import requests
from pygbif import occurrences #gbif API
from pathlib import Path
import zipfile
from kgcpy import lookupCZ # for biome data (koppen-geiger)
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HERB_KERNEL_DIR = DATA_DIR / "herb_kernel"
DISEASE_KERNEL_DIR = DATA_DIR / "disease_kernel"
ASSOC_DIR = DATA_DIR / "disease_herb"

def get_gbif_data(datasets):
    """
    Include the biome regions of herbs to make it easier to perform

    """

    try:
        # Load herb IDs from the dataset to map with occurrence data
        herb_ids = datasets
        print(f"Loaded {len(herb_ids)} herbs for climate integration")

        # Example: Get climate data for herb species (pseudo-code)
        # You'll need to implement actual GBIF API calls for your specific herb species

        herb_species = datasets["sci_name"]

        gbif_data = {}
        for species in herb_species:
            try:
                # Search for occurrences
                occ = occurrences.search(scientificName=species, limit=50)
                if occ['results']:
                    gbif_data[species] = occ['results']
                    print(f"Found {len(occ['results'])} occurrences for {species}")

                    #Extract climate data for these occurrences
                    climate_data = extract_climate_data({species: occ['results']})
                    print(f"Climate data for {species}: {climate_data[species][:2]}")
                else:
                    print(f"No GBIF data found for {species}")
                    datasets.drop(species)
            except Exception as e:
                print(f"Error fetching GBIF data for {species}: {e}")

        all_climate_data = extract_climate_data(gbif_data)

        return gbif_data, all_climate_data

    except Exception as e:
        print(f"Climate data integration error: {e}")
        return {}

def extract_climate_data(gbif_occurrences):
    """
    Extract coordinates from GBIF data and get climate/biome information
    """
    climate_results = {}
    
    for species, occurrences_list in gbif_occurrences.items():
        print(f"Processing climate data for {species}")
        species_climate_data = []
        
        for occ in occurrences_list:
            # Extract coordinates if available
            if 'decimalLatitude' in occ and 'decimalLongitude' in occ:
                lat = occ.get('decimalLatitude')
                lon = occ.get('decimalLongitude')
                
                if lat is not None and lon is not None:
                    # Get climate data for these coordinates
                    climate_info = get_climate_from_coordinates(lat, lon)
                    
                    record_data = {
                        'species': species,
                        'latitude': lat,
                        'longitude': lon,
                        'country': occ.get('country', 'Unknown'),
                        'year': occ.get('year'),
                        'climate_data': climate_info
                    }
                    species_climate_data.append(record_data)
        
        climate_results[species] = species_climate_data
    
    return climate_results

def get_climate_from_coordinates(lat, lon):
    climate_info = {}
    
    try:
        climate_info['koppen_geiger'] = get_koppen_geiger(lat, lon)
        
    except Exception as e:
        print(f"Error getting climate data for {lat}, {lon}: {e}")
    
    return climate_info

def get_koppen_geiger(lat, lon):
    return lookupCZ(lat, lon)

# Function to map climate zone to biome
def map_climate_to_biome(climate_zone):
    if pd.isna(climate_zone) or climate_zone == 'Unknown':
        return 'Unknown'
    
    first_letter = str(climate_zone)[0]
    biome_map = {
        'A': 'Tropical',
        'B': 'Dry',
        'C': 'Temperate', 
        'D': 'Continental',
        'E': 'Polar',
        'O': 'Ocean'
    }
    return biome_map.get(first_letter, 'Unknown')

# If columns are separated by a specific delimiter (comma, tab, space, etc.)
climate_mapped_data = PROJECT_ROOT / DATA_DIR / HERB_KERNEL_DIR / 'herb_ecoregion_mapping.csv'
disease_data = PROJECT_ROOT / DATA_DIR / DISEASE_KERNEL_DIR / 'disease_id.csv'

df_h = pd.read_csv(climate_mapped_data, header=None, delimiter=',')
df_h.columns = ['#', 'ID', 'sci_name']

df_d = pd.read_csv(disease_data, header=None, delimiter=',')
df_d.columns = ['#', 'ID']
print(f"{df_h}\n{df_d}")

gbif_occurrences, climate_information = get_gbif_data(df_h)

climate_df_data = []
for species, records in climate_information.items():
    for record in records:
                climate_df_data.append({
                    'HerbID': df_h[df_h['sci_name'] == species]['ID'].values[0],
                    'species': species,
                    'biome': record['climate_data'].get('biome', 'Unknown'),
                    'climate_zone': record['climate_data'].get('koppen_geiger', 'Unknown'),
                })

climate_df = pd.DataFrame(climate_df_data)
climate_df.to_csv(PROJECT_ROOT / DATA_DIR / HERB_KERNEL_DIR / 'herb_climate_data.csv', index=False)

# Apply the mapping to update the biome column
climate_df['biome'] = climate_df['climate_zone'].apply(map_climate_to_biome)

climate_df = climate_df.drop_duplicates(subset=['HerbID', 'climate_zone']).reset_index(drop=True)

# Save the updated CSV
climate_df.to_csv(PROJECT_ROOT / DATA_DIR / HERB_KERNEL_DIR / 'herb_climate_data.csv', index=False)

print(climate_df[['climate_zone', 'biome']].value_counts())