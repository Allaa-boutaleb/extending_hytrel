import pandas as pd
from pathlib import Path
import pickle

# Get the current directory
current_dir = Path(__file__).parent

# Read the CSV file
csv_path = current_dir / 'santos_small_benchmark_groundtruth.csv'
df = pd.read_csv(csv_path)

# Create the dictionary
santos_dict = df.groupby('query_table')['data_lake_table'].apply(list).to_dict()

# Save the dictionary as a pickle file
pickle_path = current_dir / 'santosUnionBenchmark.pickle'
with open(pickle_path, 'wb') as pickle_file:
    pickle.dump(santos_dict, pickle_file)

print(f"Pickle file '{pickle_path}' has been created successfully.")