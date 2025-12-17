import pandas as pd
import json
from collections import defaultdict

# Define the file paths
tsv_file = 'Path to your input TSV file'  # Replace with the path to your TSV file

# tsv_file = 'Path to your input TSV file'  # Replace with the path to your TSV file
output_json_file = 'Path to your output JSON file'  # Replace with the desired output file path

# Read the TSV file
df = pd.read_csv(tsv_file, sep='\t', header=0, names=['qid', 'relevant', 'non_relevant'])

# Create a dictionary to hold the data
data = defaultdict(lambda: {"p": None, "n": []})
i=0
# Iterate through the rows in the dataframe
for index, row in df.iterrows():
    qid = str(row['qid'])
    relevant = row['relevant']
    non_relevant = row['non_relevant']

    if not pd.notna(non_relevant):
        data[qid]["n"].append(non_relevant)

    # If this is the first time we're seeing this qid, assign the relevant candidate to "p"
    if data[qid]["p"] is None:
        data[qid]["p"] = relevant

    # Append non-relevant candidates to "n" list
    if pd.notna(non_relevant):
        data[qid]["n"].append(non_relevant)

# # Save the result as a JSON file
with open(output_json_file, 'w') as f:
    json.dump(data, f, indent=4)

print(f"TSV file successfully converted to JSON and saved to {output_json_file}")
