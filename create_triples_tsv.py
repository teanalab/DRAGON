import pandas as pd

# Load the TSV file
input_file = 'Path to your input TSV file'  # Replace with the path to your TSV file
output_file = 'Path to your output TSV file'  # Replace with the desired output file path

# Read the input file
df = pd.read_csv(input_file, sep='\t')

# Separate relevant and non-relevant candidates
relevant_df = df[df['relevance'] == 1][['question_id', 'candidate']].rename(columns={'candidate': 'relevant_candidate'})
non_relevant_df = df[df['relevance'] == 0][['question_id', 'candidate']].rename(columns={'candidate': 'non_relevant_candidate'})

# Merge relevant and non-relevant candidates based on `question_id` with a cross join to get all combinations
result_df = pd.merge(relevant_df, non_relevant_df, on='question_id', how='outer')

# Save the resulting triples to a new TSV file
result_df.to_csv(output_file, sep='\t', index=False, header=['qid', 'relevant candidate', 'non-relevant candidate'])

print(f'The processed TSV file has been saved as {output_file}.')
