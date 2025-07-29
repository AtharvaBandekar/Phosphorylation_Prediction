from Bio import SeqIO
import os
import pandas as pd
import random

# Parse fasta file and create a {UniProt ID: Sequence} dictionary 
script_dir = os.path.dirname(__file__)
fasta_file_path = os.path.join(script_dir, "..", 'data', 'raw', 'uniprot_mapping.fasta')

protein_sequences = {}

try: 
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        uniprot_id = record.id.split('|')[1]
        protein_sequences[uniprot_id] = str(record.seq)

    print(f"Loaded {len(protein_sequences)} protein sequences.")
    print(list(protein_sequences.items())[:5])

except FileNotFoundError:
    print(f"Error: FASTA file not found at {fasta_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")


# Parse phosphorylation sites txt file and generate positive IDs
site_file_path = os.path.join("..", "data", "raw", "Homo_sapiens.txt")

df_site = pd.read_csv(site_file_path, sep='\t')
print(f"Loaded {len(df_site)} phosphorylation sites from TXT.")

window_half_size = 10
total_window_length = (2 * window_half_size) + 1

positive_samples = []
missing_uniprot_ids = set()

for index, row in df_site.iterrows():
    uniprot_id = row['UniProt ID']
    position_1_based = int(row['Position'])
    central_aa = row['AA']

    if uniprot_id in protein_sequences:
        full_sequence = protein_sequences[uniprot_id]
        sequence_length = len(full_sequence)
        position_0_based = position_1_based - 1

        if not (0 <= position_0_based < sequence_length):
            print(f"Skipping site {uniprot_id} at {position_1_based}: Position out of bounds for sequence length {sequence_length}.")
            continue

        start_index = position_0_based - window_half_size
        end_index = position_0_based + window_half_size + 1

        left_pad_needed = max(0, -start_index)
        right_pad_needed = max(0, end_index - sequence_length)

        actual_start = max(0, start_index)
        actual_end = min(sequence_length, end_index)
        extracted_window = full_sequence[actual_start:actual_end]

        padded_window = ('X' * left_pad_needed) + extracted_window + ('X' * right_pad_needed)

        if len(padded_window) == total_window_length:
            positive_samples.append((padded_window, 1))
        else:
            print(f"Error: Padded window for {uniprot_id} at {position_1_based} is not {total_window_length} length. Actual: {len(padded_window)}")
        
    else:
        missing_uniprot_ids.add(uniprot_id)

print(f"Extracted {len(positive_samples)} positive samples.")

if missing_uniprot_ids:
    print(f"Warning: {len(missing_uniprot_ids)} UniProt IDs from TXT were not found in FASTA data.")


# Create negative samples
known_sites = set()
for index, row in df_site.iterrows():
    known_sites.add((row['UniProt ID'], int(row['Position']), row['AA']))
print(f"Created a set of {len(known_sites)} known phosphorylation sites for lookup.")

potential_phospho_aas = {'S', 'T', 'Y'}
all_negative_candidates = []

for uniprot_id, full_sequence in protein_sequences.items():
    sequence_length = len(full_sequence)

    for position_0_based, aa in enumerate(full_sequence):
        if aa in potential_phospho_aas:
            position_1_based = position_0_based + 1

            if (uniprot_id, position_1_based, aa) not in known_sites:
                start_index = position_0_based - window_half_size
                end_index = position_0_based + window_half_size + 1

                left_pad_needed = max(0, -start_index)
                right_pad_needed = max(0, end_index - sequence_length)

                actual_start = max(0, start_index)
                actual_end = min(sequence_length, end_index)
                extracted_segment = full_sequence[actual_start:actual_end]

                padded_window = ('X' * left_pad_needed) + extracted_segment + ('X' * right_pad_needed)
                
                if len(padded_window) == total_window_length:
                    all_negative_candidates.append((padded_window, 0))
                else:
                    print(f"Error: Padded negative window for {uniprot_id} at {position_1_based} is not {total_window_length} length. Actual: {len(padded_window)}")

print(f"Collected {len(all_negative_candidates)} potential negative samples.")

# Balancing the dataset and select random samples
num_positive_samples = len(positive_samples)
num_neg_to_sample = min(num_positive_samples, len(all_negative_candidates))

negative_samples = random.sample(all_negative_candidates, num_neg_to_sample)

print(f"Selected {len(negative_samples)} negative samples for a balanced dataset.")