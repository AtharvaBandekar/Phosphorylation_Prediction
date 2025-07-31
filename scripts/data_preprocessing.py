from Bio import SeqIO
import os
import pandas as pd
import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pickle

# Amino acid encodings
amino_acids = "ACDEFGHIKLMNPQRSTVWXY" 
aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
num_amino_acids = len(amino_acids)

def one_hot_encode_sequence(sequence, aa_to_int_map, num_aa):
    encoded_sequence = torch.zeros(len(sequence), num_aa)
    for i, aa in enumerate(sequence):
        if aa in aa_to_int_map:
            encoded_sequence[i, aa_to_int_map[aa]] = 1
        else:
            pass
    return encoded_sequence

# Define a Dataset class
class PhosphoDataset(Dataset):
    def __init__(self, data_list, aa_to_int_map, num_aa):
        self.data = data_list
        self.aa_to_int_map = aa_to_int_map
        self.num_aa = num_aa

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sequence, label = self.data[index]
        encoded_sequence = one_hot_encode_sequence(sequence, self.aa_to_int_map, self.num_aa)
        label_tensor = torch.tensor(label, dtype=torch.float32) 
        return encoded_sequence, label_tensor

# Define a function to get DataLoaders
def get_dataloaders(batch_size: int = 64, sequence_length: int = 21, force_reprocess: bool = False):
    script_dir_in_function = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.join(script_dir_in_function, '..', 'data', 'processed')
    raw_data_dir = os.path.join(script_dir_in_function, '..', 'data', 'raw')

    train_data_path = os.path.join(processed_data_dir, 'train_data.pkl')
    val_data_path = os.path.join(processed_data_dir, 'val_data.pkl')
    test_data_path = os.path.join(processed_data_dir, 'test_data.pkl')

    if not force_reprocess and os.path.exists(train_data_path) and \
       os.path.exists(val_data_path) and os.path.exists(test_data_path):
        print(f"{os.path.basename(__file__)}: Loading preprocessed data from .pkl files...")
        with open(train_data_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(val_data_path, 'rb') as f:
            val_data = pickle.load(f)
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
        print(f"{os.path.basename(__file__)}: Preprocessed data loaded successfully.")
    else:
        print(f"{os.path.basename(__file__)}: Running full data preprocessing (or reprocessing forced)...")
        
        # Parse fasta file and create a {UniProt ID: Sequence} dictionary 
        fasta_file_path = os.path.join(raw_data_dir, 'uniprot_mapping.fasta')
        protein_sequences = {}

        try: 
            for record in SeqIO.parse(fasta_file_path, "fasta"):
                uniprot_id = record.id.split('|')[1]
                protein_sequences[uniprot_id] = str(record.seq)

            print(f"{os.path.basename(__file__)}: Loaded {len(protein_sequences)} protein sequences.")

        except FileNotFoundError:
            print(f"{os.path.basename(__file__)}: Error: FASTA file not found at {fasta_file_path}")
            return None, None, None, None, None 
        except Exception as e:
            print(f"{os.path.basename(__file__)}: An error occurred during FASTA parsing: {e}")
            return None, None, None, None, None

        # Parse phosphorylation sites txt file and generate positive IDs
        site_file_path = os.path.join(raw_data_dir, "Homo_sapiens.txt")

        try:
            df_site = pd.read_csv(site_file_path, sep='\t')
            print(f"{os.path.basename(__file__)}: Loaded {len(df_site)} phosphorylation sites from TXT.")
        except FileNotFoundError:
            print(f"{os.path.basename(__file__)}: Error: Phosphorylation site file not found at {site_file_path}")
            return None, None, None, None, None
        except Exception as e:
            print(f"{os.path.basename(__file__)}: An error occurred during site file parsing: {e}")
            return None, None, None, None, None


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
                    print(f"{os.path.basename(__file__)}: Error: Padded window for {uniprot_id} at {position_1_based} is not {total_window_length} length. Actual: {len(padded_window)}")
                
            else:
                missing_uniprot_ids.add(uniprot_id)

        print(f"{os.path.basename(__file__)}: Extracted {len(positive_samples)} positive samples.")

        if missing_uniprot_ids:
            print(f"{os.path.basename(__file__)}: Warning: {len(missing_uniprot_ids)} UniProt IDs from TXT were not found in FASTA data.")


        # Create negative samples
        known_sites = set()
        for index, row in df_site.iterrows():
            known_sites.add((row['UniProt ID'], int(row['Position']), row['AA']))
        print(f"{os.path.basename(__file__)}: Created a set of {len(known_sites)} known phosphorylation sites for lookup.")

        potential_phospho_aas = {'S', 'T', 'Y'}
        all_negative_candidates = []

        for uniprot_id, full_sequence in protein_sequences.items():
            sequence_length = len(full_sequence)

            for position_0_based, aa in enumerate(full_sequence):
                if aa in potential_phospho_aas:
                    position_1_based = position_0_based + 1

                    if (uniprot_id, position_1_based, aa) not in known_sites:
                        if not (0 <= position_0_based < sequence_length):
                            continue

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
                            print(f"{os.path.basename(__file__)}: Error: Padded negative window for {uniprot_id} at {position_1_based} is not {total_window_length} length. Actual: {len(padded_window)}")

        print(f"{os.path.basename(__file__)}: Collected {len(all_negative_candidates)} potential negative samples.")

        # Balancing the dataset and select random samples
        num_positive_samples = len(positive_samples)
        num_neg_to_sample = min(num_positive_samples, len(all_negative_candidates))

        negative_samples = random.sample(all_negative_candidates, num_neg_to_sample)

        print(f"{os.path.basename(__file__)}: Selected {len(negative_samples)} negative samples for a balanced dataset.")

        # Combining and randomizing samples for splitting
        all_samples = positive_samples + negative_samples
        random.shuffle(all_samples)
        print(f"{os.path.basename(__file__)}: Total combined samples after balancing: {len(all_samples)}")

        # Train-validation-test split ratios
        train_ratio = 0.65
        val_ratio = 0.15
        test_ratio = 0.2

        train_data, temp_data = train_test_split(all_samples, test_size=(val_ratio + test_ratio), random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

        print(f"{os.path.basename(__file__)}: Train samples: {len(train_data)}")
        print(f"{os.path.basename(__file__)}: Validation samples: {len(val_data)}")
        print(f"{os.path.basename(__file__)}: Test samples: {len(test_data)}")

        # Save data
        os.makedirs(processed_data_dir, exist_ok=True)
        with open(train_data_path, 'wb') as f: pickle.dump(train_data, f)
        with open(val_data_path, 'wb') as f: pickle.dump(val_data, f)
        with open(test_data_path, 'wb') as f: pickle.dump(test_data, f)
        print(f"{os.path.basename(__file__)}: Processed train, validation, and test data saved to 'data/processed/'")
        

    # Create Dataset and DataLoader instances from the loaded data
    train_dataset = PhosphoDataset(train_data, aa_to_int, num_amino_acids)
    val_dataset = PhosphoDataset(val_data, aa_to_int, num_amino_acids)
    test_dataset = PhosphoDataset(test_data, aa_to_int, num_amino_acids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"{os.path.basename(__file__)}: DataLoaders created: Train batches={len(train_loader)}, Val batches={len(val_loader)}, Test batches={len(test_loader)}")
    
    return train_loader, val_loader, test_loader, num_amino_acids, aa_to_int

# Execution script
if __name__ == "__main__":
    print(f"Running {os.path.basename(__file__)}...")
    
    train_dl, val_dl, test_dl, num_aa, aa_int_map = get_dataloaders(batch_size=32, force_reprocess=True)
    
    if train_dl and val_dl and test_dl:
        print("\nVerification of first batch shapes:")
        first_train_batch_features, first_train_batch_labels = next(iter(train_dl))
        print(f"  Train batch features shape: {first_train_batch_features.shape}") 
        print(f"  Train batch labels shape: {first_train_batch_labels.shape}")   

        first_val_batch_features, first_val_batch_labels = next(iter(val_dl))
        print(f"  Validation batch features shape: {first_val_batch_features.shape}")
        print(f"  Validation batch labels shape: {first_val_batch_labels.shape}")

        print(f"\nNumber of amino acids (from module): {num_aa}")
        print(f"Amino acid to int map (from module): {aa_int_map}")
    else:
        print("Failed to get DataLoaders. Check previous error messages.")






