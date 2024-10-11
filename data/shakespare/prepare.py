import os
import requests
import numpy as np
import tiktoken


# Get the directory of the current script.
current_script_directory = os.path.dirname(__file__)

# Create a path to the input file in the same directory.
input_file_path = os.path.join(current_script_directory, "input.txt")

# Check if the input file already exists.
if not os.path.exists(input_file_path):
    # URL to download the Tiny Shakespeare dataset.
    dataset_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    
    # Open the input file in write mode to save the dataset.
    with open(input_file_path, mode='w', encoding="utf-8") as file_object:
        # Download the Tiny Shakespeare dataset and write it to the file.
        dataset_content = requests.get(url=dataset_url).text
        file_object.write(dataset_content)
        
# Open the input file in read mode to load the dataset.
with open(input_file_path, mode='r', encoding='utf-8') as file_reader:
    dataset_content = file_reader.read()
    
# Calculate the length of the dataset.
total_length = len(dataset_content)

# Split the dataset into training and validation data.
training_data = dataset_content[:int(total_length * 0.9)]  
validation_data = dataset_content[int(total_length * 0.9):]  



# Get the encoding for GPT-2 using byte-pair encoding (BPE).
encoding = tiktoken.get_encoding("gpt2")

# Encode the training and validation data into token IDs using the BPE encoding.
training_ids = encoding.encode_ordinary(training_data)
validation_ids = encoding.encode_ordinary(validation_data)

# Print the number of tokens in the training and validation datasets.
print(f"Training data has {len(training_ids):,} tokens.")    # Training data has 301,966 tokens.
print(f"Validation data has {len(validation_ids):,} tokens.") # Validation data has 36,059 tokens.

# Convert the token IDs to NumPy arrays with unsigned 16-bit integers.
training_ids = np.array(training_ids, dtype=np.uint16)
validation_ids = np.array(validation_ids, dtype=np.uint16)

# Export the token ID arrays to binary files for efficient storage and retrieval.
training_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
validation_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

