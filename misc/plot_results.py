import re

import matplotlib.pyplot as plt

with open('results-val.txt') as f:
    data = f.read()

# Regular expressions to extract the data
epoch_pattern = re.compile(r'Epoch (\d+)')
cmc_rank_1_pattern = re.compile(r'CMC Rank-1: (\d+\.\d+)%')
cmc_rank_3_pattern = re.compile(r'CMC Rank-3: (\d+\.\d+)%')
cmc_rank_5_pattern = re.compile(r'CMC Rank-5: (\d+\.\d+)%')
map_pattern = re.compile(r'mAP: (\d+\.\d+)%')

# Lists to store the extracted data
epochs = []
cmc_rank_1 = []
cmc_rank_3 = []
cmc_rank_5 = []
maps = []

# Extracting the data
for match in re.finditer(epoch_pattern, data):
    epoch = int(match.group(1))
    epochs.append(epoch)
    
    cmc_rank_1_match = cmc_rank_1_pattern.search(data, match.end())
    cmc_rank_3_match = cmc_rank_3_pattern.search(data, match.end())
    cmc_rank_5_match = cmc_rank_5_pattern.search(data, match.end())
    map_match = map_pattern.search(data, match.end())
    
    if cmc_rank_1_match and cmc_rank_3_match and cmc_rank_5_match and map_match:
        cmc_rank_1.append(float(cmc_rank_1_match.group(1)))
        cmc_rank_3.append(float(cmc_rank_3_match.group(1)))
        cmc_rank_5.append(float(cmc_rank_5_match.group(1)))
        maps.append(float(map_match.group(1)))

# Plotting the data
plt.figure(figsize=(12, 8))
plt.plot(epochs, cmc_rank_1, label='CMC Rank-1', marker='o')
plt.plot(epochs, cmc_rank_3, label='CMC Rank-3', marker='o')
plt.plot(epochs, cmc_rank_5, label='CMC Rank-5', marker='o')
plt.plot(epochs, maps, label='mAP', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Percentage')
plt.title('CMC Rank-1, Rank-3, Rank-5, and mAP over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('results-val.png')
plt.show()
