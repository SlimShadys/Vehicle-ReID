import os
import re

import matplotlib.pyplot as plt

epochs_to_filter = 5
data_path = os.path.join('results', 'Test-14', 'results-val.txt')

# Read the data from the file
with open(data_path) as f:
    data = f.read()

# Regular expressions to extract the data
epoch_pattern = re.compile(r'Epoch (\d+)')
cmc_rank_1_pattern = re.compile(r'CMC Rank-1: (\d+\.\d+)% \((\d+\.\d+)%\)')
#cmc_rank_3_pattern = re.compile(r'CMC Rank-3: (\d+\.\d+)% \((\d+\.\d+)%\)')
cmc_rank_5_pattern = re.compile(r'CMC Rank-5: (\d+\.\d+)% \((\d+\.\d+)%\)')
cmc_rank_10_pattern = re.compile(r'CMC Rank-10: (\d+\.\d+)% \((\d+\.\d+)%\)')
map_pattern = re.compile(r'mAP: (\d+\.\d+)% \((\d+\.\d+)%\)')

# Lists to store the extracted data
epochs = []
cmc_rank_1 = []
cmc_rank_1_reranked = []
#cmc_rank_3 = []
#cmc_rank_3_reranked = []
cmc_rank_5 = []
cmc_rank_5_reranked = []
cmc_rank_10 = []
cmc_rank_10_reranked = []
maps = []
maps_reranked = []

# Extracting the data
for match in re.finditer(epoch_pattern, data):
    epoch = int(match.group(1))
    epochs.append(epoch)
    
    cmc_rank_1_match = cmc_rank_1_pattern.search(data, match.end())
    #cmc_rank_3_match = cmc_rank_3_pattern.search(data, match.end())
    cmc_rank_5_match = cmc_rank_5_pattern.search(data, match.end())
    cmc_rank_10_match = cmc_rank_10_pattern.search(data, match.end())
    map_match = map_pattern.search(data, match.end())
    
    if cmc_rank_1_match and cmc_rank_5_match and cmc_rank_10_match and map_match: # and cmc_rank_3_match:
        cmc_rank_1.append(float(cmc_rank_1_match.group(1)))
        cmc_rank_1_reranked.append(float(cmc_rank_1_match.group(2)))
        #cmc_rank_3.append(float(cmc_rank_3_match.group(1)))
        #cmc_rank_3_reranked.append(float(cmc_rank_3_match.group(2)))
        cmc_rank_5.append(float(cmc_rank_5_match.group(1)))
        cmc_rank_5_reranked.append(float(cmc_rank_5_match.group(2)))
        cmc_rank_10.append(float(cmc_rank_10_match.group(1)))
        cmc_rank_10_reranked.append(float(cmc_rank_10_match.group(2)))
        maps.append(float(map_match.group(1)))
        maps_reranked.append(float(map_match.group(2)))

# Filter the data to only show every 5 epochs
epochs = epochs[::epochs_to_filter]
cmc_rank_1 = cmc_rank_1[::epochs_to_filter]
cmc_rank_1_reranked = cmc_rank_1_reranked[::epochs_to_filter]
#cmc_rank_3 = cmc_rank_3[::epochs_to_filter]
#cmc_rank_3_reranked = cmc_rank_3_reranked[::epochs_to_filter]
cmc_rank_5 = cmc_rank_5[::epochs_to_filter]
cmc_rank_5_reranked = cmc_rank_5_reranked[::epochs_to_filter]
cmc_rank_10 = cmc_rank_10[::epochs_to_filter]
cmc_rank_10_reranked = cmc_rank_10_reranked[::epochs_to_filter]
maps = maps[::epochs_to_filter]
maps_reranked = maps_reranked[::epochs_to_filter]

# Plotting the data
plt.figure(figsize=(12, 8))
plt.plot(epochs, cmc_rank_1, label='CMC Rank-1', marker='o')
#plt.plot(epochs, cmc_rank_1_reranked, label='CMC Rank-1 (Re-ranked)', marker='s')
#plt.plot(epochs, cmc_rank_3, label='CMC Rank-3', marker='o')
##plt.plot(epochs, cmc_rank_3_reranked, label='CMC Rank-3 (Re-ranked)', marker='s')
plt.plot(epochs, cmc_rank_5, label='CMC Rank-5', marker='o')
#plt.plot(epochs, cmc_rank_5_reranked, label='CMC Rank-5 (Re-ranked)', marker='s')
plt.plot(epochs, cmc_rank_10, label='CMC Rank-10', marker='o')
#plt.plot(epochs, cmc_rank_10_reranked, label='CMC Rank-10 (Re-ranked)', marker='s')
plt.plot(epochs, maps, label='mAP', linestyle=':', marker='o')
plt.plot(epochs, maps_reranked, label='mAP (Re-ranked)', linestyle=':', marker='s')

plt.xlabel('Epochs')
plt.ylabel('Percentage')
plt.title('CMC Ranks and mAP over Epochs (with Re-ranked values)')
plt.legend()
plt.grid(True)
#plt.savefig('results.png')
plt.show()