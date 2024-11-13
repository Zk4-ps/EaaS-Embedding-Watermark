import json
import torch
from torch.utils.data import Dataset, DataLoader


# CustomDataset
class CustomDataset(Dataset):
    # the embedding data format: idx: embedding
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        self.keys = list(self.data.keys())

    def __len__(self):
        return len(self.keys)

    # according to pos idx
    def __getitem__(self, pos):
        idx = self.keys[pos]
        embedding = torch.tensor(self.data[idx])

        return idx, embedding

    # return emb with list type
    def get_item_embedding(self, idx):
        embedding = self.data[str(idx)]
        return torch.tensor(embedding)


"""
train_dataset = CustomDataset('../data/enron/train_emb.json')
test_dataset = CustomDataset('../data/enron/test_emb.json')

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print(train_dataset.get_item_embedding(11929))
for batch in train_loader:
    batch_idx, batch_emb = batch
    print(f'Keys: {batch_idx}, Values: {batch_emb}')

    exit(0)
"""
