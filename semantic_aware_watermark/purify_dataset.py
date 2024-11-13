import json


dataset = 'mind'

with open(f'data/{dataset}/wm_train_emb.json', 'r') as f:
    wm_train_emb = json.load(f)

with open(f'data/{dataset}/new_train_subset_result.json', 'r') as f:
    new_train_result = json.load(f)

print(len(wm_train_emb.keys()))
print(len(new_train_result))

remove_len = 80
sorted_cos_result = sorted(new_train_result, key=lambda x: x['avg_cos_dist'], reverse=True)  # larger better
sorted_L2_result = sorted(new_train_result, key=lambda x: x['avg_L2_dist'])  # lower better
sorted_pca_result = sorted(new_train_result, key=lambda x: x['avg_pca_dist'])  # lower better

remove_cos_list = sorted_cos_result[:remove_len]
remove_L2_list = sorted_L2_result[:remove_len]
remove_pca_list = sorted_pca_result[:remove_len]

remove_cos_idx = [item['idx'] for item in remove_cos_list]
remove_L2_idx = [item['idx'] for item in remove_L2_list]
remove_pca_idx = [item['idx'] for item in remove_pca_list]

idx_set1 = set(remove_cos_idx)
idx_set2 = set(remove_L2_idx)
idx_set3 = set(remove_pca_idx)

remove_idx = idx_set3
print(len(remove_idx))

for rm_idx in remove_idx:
    if str(rm_idx) in wm_train_emb:
        del wm_train_emb[str(rm_idx)]

with open(f'data/{dataset}/pf_wm_train_emb.json', 'w') as f:
    json.dump(wm_train_emb, f)
