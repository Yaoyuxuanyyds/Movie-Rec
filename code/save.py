import utils
import torch
import numpy as np
import os
import pickle

# ==============================
import world

utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)

weight_file = utils.getFileName()
print(f"load {weight_file}")


Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
world.cprint(f"loaded model weights from {weight_file}")
Recmodel.eval()


users_embeddings, items_embeddings = Recmodel.getAllEmbeddings()


import pickle

# 保存 embeddings 到文件
users_embeddings_file = './save/'+world.dataset+'/users_embeddings.npy'
items_embeddings_file = './save/'+world.dataset+'/items_embeddings.npy'
# 先创建保存 embeddings 的文件夹

os.makedirs('./save/'+world.dataset+'/')
# 然后保存 embeddings
np.save(users_embeddings_file, users_embeddings.detach().cpu().numpy())  # 确保数据在 CPU 上
np.save(items_embeddings_file, items_embeddings.detach().cpu().numpy())
print(f"Saved user embeddings to {users_embeddings_file}")
print(f"Saved item embeddings to {items_embeddings_file}")

# 保存字典到文件
maps_file = './save/'+world.dataset+'/dataset_maps.pkl'
dataset_maps = {
    'movie_map': dataset.movie_map,
    'user_map': dataset.user_map
}
with open(maps_file, 'wb') as f:
    pickle.dump(dataset_maps, f)

print(f"Saved dataset maps to {maps_file}")

# ===============================================





