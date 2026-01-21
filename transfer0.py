##-R "select[gpu80gb]" 
import numpy as np
import pandas as pd
import json
import ase
import ase.db
from ase import Atoms
import matplotlib.pyplot as plt
import numpy.linalg as la
import scipy.stats as ss
import scipy as sp
from statistics import mean
import os

json_train='./train.json'
df_train=pd.read_json(json_train)

#-----------------------------------------------
import torch, torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
#-----------------------------------------------

class MaterialsDataset(Dataset):
    def __init__(self, data_frame, root=None, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.df = data_frame.reset_index(drop=True)

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]
        z = torch.tensor(row['atoms']['numbers'], dtype=torch.long)
        pos = torch.tensor(row['atoms']['positions'], dtype=torch.float)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        adj = (dist < 10.0) & (dist > 0)
        edge_index = adj.nonzero().t().contiguous()
        y = torch.tensor([row.get('hform', 0.0)], dtype=torch.float)
        data = Data(z=z, pos=pos, y=y, edge_index=edge_index)
        return data

train_df, val_df = df_train[:6200], df_train[6200:8000]
train_loader = DataLoader(MaterialsDataset(train_df), batch_size=32, shuffle=True)
val_loader = DataLoader(MaterialsDataset(val_df), batch_size=32)

# model--------------------------------------------------------
from torch_geometric.nn import SchNet, global_add_pool
#import torch.hub

#pretrained_model = torch.hub.load('atomistic-machine-learning/schnetpack', 'schnet_qm9', pretrained=True)
#pretrained_model = torch.hub.load('atomistic-machine-learning/schnetpack:v1.0.0', 'schnet_qm9', pretrained=True, trust_repo=True)

class CustomSchNet(SchNet):
    def forward(self, z, pos, batch=None, edge_index=None):
        if edge_index is None:
            raise ValueError("edge_index must be provided to bypass torch-cluster")
        
        x = self.embedding(z)
        # Distance Expansion
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)
        # Interaction Blocks
        for interaction in self.interactions:
            x = x + interaction(x, edge_index, edge_weight, edge_attr)
        # Output MLP
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        # Global Pooling (fix for AttributeError)
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            
        return global_add_pool(x, batch)

      
#Wrapped model: 
model = CustomSchNet(hidden_channels=32, num_filters=32, num_interactions=3, num_gaussians=25, cutoff=4.0)

# transfer learning settings -----------------------------------------------------
import torch.nn.functional as F

#model.load_state_dict(pretrained_model.state_dict(), strict=False,trust_repo=True)
#print("Succes: Prætrænede vægte fra QM9 er indlæst!")

'''
for param in model.parameters():
    param.requires_grad = False     # freeze all gradients

# unfreeze layers I want to train (transfer learning) 
# & NOT embeddings bc. these are the ones I suspect have useful domain 'knowledge'
for param in model.interactions[-2:].parameters(): # last 2 interaction layers
    param.requires_grad = True

# also opening the last linear layers (vigtigt for tilpasning til nyt target)
for param in model.lin1.parameters():
    param.requires_grad = True
for param in model.lin2.parameters():
    param.requires_grad = True
'''

# basic optimizer with filter: it now only sees the params where requires_grad=True
criterion = torch.nn.MSELoss()
#optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-5)

# training ---------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.z, data.pos, data.batch, edge_index=data.edge_index)
        loss = criterion(out.view(-1), data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def validate():
    model.eval()
    all_preds = []
    all_targets = []
    for data in val_loader:
        data = data.to(device)
        out = model(data.z, data.pos, data.batch, edge_index=data.edge_index)
        all_preds.append(out)
        all_targets.append(data.y)
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return preds, targets 

#current_dir = os.path.dirname(os.path.abspath(__file__))
working_dir='/zhome/36/d/168246/MaterialsDesignML'
model_dir = '/work3/s214471/materials/weights'
os.makedirs(model_dir, exist_ok=True)
log_dir = os.path.join(working_dir, 'lib/logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'training_loss.txt')
results_dir = os.path.join(working_dir, 'lib/results')
os.makedirs(results_dir, exist_ok=True)



#-----------------------------------------
epochs = 15000


with open(log_file, 'w') as f:
    rmse = 0.0 
    preds_final = None
    targets_final = None
    for epoch in range(1, epochs + 1):
        train_loss = train()
        if epoch % 20 == 0 or epoch == epochs:
            preds_gpu, targets_gpu = validate()
            rmse_tensor = torch.sqrt(torch.mean((preds_gpu - targets_gpu)**2))
            rmse = rmse_tensor.item()
            preds_final = preds_gpu.cpu().numpy().flatten()
            targets_final = targets_gpu.cpu().numpy().flatten()
        status = f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val RMSE: {rmse:.4f}"
        f.write(status + "\n")
        if epoch % 20 == 0:
            f.flush()

if preds_final is not None:
    torch.save(preds_final, os.path.join(results_dir, 'predict.pt'))

save_path = os.path.join(model_dir, "transfer_1.pth")
torch.save(model.state_dict(), save_path)
#-----------------------------------------------------------------

figure_dir=os.path.join(working_dir, 'lib/figures')
os.makedirs(figure_dir, exist_ok=True)

y_val = targets_final 
testpred = preds_final

fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].plot(y_val, testpred, '.', markersize=3, color='b')
ax[0].plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], '-k')
ax[0].set_xlabel('True Formation Energy',fontsize=20)
ax[0].set_ylabel('SchNet transfer learning prediction',fontsize=20)

ax[1].hist((testpred - y_val), bins=50,color='skyblue', edgecolor='black')
ax[1].set_xlabel("Error distribution",fontsize=20)
ax[1].set_ylabel("Freq.", fontsize=20)
plt.tight_layout() 
plt.savefig(os.path.join(figure_dir, 'plot.png'),dpi=300)
plt.show()
