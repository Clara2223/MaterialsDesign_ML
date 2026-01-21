#code written: Wednesday 21 of January 2026
#by: Clara Wimmelmann
#at: Technical University of Denmark

import numpy as np
import pandas as pd
import os
import torch, torch_geometric
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import QM9
from torch_geometric.nn import SchNet, global_add_pool

import json
import ase
import ase.db
from ase import Atoms
import numpy.linalg as la
import scipy.stats as ss
import scipy as sp
from statistics import mean

#-----------------------------------------------
json_train='./train.json'
df_train=pd.read_json(json_train)
train_df, val_df = df_train[:6200], df_train[6200:8000]
#-----------------------------------------------

class MaterialsDataset(Dataset):
    def __init__(self, data_frame):
        super().__init__()
        self.df = data_frame.reset_index(drop=True)
        self.processed_data = []
        print("preparing graphs ..")
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            z = torch.tensor(row['atoms']['numbers'], dtype=torch.long)
            pos = torch.tensor(row['atoms']['positions'], dtype=torch.float)
            # Beregn afstande
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)
            dist = torch.norm(diff, dim=-1)
            adj = (dist < 6.0) & (dist > 0)
            edge_index = adj.nonzero().t().contiguous()
            y = torch.tensor([row.get('hform', 0.0)], dtype=torch.float)
            self.processed_data.append(Data(z=z, pos=pos, y=y, edge_index=edge_index))
    def len(self):
        return len(self.processed_data)
    def get(self, idx):
        return self.processed_data[idx]

train_dataset = MaterialsDataset(train_df)
val_dataset = MaterialsDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)


# model--------------------------------------------------------
#import torch.hub
#pretrained_model = torch.hub.load('atomistic-machine-learning/schnetpack', 'schnet_qm9', pretrained=True)
#pretrained_model = torch.hub.load('atomistic-machine-learning/schnetpack:v1.0.0', 'schnet_qm9', pretrained=True, trust_repo=True)

class CustomSchNet(SchNet):
    def forward(self, z, pos, batch=None, edge_index=None):
        if edge_index is None:
            raise ValueError("edge_index skal medsendes!")
        x = self.embedding(z)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)
        for interaction in self.interactions:
            x = x + interaction(x, edge_index, edge_weight, edge_attr)
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        return global_add_pool(x, batch)

#my Wrapped model (just helps with the edge-index bug-fixing): 
model = CustomSchNet(hidden_channels=128, num_filters=128, num_interactions=6, num_gaussians=50, cutoff=10.0)


# --- loading QM9 SchNet pretrained weights --------------------------------------
#model.load_state_dict(pretrained_model.state_dict(), strict=False,trust_repo=True)
qm9_path = './data/QM9'
print("henter vægte ...")
dataset_qm9 = QM9(qm9_path)
pretrained_model, _ = SchNet.from_qm9_pretrained(qm9_path, dataset_qm9, target=7)
model.load_state_dict(pretrained_model.state_dict(), strict=False)
print("Succes: vægte hentet")

# transfer learning settings -----------------------------------------------------
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

# basic optimizer with filter: it now only sees the params where requires_grad=True
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-5)

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
    all_preds, all_targets = [], []
    for data in val_loader:
        data = data.to(device)
        out = model(data.z, data.pos, data.batch, edge_index=data.edge_index)
        all_preds.append(out.cpu())
        all_targets.append(data.y.cpu())
    return torch.cat(all_preds).numpy().flatten(), torch.cat(all_targets).numpy().flatten()


#current_dir = os.path.dirname(os.path.abspath(__file__))
working_dir='/zhome/36/d/168246/MaterialsDesignML'
model_dir = '/work3/s214471/materials/weights'
os.makedirs(model_dir, exist_ok=True)
log_dir = os.path.join(working_dir, 'lib/logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'training_loss.txt')
results_dir = os.path.join(working_dir, 'lib/results')
os.makedirs(results_dir, exist_ok=True)

#------------------------------------------------------------------------------
epochs = 500

with open(log_file, 'w') as f:
    for epoch in range(1, epochs + 1):
        train_loss = train()
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            preds, targets = validate()
            rmse = np.sqrt(np.mean((preds - targets)**2))
            status = f"epoch {epoch:03d} | train loss: {train_loss:.4f} | Val RMSE: {rmse:.4f}"
            print(status)
            f.write(status + "\n")
            f.flush()
            if epoch == epochs:
                torch.save(preds, os.path.join(results_dir, 'predictQM9.pt'))
                torch.save(model.state_dict(), os.path.join(model_dir, "transferQM9.pth"))
                final_preds, final_targets = preds, targets

#-----------------------------------------------------------------

figure_dir=os.path.join(working_dir, 'lib/figures')
os.makedirs(figure_dir, exist_ok=True)

y_val = final_targets
testpred = final_preds

fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].plot(y_val, testpred, '.', markersize=3, color='b')
ax[0].plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], '-k')
ax[0].set_xlabel('True Formation Energy',fontsize=20)
ax[0].set_ylabel('SchNet transfer learning prediction',fontsize=20)

ax[1].hist((testpred - y_val), bins=50,color='skyblue', edgecolor='black')
ax[1].set_xlabel("Error distribution",fontsize=20)
ax[1].set_ylabel("Freq.", fontsize=20)
plt.tight_layout() 
plt.savefig(os.path.join(figure_dir, 'plotQM9.png'),dpi=300)
plt.show()
