import torch
import os

model_dir = '/work3/s214471/materials/weights'
save_path = os.path.join(model_dir, "schnet_qm9_weights.pth")

model = torch.hub.load('atomistic-machine-learning/schnetpack:v1.0.0', 'schnet_qm9', pretrained=True, trust_repo=True)
torch.save(model.state_dict(), save_path)
print('\n--- FIL GEMT SOM schnet_qm9_weights.pth ---')


