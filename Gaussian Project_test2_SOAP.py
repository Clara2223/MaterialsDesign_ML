# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 15:09:42 2026

@author: willi
"""

import json
import numpy as np
import pandas as pd
import time
from ase import Atoms
from dscribe.descriptors import SOAP
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# --- 1. Data Loading ---
json_train = r"C:\Users\willi\Desktop\DTU\Master\Materials design with machine learning and AI\train.json"
json_test  = r"C:\Users\willi\Desktop\DTU\Master\Materials design with machine learning and AI\test.json"

def atoms_from_record(atoms_rec):
    if isinstance(atoms_rec, str): atoms_rec = json.loads(atoms_rec)
    return Atoms(numbers=atoms_rec["numbers"], 
                 positions=atoms_rec["positions"], 
                 cell=atoms_rec.get("cell"), 
                 pbc=atoms_rec.get("pbc", [True, True, True]))

df_train = pd.read_json(json_train).iloc[:8000] # Full dataset
df_test = pd.read_json(json_test)

atoms_train = [atoms_from_record(r) for r in df_train["atoms"]]
atoms_test = [atoms_from_record(r) for r in df_test["atoms"]]
y_all = df_train["hform"].values

# --- 2. SOAP Descriptor Setup ---
# We need the list of all unique elements across both sets
all_species = sorted(list(set([sym for a in atoms_train + atoms_test for sym in a.get_chemical_symbols()])))

soap = SOAP(
    species=all_species,
    periodic=True,
    rcut=5.0,           # Interaction radius (5.0 Angstroms is standard)
    nmax=8,             # Number of radial basis functions
    lmax=6,             # Degree of spherical harmonics
    average="inner",    # Average local environments into one global crystal fingerprint
    sparse=False
)



print(f"Generating SOAP features for {len(atoms_train)} structures...")
start_feat = time.time()
X_all_soap = soap.create(atoms_train, n_jobs=-1) 
X_test_soap = soap.create(atoms_test, n_jobs=-1)
print(f"Feature extraction took {time.time() - start_feat:.2f} seconds.")

# --- 3. Split and Scale ---
X_train, X_val, y_train, y_val = train_test_split(X_all_soap, y_all, test_size=0.15, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_soap)

# --- 4. GPR with DotProduct Kernel ---
# For high-dimensional SOAP vectors, DotProduct (linear/polynomial) often 
# performs better and faster than RBF.
kernel = ConstantKernel(1.0) * DotProduct(sigma_0=1.0)**2 + WhiteKernel(noise_level=0.1)

gpr = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=2
)

print("Starting GPR training...")
start_train = time.time()
gpr.fit(X_train_scaled, y_train)
print(f"Training took {time.time() - start_train:.2f} seconds.")

# --- 5. Evaluation ---
y_pred_val = gpr.predict(X_val_scaled)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
mae = mean_absolute_error(y_val, y_pred_val)

print(f"\nSOAP + GPR RESULTS:")
print(f"Validation MAE:  {mae:.4f} eV/atom")
print(f"Validation RMSE: {rmse:.4f} eV/atom")

# --- 6. Final Submission ---
y_pred_test = gpr.predict(X_test_scaled)
pd.DataFrame({"id": df_test["id"], "hform": y_pred_test}).to_csv("submission_csv", index=False)