# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 10:57:36 2026

@author: willi
"""

import json
import numpy as np
import pandas as pd
from ase import Atoms
from ase.data import chemical_symbols, covalent_radii, atomic_masses
from ase.neighborlist import neighbor_list
from collections import Counter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# --- 1. Robust Data & Fallbacks ---
en_map = {
    'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
    'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55,
    'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01,
    'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33,
    'Nb': 1.6, 'Mo': 2.16, 'Ag': 1.93, 'Cd': 1.69, 'In': 1.78, 'Sn': 1.96, 'Sb': 2.05,
    'Te': 2.1, 'I': 2.66, 'Cs': 0.79, 'Ba': 0.89, 'Pt': 2.28, 'Au': 2.54, 'Pb': 2.33
}

# --- 2. Load Data ---
json_train = r"C:\Users\willi\Desktop\DTU\Master\Materials design with machine learning and AI\train.json"
json_test  = r"C:\Users\willi\Desktop\DTU\Master\Materials design with machine learning and AI\test.json"

def atoms_from_record(atoms_rec):
    if isinstance(atoms_rec, str): atoms_rec = json.loads(atoms_rec)
    return Atoms(numbers=atoms_rec["numbers"], positions=atoms_rec["positions"], 
                 cell=atoms_rec.get("cell"), pbc=atoms_rec.get("pbc", [True, True, True]))

df_all = pd.read_json(json_train).iloc[:8000] 
df_test = pd.read_json(json_test)

atoms_train = [atoms_from_record(r) for r in df_all["atoms"]]
atoms_test = [atoms_from_record(r) for r in df_test["atoms"]]
y_all = df_all["hform"].values

nums = set()
for a in atoms_train + atoms_test: nums.update(a.get_atomic_numbers())
species = sorted({chemical_symbols[n] for n in nums})

# --- 3. Refined Feature Engineering ---
def improved_feat(a):
    n_atoms = len(a)
    syms = [chemical_symbols[n] for n in a.get_atomic_numbers()]
    counts = Counter(syms)
    
    # A. Composition (Atomic Fraction instead of raw counts)
    # This helps the model generalize across different unit cell sizes
    comp = [counts.get(s, 0) / n_atoms for s in species]
    
    # B. Global Structure
    vol_per_atom = a.get_volume() / n_atoms
    
    # C. Coordination Number
    i, j = neighbor_list('ij', a, cutoff=3.5, self_interaction=False)
    avg_coordination = len(i) / n_atoms if n_atoms > 0 else 0
    
    # D. Enhanced Chemical Properties
    en_list = [en_map.get(s, 2.0) for s in syms]
    radii_list = [covalent_radii[n] for n in a.get_atomic_numbers()]
    
    mean_en = np.mean(en_list)
    max_en_diff = np.max(en_list) - np.min(en_list) # Strong indicator of ionic stability
    std_radius = np.std(radii_list) # Lattice strain indicator
    
    # E. Smoother RDF (Lower Bins to reduce feature noise)
    if n_atoms > 1:
        dm = a.get_all_distances(mic=True)
        dists = dm[np.triu_indices(n_atoms, k=1)]
        # Reduced to 15 bins to make the structural signal more robust
        rdf_hist, _ = np.histogram(dists, bins=15, range=(1.0, 6.0))
        rdf_feat = rdf_hist / n_atoms
    else:
        rdf_feat = np.zeros(15)

    return np.array(comp + [vol_per_atom, avg_coordination, mean_en, max_en_diff, std_radius] + rdf_feat.tolist(), dtype=float)

X_all = np.vstack([improved_feat(a) for a in atoms_train])
X_test_final = np.vstack([improved_feat(a) for a in atoms_test])

# --- 4. Split, Scale, and Train ---
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.15, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_final)

# --- 5. Revised GPR Strategy ---
# We use a larger length_scale to avoid overfitting "noisy" structural details
# alpha is increased to 0.05 to regularize the model (make it smoother)
kernel = ConstantKernel(1.0) * RBF(length_scale=5.0) + WhiteKernel(noise_level=1e-3)

gpr = GaussianProcessRegressor(
    kernel=kernel, 
    normalize_y=True, 
    alpha=0.05, 
    n_restarts_optimizer=2
)

print(f"Training on {len(X_train)} structures...")
gpr.fit(X_train_scaled, y_train)

y_pred_val = gpr.predict(X_val_scaled)
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"\nFinal Validation RMSE: {val_rmse:.4f}")

# Save Submission
y_pred_test = gpr.predict(X_test_scaled)
pd.DataFrame({"id": df_test["id"], "hform": y_pred_test}).to_csv("submission.csv", index=False)



#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Update this line in your main script
X_train, X_val, y_train, y_val, atoms_v_train, atoms_v_val = train_test_split(
    X_all, y_all, atoms_train, test_size=0.15, random_state=42
)

def run_diagnostic_analysis(gpr_model, X_scaled, atoms_list, target_element='O'):
    """
    Standalone function to perform PCA clustering visualization 
    and GPR Hyperparameter LML landscape analysis.
    """
    
    # --- Part 1: PCA Biplot (Feature Space) ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Identify structures containing the target element for color-coding
    has_el = np.array([target_element in a.get_chemical_symbols() for a in atoms_list])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # PCA Scatter Plot
    scatter = ax1.scatter(X_pca[~has_el, 0], X_pca[~has_el, 1], c='lightgrey', alpha=0.4, s=15, label=f'Other')
    scatter2 = ax1.scatter(X_pca[has_el, 0], X_pca[has_el, 1], c='teal', alpha=0.7, s=20, label=f'Contains {target_element}')
    
    ax1.set_title(f'PCA Feature Map\n(Explains {np.sum(pca.explained_variance_ratio_)*100:.1f}% Variance)')
    ax1.set_xlabel(f'Principal Component 1')
    ax1.set_ylabel(f'Principal Component 2')
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # --- Part 2: LML Contour Plot (Hyperparameter Space) ---
    # We examine the Relationship between Signal Variance (ConstantKernel) 
    # and Length Scale (RBF). 
    
    # Create a grid of hyperparameter values (log-space)
    # Adjust ranges based on your kernel's typical values
    sig_var_range = np.logspace(-2, 4, 30)   # Constant Kernel / Signal Variance
    length_scale_range = np.logspace(-1, 2, 30) # RBF Length Scale
    
    V, L = np.meshgrid(sig_var_range, length_scale_range)
    
    # Compute Log-Marginal Likelihood (LML) for each point on the grid
    # We assume a kernel structure: ConstantKernel * RBF + WhiteKernel
    # Note: We fix the WhiteKernel noise at 1e-3 for the visualization
    lml_grid = np.array([
        gpr_model.log_marginal_likelihood(np.log([v, l, 1e-3])) 
        for v, l in zip(V.ravel(), L.ravel())
    ]).reshape(V.shape)

    # Plot Contours
    cp = ax2.contourf(V, L, lml_grid, levels=40, cmap='RdGy_r')
    fig.colorbar(cp, ax=ax2, label='Log-Marginal Likelihood')
    
    # Mark the optimal hyperparameters found during .fit()
    opt_params = np.exp(gpr_model.kernel_.theta)
    ax2.scatter(opt_params[0], opt_params[1], color='yellow', marker='*', s=200, 
                edgecolor='black', label='Optimized Point')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('GPR Hyperparameter Landscape\n(Log-Marginal Likelihood)')
    ax2.set_xlabel('Signal Variance (Constant Kernel)')
    ax2.set_ylabel('Length Scale (RBF)')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# --- HOW TO CALL ---
# Ensure you use the scaled training data and the list of Atoms objects used for training
run_diagnostic_analysis(gpr, X_train_scaled, atoms_v_train, target_element='O')





