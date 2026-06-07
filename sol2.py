#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem import Draw

st.set_page_config(page_title="Solubility Predictor", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F2EDE1;
    }
    html, body, .stTextInput label, .stSelectbox label, .stMarkdown, .stSubheader, .stCaption, .stAlert {
        color: black !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }
    .stButton > button {
        color: white !important;
        background-color: #2c3e50 !important;
        border: none;
    }
    .stAlert {
        background-color: #d4edda !important;
        color: black !important;
    }
    div[data-testid="stAlert"] {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Aqueous Solubility Predictor")
st.markdown("Enter a SMILES string to predict logS (mol/L) using trained hybrid models.")

@st.cache_resource
def load_models():
    return {
        'RF_Morgan': joblib.load('morgan_fp_model.pkl'),
        'RF_Physics': joblib.load('physics_model.pkl'),
        'RF_Hybrid': joblib.load('hybrid_model.pkl'),
        'XGB_Morgan': joblib.load('xgb_morgan_model.pkl'),
        'XGB_Physics': joblib.load('xgb_physics_model.pkl'),
        'XGB_Hybrid': joblib.load('xgb_hybrid_model.pkl')
    }

models = load_models()

FP_SIZE = 2048
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)

def get_morgan_fingerprint(mol):
    return np.array(morgan_gen.GetFingerprint(mol))

def compute_physics_descriptors(mol):
    return {
        'MolLogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'Fsp3': rdMolDescriptors.CalcFractionCSP3(mol),
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol)
    }

def predict_solubility(smiles, model_key):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    fp = get_morgan_fingerprint(mol)
    phys = compute_physics_descriptors(mol)
    phys_arr = np.array([phys['MolLogP'], phys['TPSA'], phys['Fsp3'],
                         phys['NumAromaticRings'], phys['HBD'], phys['HBA']])
    if 'Morgan' in model_key:
        features = fp.reshape(1, -1)
    elif 'Physics' in model_key:
        features = phys_arr.reshape(1, -1)
    else:
        features = np.hstack([fp, phys_arr]).reshape(1, -1)
    pred = models[model_key].predict(features)[0]
    return pred, phys, mol

col1, col2 = st.columns(2)
with col1:
    smiles_input = st.text_input("SMILES", value="CC(=O)OC1=CC=CC=C1C(=O)O")
with col2:
    model_type = st.selectbox("Model type", ['Morgan FP', 'Physics', 'Hybrid'])
    algorithm = st.selectbox("Algorithm", ['Random Forest', 'XGBoost'])
    short_type = model_type.split()[0]
    prefix = 'RF' if algorithm == 'Random Forest' else 'XGB'
    model_key = f"{prefix}_{short_type}"

if st.button("Predict"):
    if smiles_input:
        pred, phys, mol = predict_solubility(smiles_input, model_key)
        if pred is not None:
            left_col, right_col = st.columns(2)
            with left_col:
                img = Draw.MolToImage(mol, size=(300, 300))
                st.image(img, caption="Molecule Structure", use_container_width=False)
            with right_col:
                st.markdown(f"**Predicted logS:** {pred:.3f}")
                st.subheader("Computed Physics Descriptors")
                st.json(phys)
        else:
            st.error("Invalid SMILES string.")
    else:
        st.warning("Please enter a SMILES string.")

st.markdown("---")
st.caption("Trained on combined AqSolDB + ESOL dataset (~10,000 compounds). Hybrid features: Morgan fingerprints + 6 physics descriptors.")