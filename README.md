# Aqueous Solubility Prediction

A machine learning framework for predicting aqueous solubility (logS) from molecular structure. This project compares different molecular representations and ML models to accurately predict solubility from SMILES strings.

## Overview

Aqueous solubility is a critical physicochemical property in drug discovery, environmental chemistry, and material science. Experimental measurement is time-consuming and expensive, driving the need for reliable computational models.

This project implements and evaluates:
- **3 molecular feature sets**: Morgan fingerprints, physicochemical descriptors, and a hybrid combination
- **2 ML algorithms**: Random Forest and XGBoost
- **Model interpretability**: SHAP analysis to understand feature importance


### Key Findings

**Hybrid features outperform individual representations** - Combining structural fingerprints with physicochemical descriptors improves predictions (R² 0.783 vs 0.683)
**Random Forest outperforms XGBoost** on this dataset
**MolLogP is the most important feature** - reflects the centrality of hydrophobicity in solubility
**Model learns chemically meaningful relationships** - SHAP analysis confirms chemical intuition

