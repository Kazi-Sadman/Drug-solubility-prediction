import streamlit as st
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

# Load model and preprocessing objects
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Drug Solubility Prediction System")

smiles = st.text_input("Enter SMILES String")

def extract_features(smiles, radius=2, n_bits=256):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    features = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol)
    ]

    fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    features.extend(int(bit) for bit in fp)
    return features

if st.button("Predict"):
    feature_vector = extract_features(smiles)

    if feature_vector is None:
        st.error("Invalid SMILES String")
    else:
        features_scaled = scaler.transform([feature_vector])
        prediction = model.predict(features_scaled)
        predicted_label = encoder.inverse_transform(prediction)[0]

        if predicted_label == "High":
            st.success(f"✅ Predicted Solubility Class: {predicted_label}")
        elif predicted_label == "Medium":
            st.warning(f"⚠️ Predicted Solubility Class: {predicted_label}")
        else:  # Low
            st.error(f"❌ Predicted Solubility Class: {predicted_label}")