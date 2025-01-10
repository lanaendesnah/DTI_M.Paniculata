import pandas as pd
import requests
import time

inter_df = pd.read_csv('../../data/1-preparation/interaction/interaction_bindingdb.csv')
print("Interaksi BindingDB")

df = inter_df.drop_duplicates(subset='canonical_smiles')
print("Interaksi BindingDB")

# Kolom Canonical_Smiles yang akan digunakan
smiles_list = df['canonical_smiles']

# Function to query PubChem API
def get_compound_name_from_pubchem(smiles):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/Title/JSON"
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            compound_name = result['PropertyTable']['Properties'][0].get('Title', 'Not Found')
            return compound_name
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return str(e)

    # Proses pencarian nama senyawa dengan progres print
compound_names = []
for i, smiles in enumerate(smiles_list):
    compound_name = get_compound_name_from_pubchem(smiles)
    compound_names.append(compound_name)

    # Cetak progres
    print(f"Proses {i + 1}/{len(smiles_list)}: SMILES = {smiles}, Compound Name = {compound_name}")

    time.sleep(1)  # Jeda untuk menghindari batas permintaan API

# Menambahkan kolom hasil ke dataset asli
df['compound_name'] = compound_names

# Membuat DataFrame baru dengan hanya kolom Canonical_Smiles dan Compound_Name
result_df = df.copy()

# Menyimpan DataFrame asli yang telah diperbarui dan DataFrame baru ke CSV
result_df.to_csv("../../data/1-preparation/interaction/interaction_bindingdb_namecompound.csv", index=False)
print("Data Nama Senyawa Sudah Disimpan")