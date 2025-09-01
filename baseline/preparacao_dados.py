import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Recarregar dataset bruto

df = pd.read_csv("D:/bootcamp_cdia/projeto_final_2/dataset_limpo_final.csv")

# Padronizar rótulos de falha (binário)
df['falha_maquina'] = df['falha_maquina'].astype(str).str.strip().str.lower()
df['falha_maquina'] = df['falha_maquina'].replace({'não':0, 'nao':0, 'n':0, 'false':0,
                                                   'sim':1, 's':1, 'true':1})

# Forçar conversão numérica nas colunas de sensores (valores inválidos viram NaN)
for col in ['temperatura_ar','temperatura_processo','umidade_relativa',
            'velocidade_rotacional','torque','desgaste_da_ferramenta']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Corrigir valores incoerentes: negativos irreais
df['temperatura_ar'] = df['temperatura_ar'].apply(lambda x: np.nan if pd.notna(x) and x < 0 else x)
df['velocidade_rotacional'] = df['velocidade_rotacional'].apply(lambda x: np.nan if pd.notna(x) and x < 0 else x)

# Substituir NaN pela mediana
for col in ['temperatura_ar','temperatura_processo','umidade_relativa',
            'velocidade_rotacional','torque','desgaste_da_ferramenta']:
    df[col] = df[col].fillna(df[col].median())

# Salvar dataset limpo atualizado
df.to_csv("dataset_limpo_atualizado.csv", index=False)

# Features e Target
X = df[['temperatura_ar','temperatura_processo','umidade_relativa',
        'velocidade_rotacional','torque','desgaste_da_ferramenta']]
y = df['falha_maquina']

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                    test_size=0.3,
                                                    random_state=42,
                                                    stratify=y)

# Modelo baseline
clf = RandomForestClassifier(random_state=42, n_estimators=200)
clf.fit(X_train, y_train)

# Avaliação
y_pred = clf.predict(X_test)
results = {
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "classification_report": classification_report(y_test, y_pred, output_dict=True)
}
results