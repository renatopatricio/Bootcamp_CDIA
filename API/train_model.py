import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df):
    """Preprocessa os dados do dataset"""
    data = df.copy()

    # Padronizar a variável target (falha_maquina)
    falha_mapping = {
        'Sim': 1, 'sim': 1, 'y': 1, '1': 1,
        'Não': 0, 'não': 0, 'N': 0, '0': 0
    }
    data['falha_maquina'] = data['falha_maquina'].map(falha_mapping)

    # Tratar valores especiais nas features numéricas
    numeric_cols = ['temperatura_ar', 'temperatura_processo', 'velocidade_rotacional', 
                   'torque', 'desgaste_da_ferramenta']

    for col in numeric_cols:
        # Substituir valores especiais por NaN
        data[col] = data[col].replace([-36, -38, -161, -202], np.nan)
        # Preencher com mediana
        data[col] = data[col].fillna(data[col].median())

    # Codificar variável categórica 'tipo'
    le_tipo = LabelEncoder()
    data['tipo_encoded'] = le_tipo.fit_transform(data['tipo'])

    # Padronizar colunas de falha
    falha_cols = ['FDF (Falha Desgaste Ferramenta)', 'FDC (Falha Dissipacao Calor)', 
                  'FP (Falha Potencia)', 'FA (Falha Aleatoria)']

    for col in falha_cols:
        data[col] = data[col].astype(str).str.lower()
        data[col] = data[col].map({
            'true': 1, 'false': 0, 'sim': 1, 'não': 0, 'nao': 0, 
            'n': 0, '0': 0, '1': 1, '-': 0
        }).fillna(0).astype(int)

    # Converter FTE para int
    data['FTE (Falha Tensao Excessiva)'] = data['FTE (Falha Tensao Excessiva)'].astype(int)

    return data, le_tipo

def train_stacking_model(csv_file='bootcamp_train.csv'):
    """Treina o modelo Stacking Ensemble"""
    print("Carregando dataset...")
    df = pd.read_csv(csv_file)

    print("Preprocessando dados...")
    data_processed, le_tipo = preprocess_data(df)

    # Preparar features e target
    feature_cols = ['temperatura_ar', 'temperatura_processo', 'umidade_relativa',
                   'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta', 'tipo_encoded',
                   'FDF (Falha Desgaste Ferramenta)', 'FDC (Falha Dissipacao Calor)', 
                   'FP (Falha Potencia)', 'FTE (Falha Tensao Excessiva)', 'FA (Falha Aleatoria)']

    X = data_processed[feature_cols]
    y = data_processed['falha_maquina']

    print(f"Dataset shape: {X.shape}")
    print(f"Distribuição target: {y.value_counts()}")

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Normalizar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Definir modelos base
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(probability=True, random_state=42, class_weight='balanced'))
    ]

    # Meta-modelo
    meta_model = LogisticRegression(random_state=42, class_weight='balanced')

    # Criar Stacking Classifier
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )

    print("Treinando modelo Stacking Ensemble...")
    stacking_model.fit(X_train_scaled, y_train)

    # Avaliar modelo
    y_pred = stacking_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAcurácia: {accuracy:.4f}")
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))

    # Salvar modelos
    joblib.dump(stacking_model, 'stacking_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le_tipo, 'label_encoder.pkl')

    print("\nModelos salvos:")
    print("- stacking_model.pkl")
    print("- scaler.pkl")
    print("- label_encoder.pkl")

    return stacking_model, scaler, le_tipo

if __name__ == "__main__":
    train_stacking_model()
