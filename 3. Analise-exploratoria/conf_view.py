import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuração de visualização
plt.style.use('default')
sns.set_palette("husl")

# Carregar os dados
df = pd.read_csv("D:/bootcamp_cdia/projeto_final/bootcamp_train.csv")

print("="*80)
print("ANÁLISE EXPLORATÓRIA DE DADOS (EDA) - BOOTCAMP TRAIN")
print("="*80)

# 1. ESTRUTURA DO DATASET
print("\n1. ESTRUTURA DO DATASET")
print("-"*50)
print(f"Dimensões do dataset: {df.shape}")
print(f"Número de linhas: {df.shape[0]:,}")
print(f"Número de colunas: {df.shape[1]}")

print("\nTipos de dados:")
print(df.dtypes)

print("\nPrimeiras 5 linhas:")
print(df.head())

print("\nInformações gerais:")
print(df.info())

# 2. COLUNAS PRINCIPAIS
print("\n\n2. COLUNAS PRINCIPAIS")
print("-"*50)

# Identificadores
identificadores = ['id', 'id_produto', 'tipo']
print(f"Identificadores: {identificadores}")

# Sensores/variáveis contínuas
sensores = ['temperatura_ar', 'temperatura_processo', 'umidade_relativa', 
           'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta']
print(f"Sensores/Variáveis contínuas: {sensores}")

# Alvo principal
alvo_principal = 'falha_maquina'
print(f"Alvo principal: {alvo_principal}")

# Falhas específicas
falhas_especificas = ['FDF (Falha Desgaste Ferramenta)', 'FDC (Falha Dissipacao Calor)', 
                     'FP (Falha Potencia)', 'FTE (Falha Tensao Excessiva)', 'FA (Falha Aleatoria)']
print(f"Falhas específicas: {falhas_especificas}")

# 3. QUALIDADE DOS DADOS
print("\n\n3. QUALIDADE DOS DADOS")
print("-"*50)

# Valores faltantes
print("Valores faltantes por coluna:")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Valores Faltantes': missing_values,
    'Percentual (%)': missing_percent.round(2)
})
print(missing_df[missing_df['Valores Faltantes'] > 0])

# Verificar valores incoerentes
print("\n\nVerificação de valores incoerentes:")

# Temperatura do ar negativa
temp_ar_negativa = df[df['temperatura_ar'] < 0]
print(f"Registros com temperatura_ar negativa: {len(temp_ar_negativa)}")
if len(temp_ar_negativa) > 0:
    print(f"Valores mínimo e máximo: {df['temperatura_ar'].min():.1f}°C a {df['temperatura_ar'].max():.1f}°C")

# Velocidade rotacional negativa
vel_negativa = df[df['velocidade_rotacional'] < 0]
print(f"Registros com velocidade_rotacional negativa: {len(vel_negativa)}")
if len(vel_negativa) > 0:
    print(f"Valores mínimo e máximo: {df['velocidade_rotacional'].min():.1f} a {df['velocidade_rotacional'].max():.1f}")

# Inconsistência de rótulos na coluna falha_maquina
print(f"\nValores únicos em 'falha_maquina': {df['falha_maquina'].unique()}")
print("Contagem de valores:")
print(df['falha_maquina'].value_counts())

# 4. ESTATÍSTICAS DESCRITIVAS
print("\n\n4. ESTATÍSTICAS DESCRITIVAS - SENSORES")
print("-"*50)
print(df[sensores].describe())