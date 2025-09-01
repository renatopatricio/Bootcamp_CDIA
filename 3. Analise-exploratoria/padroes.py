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

# 5. DISTRIBUIÇÃO E PADRÕES
print("\n\n5. DISTRIBUIÇÃO E PADRÕES")
print("-"*50)

# Análise da umidade relativa
print("Análise da umidade_relativa:")
print(f"Valor mínimo: {df['umidade_relativa'].min():.2f}")
print(f"Valor máximo: {df['umidade_relativa'].max():.2f}")
print(f"Valor mais frequente: {df['umidade_relativa'].mode()[0]:.1f}")
print(f"Percentual de registros com umidade = 90: {(df['umidade_relativa'] == 90).mean()*100:.1f}%")

# Análise da temperatura do processo
print(f"\nAnálise da temperatura_processo:")
temp_processo_clean = df['temperatura_processo'].dropna()
print(f"Faixa principal: {temp_processo_clean.quantile(0.25):.1f}°C a {temp_processo_clean.quantile(0.75):.1f}°C")
print(f"Média: {temp_processo_clean.mean():.1f}°C")

# Análise do torque
print(f"\nAnálise do torque:")
torque_clean = df['torque'].dropna()
print(f"Variação: {torque_clean.min():.1f} a {torque_clean.max():.1f}")
print(f"Desvio padrão: {torque_clean.std():.1f}")

# Análise do desgaste da ferramenta
print(f"\nAnálise do desgaste_da_ferramenta:")
desgaste_clean = df['desgaste_da_ferramenta'].dropna()
print(f"Variação: {desgaste_clean.min():.1f} a {desgaste_clean.max():.1f}")
print(f"Valores negativos (incoerentes): {(desgaste_clean < 0).sum()}")

# 6. RÓTULOS DAS CLASSES
print("\n\n6. ANÁLISE DOS RÓTULOS DAS CLASSES")
print("-"*50)

# Padronizar falha_maquina para análise
def padronizar_falha(valor):
    if pd.isna(valor):
        return 'Desconhecido'
    valor_str = str(valor).lower().strip()
    if valor_str in ['não', 'n', '0']:
        return 'Não'
    elif valor_str in ['sim', 'y', '1']:
        return 'Sim'
    else:
        return valor_str

df['falha_maquina_padronizada'] = df['falha_maquina'].apply(padronizar_falha)

print("Distribuição da falha_maquina (padronizada):")
falha_dist = df['falha_maquina_padronizada'].value_counts()
print(falha_dist)
print(f"\nPercentual de falhas: {(falha_dist['Sim'] / len(df))*100:.2f}%")
print(f"Percentual sem falhas: {(falha_dist['Não'] / len(df))*100:.2f}%")

# Análise das falhas específicas
# Falhas específicas
falhas_especificas = ['FDF (Falha Desgaste Ferramenta)', 'FDC (Falha Dissipacao Calor)', 
                     'FP (Falha Potencia)', 'FTE (Falha Tensao Excessiva)', 'FA (Falha Aleatoria)']
print(f"Falhas específicas: {falhas_especificas}")

print("\n\nDistribuição das falhas específicas:")
for col in falhas_especificas:
    if col in df.columns:
        valores_unicos = df[col].value_counts()
        print(f"\n{col}:")
        print(valores_unicos)
        
        # Contar valores que indicam falha (True, 'True', 'Sim', etc.)
        falhas_positivas = 0
        for valor, count in valores_unicos.items():
            if str(valor).lower() in ['true', 'sim', '1', 'y']:
                falhas_positivas += count
        
        print(f"Registros com falha: {falhas_positivas} ({(falhas_positivas/len(df))*100:.3f}%)")

# 7. ANÁLISE DO TIPO DE MÁQUINA
print("\n\n7. ANÁLISE DO TIPO DE MÁQUINA")
print("-"*50)
print("Distribuição por tipo:")
tipo_dist = df['tipo'].value_counts()
print(tipo_dist)
print(f"\nPercentual por tipo:")
for tipo, count in tipo_dist.items():
    print(f"Tipo {tipo}: {(count/len(df))*100:.1f}%")