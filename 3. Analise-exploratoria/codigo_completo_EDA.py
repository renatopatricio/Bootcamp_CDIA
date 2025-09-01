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

# 8. VISUALIZAÇÕES
print("\n\n8. VISUALIZAÇÕES")
print("-"*50)

# Criar figura com subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Análise Exploratória - Bootcamp Train Dataset', fontsize=16, fontweight='bold')

# 1. Distribuição da falha principal
ax1 = axes[0, 0]
falha_counts = df['falha_maquina_padronizada'].value_counts()
ax1.pie(falha_counts.values, labels=falha_counts.index, autopct='%1.1f%%', startangle=90)
ax1.set_title('Distribuição de Falhas\n(Alvo Principal)')

# 2. Distribuição por tipo de máquina
ax2 = axes[0, 1]
tipo_counts = df['tipo'].value_counts()
colors = ['skyblue', 'lightcoral', 'lightgreen']
ax2.bar(tipo_counts.index, tipo_counts.values, color=colors)
ax2.set_title('Distribuição por Tipo de Máquina')
ax2.set_ylabel('Quantidade')

# 3. Histograma temperatura_ar (com outliers)
ax3 = axes[0, 2]
temp_ar_clean = df['temperatura_ar'].dropna()
ax3.hist(temp_ar_clean, bins=50, alpha=0.7, color='orange')
ax3.set_title('Distribuição Temperatura do Ar')
ax3.set_xlabel('Temperatura (°C)')
ax3.set_ylabel('Frequência')
ax3.axvline(0, color='red', linestyle='--', alpha=0.7, label='0°C')
ax3.legend()

# 4. Histograma temperatura_processo
ax4 = axes[1, 0]
temp_processo_clean = df['temperatura_processo'].dropna()
ax4.hist(temp_processo_clean, bins=50, alpha=0.7, color='green')
ax4.set_title('Distribuição Temperatura do Processo')
ax4.set_xlabel('Temperatura (°C)')
ax4.set_ylabel('Frequência')

# 5. Boxplot do torque por tipo de máquina
ax5 = axes[1, 1]
df_torque_clean = df.dropna(subset=['torque'])
sns.boxplot(data=df_torque_clean, x='tipo', y='torque', ax=ax5)
ax5.set_title('Torque por Tipo de Máquina')

# 6. Scatter plot: velocidade vs torque
ax6 = axes[1, 2]
df_scatter = df.dropna(subset=['velocidade_rotacional', 'torque'])
scatter = ax6.scatter(df_scatter['velocidade_rotacional'], df_scatter['torque'], 
                     c=df_scatter['falha_maquina_padronizada'].map({'Não': 0, 'Sim': 1}), 
                     alpha=0.6, cmap='coolwarm')
ax6.set_title('Velocidade vs Torque\n(Cor: Falha)')
ax6.set_xlabel('Velocidade Rotacional')
ax6.set_ylabel('Torque')

# 7. Histograma desgaste da ferramenta
ax7 = axes[2, 0]
desgaste_clean = df['desgaste_da_ferramenta'].dropna()
ax7.hist(desgaste_clean, bins=50, alpha=0.7, color='purple')
ax7.set_title('Distribuição Desgaste da Ferramenta')
ax7.set_xlabel('Desgaste')
ax7.set_ylabel('Frequência')
ax7.axvline(0, color='red', linestyle='--', alpha=0.7, label='0')
ax7.legend()

# 8. Heatmap de correlação
ax8 = axes[2, 1]
# Selecionar apenas colunas numéricas para correlação
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax8, fmt='.2f')
ax8.set_title('Matriz de Correlação\n(Variáveis Numéricas)')

# 9. Distribuição das falhas específicas
ax9 = axes[2, 2]
falhas_counts = []
falhas_names = []

for col in falhas_especificas:
    if col in df.columns:
        # Contar valores que indicam falha
        falhas_positivas = 0
        for valor in df[col].unique():
            if str(valor).lower() in ['true', 'sim', '1', 'y']:
                falhas_positivas += (df[col] == valor).sum()
        
        falhas_counts.append(falhas_positivas)
        falhas_names.append(col.split('(')[0].strip())

ax9.bar(range(len(falhas_names)), falhas_counts, color='red', alpha=0.7)
ax9.set_title('Distribuição das Falhas Específicas')
ax9.set_xlabel('Tipo de Falha')
ax9.set_ylabel('Quantidade')
ax9.set_xticks(range(len(falhas_names)))
ax9.set_xticklabels(falhas_names, rotation=45, ha='right')

plt.tight_layout()
plt.show()

# FUNÇÃO PARA PADRONIZAR RÓTULOS
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

# RESUMO DOS PROBLEMAS IDENTIFICADOS
print("\n\n9. RESUMO DOS PROBLEMAS IDENTIFICADOS")
print("-"*50)
print("✗ Valores incoerentes:")
print(f"  - {len(df[df['temperatura_ar'] < 0]):,} registros com temperatura_ar negativa")
print(f"  - {len(df[df['velocidade_rotacional'] < 0]):,} registros com velocidade_rotacional negativa") 
print(f"  - {(df['desgaste_da_ferramenta'] < 0).sum():,} registros com desgaste_da_ferramenta negativo")

print(f"\n✗ Valores faltantes:")
for col, missing in missing_df[missing_df['Valores Faltantes'] > 0].iterrows():
    print(f"  - {col}: {missing['Valores Faltantes']:,} ({missing['Percentual (%)']:.2f}%)")

print(f"\n✗ Inconsistência de rótulos:")
print(f"  - falha_maquina tem {len(df['falha_maquina'].unique())} valores únicos diferentes")
print(f"  - Falhas específicas usam formatos mistos (True/False, Sim/Não, 0/1, etc.)")

falha_dist = df['falha_maquina_padronizada'].value_counts()
print(f"\n✗ Desbalanceamento extremo:")
print(f"  - Apenas {(falha_dist['Sim'] / len(df))*100:.2f}% dos registros têm falha")

print(f"\n✓ Observações importantes:")
print(f"  - umidade_relativa praticamente constante (99.9% = 90)")
print(f"  - temperatura_processo em faixa estável (308-311°C)")
print(f"  - Tipos de máquina: L (67.7%), M (25.0%), H (7.4%)")

print("\n" + "="*80)
print("PRÓXIMOS PASSOS RECOMENDADOS:")
print("="*80)
print("1. Limpeza de dados: remover/corrigir valores incoerentes")
print("2. Padronização de rótulos: unificar formato das classes")
print("3. Tratamento de valores faltantes: imputação ou remoção")
print("4. Estratégias para desbalanceamento: SMOTE, class_weight, etc.")
print("5. Feature engineering: criar variáveis derivadas")
print("6. Definir estratégia de modelagem: multiclasse vs multirrótulo")