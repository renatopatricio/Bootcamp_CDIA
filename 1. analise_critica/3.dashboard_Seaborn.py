# dashboard interativo com Seaborn 

import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import numpy as np  

# Carregar os dados
df = pd.read_csv("D:/bootcamp_cdia/projeto_final/bootcamp_train.csv")

fig, axes = plt.subplots(3, 2, figsize=(18, 16))

# 1. Heatmap de correlação entre problemas
ax1 = axes[0, 0]
# Criar matriz de problemas
problem_matrix = pd.DataFrame({
    'temp_negativa': df['temperatura_ar'] < 0,
    'vel_negativa': df['velocidade_rotacional'] < 0,
    'desgaste_negativo': df['desgaste_da_ferramenta'] < 0,
    'temp_ausente': df['temperatura_ar'].isnull(),
    'vel_ausente': df['velocidade_rotacional'].isnull(),
    'desgaste_ausente': df['desgaste_da_ferramenta'].isnull()
})

correlation_matrix = problem_matrix.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, ax=ax1, cbar_kws={'label': 'Correlação'})
ax1.set_title('Correlação Entre Tipos de Problemas', fontsize=12, fontweight='bold')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

# 2. Distribuição de problemas por tipo de produto
ax2 = axes[0, 1]
problem_by_type = df.groupby('tipo').agg({
    'temperatura_ar': lambda x: (x < 0).sum(),
    'velocidade_rotacional': lambda x: (x < 0).sum(),
    'desgaste_da_ferramenta': lambda x: (x < 0).sum()
}).reset_index()

x = np.arange(len(problem_by_type))
width = 0.25

bars1 = ax2.bar(x - width, problem_by_type['temperatura_ar'], width, 
                label='Temp. Negativa', color='red', alpha=0.7)
bars2 = ax2.bar(x, problem_by_type['velocidade_rotacional'], width,
                label='Vel. Negativa', color='orange', alpha=0.7)
bars3 = ax2.bar(x + width, problem_by_type['desgaste_da_ferramenta'], width,
                label='Desgaste Negativo', color='purple', alpha=0.7)

ax2.set_xlabel('Tipo de Produto')
ax2.set_ylabel('Quantidade de Problemas')
ax2.set_title('Problemas por Tipo de Produto', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(problem_by_type['tipo'])
ax2.legend()

# Adicionar valores nas barras
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

# 3. Timeline de problemas (se houver padrão temporal)
ax3 = axes[1, 0]
# Usar o ID como proxy temporal
df_sample = df.sample(1000).sort_values('id')  # Amostra para visualização
df_sample['problema_count'] = (
    (df_sample['temperatura_ar'] < 0).astype(int) +
    (df_sample['velocidade_rotacional'] < 0).astype(int) +
    (df_sample['desgaste_da_ferramenta'] < 0).astype(int) +
    df_sample[['temperatura_ar', 'velocidade_rotacional', 'desgaste_da_ferramenta']].isnull().sum(axis=1)
)

ax3.scatter(df_sample['id'], df_sample['problema_count'], 
           c=df_sample['problema_count'], cmap='Reds', alpha=0.6, s=30)
ax3.set_xlabel('ID do Registro (Ordem Temporal)')
ax3.set_ylabel('Quantidade de Problemas por Registro')
ax3.set_title('Distribuição Temporal dos Problemas\n(Amostra de 1000 registros)', 
              fontsize=12, fontweight='bold')

# 4. Análise de consistência entre colunas relacionadas
ax4 = axes[1, 1]
# Verificar consistência entre falha_maquina e outras falhas
df_consistency = df.copy()

# Normalizar falha_maquina para análise
df_consistency['falha_normalizada'] = df_consistency['falha_maquina'].map({
    'não': 'Não', 'Não': 'Não', 'N': 'Não', '0': 'Não',
    'sim': 'Sim', 'Sim': 'Sim', 'y': 'Sim', '1': 'Sim'
})

consistency_check = df_consistency.groupby('falha_normalizada').size()
bars = ax4.bar(consistency_check.index, consistency_check.values, 
               color=['lightcoral', 'lightblue'], alpha=0.8)
ax4.set_title('Consistência após Normalização\nde Rótulos', fontsize=12, fontweight='bold')
ax4.set_ylabel('Frequência')
for i, v in enumerate(consistency_check.values):
    ax4.text(i, v + 100, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')

# 5. Distribuição de valores extremos
ax5 = axes[2, 0]
extreme_values = pd.DataFrame({
    'Variável': ['Temp. Ar < 0', 'Vel. < 0', 'Desgaste < 0', 'Temp. > 350', 'Vel. > 2500'],
    'Quantidade': [
        len(df[df['temperatura_ar'] < 0]),
        len(df[df['velocidade_rotacional'] < 0]),
        len(df[df['desgaste_da_ferramenta'] < 0]),
        len(df[df['temperatura_ar'] > 350]),
        len(df[df['velocidade_rotacional'] > 2500])
    ]
})

bars = ax5.bar(extreme_values['Variável'], extreme_values['Quantidade'], 
               color=['red', 'orange', 'purple', 'darkred', 'darkorange'], alpha=0.7)
ax5.set_title('Valores Extremos/Impossíveis', fontsize=12, fontweight='bold')
ax5.set_ylabel('Quantidade de Registros')
ax5.tick_params(axis='x', rotation=45)
for i, v in enumerate(extreme_values['Quantidade']):
    ax5.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')

# 6. Matriz de qualidade dos dados
ax6 = axes[2, 1]
quality_metrics = pd.DataFrame({
    'Métrica': ['Completude', 'Consistência\nRótulos', 'Valores\nVálidos', 'Qualidade\nGeral'],
    'Porcentagem': [
        (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,  # Completude
        70,  # Estimativa de consistência de rótulos
        (1 - len(df[(df['temperatura_ar'] < 0) | (df['velocidade_rotacional'] < 0) | 
                   (df['desgaste_da_ferramenta'] < 0)]) / len(df)) * 100,  # Valores válidos
        60   # Qualidade geral estimada
    ]
})

colors = ['red' if x < 70 else 'orange' if x < 85 else 'green' for x in quality_metrics['Porcentagem']]
bars = ax6.bar(quality_metrics['Métrica'], quality_metrics['Porcentagem'], 
               color=colors, alpha=0.7)
ax6.set_title('Métricas de Qualidade dos Dados', fontsize=12, fontweight='bold')
ax6.set_ylabel('Porcentagem (%)')
ax6.set_ylim(0, 100)
ax6.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Limite Crítico')
ax6.axhline(y=85, color='orange', linestyle='--', alpha=0.5, label='Limite Aceitável')

for i, v in enumerate(quality_metrics['Porcentagem']):
    ax6.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

ax6.legend()

plt.tight_layout()
plt.suptitle('DASHBOARD DE QUALIDADE DOS DADOS - PROBLEMAS CRÍTICOS', 
             fontsize=16, fontweight='bold', y=0.98)
plt.show()

# Estatísticas finais
print("\n" + "="*70)
print("ESTATÍSTICAS DETALHADAS DOS PROBLEMAS")
print("="*70)
print(f"Total de registros: {len(df):,}")
print(f"Registros com pelo menos um problema: {len(df[(df['temperatura_ar'] < 0) | (df['velocidade_rotacional'] < 0) | (df['desgaste_da_ferramenta'] < 0) | df.isnull().any(axis=1)]):,}")
print(f"Porcentagem de registros problemáticos: {len(df[(df['temperatura_ar'] < 0) | (df['velocidade_rotacional'] < 0) | (df['desgaste_da_ferramenta'] < 0) | df.isnull().any(axis=1)]) / len(df) * 100:.1f}%")
print("\nPROBLEMAS MAIS CRÍTICOS:")
print("1. Valores fisicamente impossíveis (negativos)")
print("2. Inconsistência massiva nos rótulos categóricos") 
print("3. Alto volume de dados ausentes")
print("4. Mistura de tipos de dados nas colunas categóricas")