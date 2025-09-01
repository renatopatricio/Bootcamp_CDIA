# gráficos específicos para problemas críticos
import matplotlib.pyplot as plt
import pandas as pd

# Carregar o DataFrame df (exemplo: de um arquivo CSV)
df = pd.read_csv('bootcamp_train.csv')  # Substitua pelo caminho correto do seu arquivo

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Análise detalhada das inconsistências de rótulos
ax1 = axes[0, 0]
# Agrupar todas as variações de "não/false"
falha_variations = df['falha_maquina'].value_counts()
no_variations = ['não', 'Não', 'N', '0']
yes_variations = ['sim', 'Sim', 'y', '1']

no_count = sum([falha_variations.get(var, 0) for var in no_variations])
yes_count = sum([falha_variations.get(var, 0) for var in yes_variations])

labels = ['Variações de "NÃO"', 'Variações de "SIM"']
sizes = [no_count, yes_count]
colors = ['lightcoral', 'lightblue']

wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                   colors=colors, startangle=90)
ax1.set_title('Problema: Múltiplas Representações\npara Mesma Categoria', 
              fontsize=12, fontweight='bold')

# Adicionar detalhes das variações
detail_text = f"Variações 'NÃO': {no_variations}\nVariações 'SIM': {yes_variations}"
ax1.text(1.3, 0, detail_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

# 2. Valores fisicamente impossíveis
ax2 = axes[0, 1]
# Destacar valores negativos
temp_clean = df['temperatura_ar'].dropna()
temp_negative = temp_clean[temp_clean < 0]
temp_positive = temp_clean[temp_clean >= 0]

ax2.hist(temp_positive, bins=30, alpha=0.7, color='skyblue', label='Valores Normais', edgecolor='black')
ax2.hist(temp_negative, bins=10, alpha=0.9, color='red', label='Valores Impossíveis (<0°C)', edgecolor='black')
ax2.set_title('Problema: Temperaturas Fisicamente\nImpossíveis', fontsize=12, fontweight='bold')
ax2.set_xlabel('Temperatura do Ar (°C)')
ax2.set_ylabel('Frequência')
ax2.legend()
ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.8)

# 3. Padrão de valores ausentes
ax3 = axes[1, 0]
missing_pattern = df.isnull().sum().sort_values(ascending=False)
missing_pattern = missing_pattern[missing_pattern > 0]

bars = ax3.barh(range(len(missing_pattern)), missing_pattern.values, color='orange', alpha=0.7)
ax3.set_yticks(range(len(missing_pattern)))
ax3.set_yticklabels([col.replace(' ', '\n') for col in missing_pattern.index], fontsize=10)
ax3.set_title('Problema: Padrão de Dados Ausentes', fontsize=12, fontweight='bold')
ax3.set_xlabel('Quantidade de Valores Ausentes')

# Adicionar valores nas barras
for i, v in enumerate(missing_pattern.values):
    ax3.text(v + 10, i, f'{v}\n({v/len(df)*100:.1f}%)', va='center', fontsize=9)

# 4. Outliers extremos em múltiplas variáveis
ax4 = axes[1, 1]
# Identificar registros com múltiplos problemas
problematic_records = df[
    (df['temperatura_ar'] < 0) | 
    (df['velocidade_rotacional'] < 0) | 
    (df['desgaste_da_ferramenta'] < 0)
].copy()

problem_types = []
if len(df[df['temperatura_ar'] < 0]) > 0:
    problem_types.append(f"Temp. Negativa\n({len(df[df['temperatura_ar'] < 0])})")
if len(df[df['velocidade_rotacional'] < 0]) > 0:
    problem_types.append(f"Vel. Negativa\n({len(df[df['velocidade_rotacional'] < 0])})")
if len(df[df['desgaste_da_ferramenta'] < 0]) > 0:
    problem_types.append(f"Desgaste Negativo\n({len(df[df['desgaste_da_ferramenta'] < 0])})")

problem_counts = [
    len(df[df['temperatura_ar'] < 0]),
    len(df[df['velocidade_rotacional'] < 0]), 
    len(df[df['desgaste_da_ferramenta'] < 0])
]

bars = ax4.bar(problem_types, problem_counts, color=['red', 'orange', 'purple'], alpha=0.7)
ax4.set_title('Problema: Valores Negativos\n(Fisicamente Impossíveis)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Quantidade de Registros')
for i, v in enumerate(problem_counts):
    ax4.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.suptitle('PROBLEMAS CRÍTICOS DETECTADOS NO DATASET', fontsize=16, fontweight='bold', y=0.98)
plt.show()

# Resumo dos problemas
print("\n" + "="*60)
print("RESUMO DOS PROBLEMAS DETECTADOS")
print("="*60)
print(f"1. VALORES AUSENTES: {df.isnull().sum().sum()} valores ausentes total")
print(f"2. INCONSISTÊNCIAS DE RÓTULOS: Múltiplas representações para mesmos valores")
print(f"3. VALORES FISICAMENTE IMPOSSÍVEIS:")
print(f"   - Temperaturas negativas: {len(df[df['temperatura_ar'] < 0])} registros")
print(f"   - Velocidades negativas: {len(df[df['velocidade_rotacional'] < 0])} registros") 
print(f"   - Desgaste negativo: {len(df[df['desgaste_da_ferramenta'] < 0])} registros")
print(f"4. TIPOS DE DADOS INCONSISTENTES: Mistura de boolean, string e numeric")