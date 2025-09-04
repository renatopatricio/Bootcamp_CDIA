# 📊 Análise Exploratória de Dados (EDA) - Bootcamp Train

Este projeto realiza uma **Análise Exploratória de Dados (EDA)** no
dataset `bootcamp_train.csv`, com foco em identificar problemas de
qualidade, padrões relevantes e insights iniciais para posterior
modelagem de falhas em máquinas industriais.

------------------------------------------------------------------------

## 🚀 Funcionalidades

O script realiza:

1.  **Estrutura do Dataset**
    -   Exibe dimensões, tipos de dados, primeiras linhas e informações
        gerais.
2.  **Identificação de Colunas**
    -   Identificadores (`id`, `id_produto`, `tipo`)\
    -   Sensores/variáveis contínuas (`temperatura_ar`,
        `temperatura_processo`, `umidade_relativa`,
        `velocidade_rotacional`, `torque`, `desgaste_da_ferramenta`)\
    -   Alvo principal (`falha_maquina`)\
    -   Falhas específicas (FDF, FDC, FP, FTE, FA)
3.  **Qualidade dos Dados**
    -   Valores faltantes\
    -   Valores incoerentes (temperaturas e velocidades negativas,
        desgaste negativo)\
    -   Inconsistência nos rótulos de falha
4.  **Estatísticas Descritivas**
    -   Média, desvio padrão, valores mínimo e máximo para sensores.
5.  **Distribuição e Padrões**
    -   Análise de **umidade relativa**, **temperatura do processo**,
        **torque** e **desgaste da ferramenta**.
6.  **Análise dos Rótulos das Classes**
    -   Padronização da variável alvo (`Sim` / `Não`).\
    -   Distribuição das falhas específicas.
7.  **Análise do Tipo de Máquina**
    -   Distribuição percentual dos tipos L, M e H.
8.  **Visualizações**
    -   Gráficos de distribuição (histogramas, boxplots, scatter plot).\
    -   Heatmap de correlação.\
    -   Distribuição das falhas específicas.
9.  **Resumo dos Problemas Identificados**
    -   Valores incoerentes\
    -   Valores faltantes\
    -   Inconsistência de rótulos\
    -   Desbalanceamento de classes
10. **Próximos Passos Recomendados**
    -   Limpeza e padronização dos dados\
    -   Tratamento de valores faltantes\
    -   Estratégias contra desbalanceamento (ex.: SMOTE,
        `class_weight`)\
    -   Feature engineering\
    -   Definição da estratégia de modelagem (multiclasse vs
        multirrótulo)

------------------------------------------------------------------------

## 📂 Estrutura

    ├── codigo_completo_EDA.py   # Script principal
    ├── bootcamp_train.csv       # Dataset de entrada (não incluído)
    └── README.md                # Documentação

------------------------------------------------------------------------

## 🛠️ Requisitos

O script utiliza as seguintes bibliotecas:

-   `pandas`
-   `numpy`
-   `matplotlib`
-   `seaborn`
-   `collections`

Instale as dependências com:

``` bash
pip install pandas numpy matplotlib seaborn
```

------------------------------------------------------------------------

## ▶️ Como Executar

1.  Coloque o arquivo `bootcamp_train.csv` no diretório definido no
    código ou ajuste o caminho no trecho:

``` python
df = pd.read_csv("D:/bootcamp_cdia/projeto_final/bootcamp_train.csv")
```

2.  Execute o script:

``` bash
python codigo_completo_EDA.py
```

3.  Os resultados serão exibidos no terminal e gráficos interativos
    serão gerados.

------------------------------------------------------------------------

## 📌 Observações Importantes

-   O dataset apresenta **forte desbalanceamento**: menos de 2% das
    amostras indicam falha.\
-   A variável `umidade_relativa` é praticamente constante (90).\
-   Há **valores incoerentes** (temperatura negativa, torque negativo).\
-   É necessário padronizar rótulos e tratar valores ausentes antes da
    modelagem.
