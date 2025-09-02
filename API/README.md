# API de Predição de Falhas em Máquinas

Este projeto implementa um modelo de Machine Learning usando Stacking Ensemble para predizer falhas em máquinas industriais, disponibilizado através de uma API REST usando FastAPI.

## Arquivos do Projeto

- `main.py` - Código principal da API FastAPI
- `stacking_model.pkl` - Modelo treinado (Stacking Ensemble)
- `scaler.pkl` - Normalizador das features
- `label_encoder.pkl` - Codificador para variável categórica 'tipo'
- `requirements.txt` - Dependências do projeto
- `test_api.py` - Script para testar a API
- `README.md` - Este arquivo

## Modelo

O modelo utiliza **Stacking Ensemble** com:
- **Modelos base**: Random Forest e Gradient Boosting
- **Meta-modelo**: Logistic Regression
- **Acurácia**: 99.30% no conjunto de teste

### Features utilizadas:
- temperatura_ar
- temperatura_processo
- umidade_relativa
- velocidade_rotacional
- torque
- desgaste_da_ferramenta
- tipo (L, M, H)
- Indicadores de falha (FDF, FDC, FP, FTE, FA)

## Instalação

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Execute a API:
```bash
python main.py
```

A API estará disponível em: `http://localhost:8000`

## Endpoints

### GET /
Endpoint raiz com informações básicas da API.

### GET /health
Verifica se a API está funcionando e os modelos estão carregados.

### POST /predict
Faz predição para uma única máquina.

**Exemplo de requisição:**
```json
{
    "temperatura_ar": 298.5,
    "temperatura_processo": 309.2,
    "umidade_relativa": 90.0,
    "velocidade_rotacional": 1500.0,
    "torque": 40.5,
    "desgaste_da_ferramenta": 120.0,
    "tipo": "L",
    "fdf_falha_desgaste_ferramenta": 0,
    "fdc_falha_dissipacao_calor": 0,
    "fp_falha_potencia": 0,
    "fte_falha_tensao_excessiva": 0,
    "fa_falha_aleatoria": 0
}
```

**Resposta:**
```json
{
    "falha_prevista": false,
    "probabilidade_falha": 0.05,
    "confianca": "Baixa"
}
```

### POST /predict_batch
Faz predições para múltiplas máquinas.

**Exemplo de requisição:**
```json
[
    {
        "temperatura_ar": 298.5,
        "temperatura_processo": 309.2,
        "umidade_relativa": 90.0,
        "velocidade_rotacional": 1500.0,
        "torque": 40.5,
        "desgaste_da_ferramenta": 120.0,
        "tipo": "L",
        "fdf_falha_desgaste_ferramenta": 0,
        "fdc_falha_dissipacao_calor": 0,
        "fp_falha_potencia": 0,
        "fte_falha_tensao_excessiva": 0,
        "fa_falha_aleatoria": 0
    }
]
```

## Testando a API

Execute o script de teste:
```bash
python test_api.py
```

## Documentação Interativa

Acesse a documentação Swagger em: `http://localhost:8000/docs`

## Parâmetros de Entrada

- **temperatura_ar**: Temperatura do ar (°C)
- **temperatura_processo**: Temperatura do processo (°C)
- **umidade_relativa**: Umidade relativa (%)
- **velocidade_rotacional**: Velocidade rotacional (RPM)
- **torque**: Torque (Nm)
- **desgaste_da_ferramenta**: Desgaste da ferramenta (μm)
- **tipo**: Tipo da máquina (L=Low, M=Medium, H=High)
- **fdf_falha_desgaste_ferramenta**: Indicador de falha por desgaste (0 ou 1)
- **fdc_falha_dissipacao_calor**: Indicador de falha por dissipação de calor (0 ou 1)
- **fp_falha_potencia**: Indicador de falha de potência (0 ou 1)
- **fte_falha_tensao_excessiva**: Indicador de falha por tensão excessiva (0 ou 1)
- **fa_falha_aleatoria**: Indicador de falha aleatória (0 ou 1)

## Níveis de Confiança

- **Alta**: Probabilidade > 80%
- **Média**: Probabilidade entre 50% e 80%
- **Baixa**: Probabilidade < 50%

## Exemplo de Uso com curl

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "temperatura_ar": 298.5,
       "temperatura_processo": 309.2,
       "umidade_relativa": 90.0,
       "velocidade_rotacional": 1500.0,
       "torque": 40.5,
       "desgaste_da_ferramenta": 120.0,
       "tipo": "L",
       "fdf_falha_desgaste_ferramenta": 0,
       "fdc_falha_dissipacao_calor": 0,
       "fp_falha_potencia": 0,
       "fte_falha_tensao_excessiva": 0,
       "fa_falha_aleatoria": 0
     }'
```

## Estrutura do Projeto

```
projeto/
├── main.py                    # API FastAPI
├── stacking_model.pkl         # Modelo treinado
├── scaler.pkl                 # Normalizador
├── label_encoder.pkl          # Codificador
├── requirements.txt           # Dependências
├── test_api.py               # Testes da API
└── README.md                 # Documentação
```

## Notas Técnicas

- O modelo foi treinado com balanceamento de classes devido ao desbalanceamento dos dados
- Valores especiais (-36, -38, -161, -202) são tratados como missing values
- Missing values são preenchidos com a mediana
- Features são normalizadas usando StandardScaler
- Cross-validation com 3 folds foi usado no Stacking
