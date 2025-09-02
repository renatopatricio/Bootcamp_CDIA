from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import Optional
import uvicorn

# Carregar modelos e preprocessadores
try:
    model = joblib.load('stacking_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError as e:
    print(f"Erro ao carregar modelos: {e}")
    model = None
    scaler = None
    label_encoder = None

app = FastAPI(
    title="API de Predição de Falhas em Máquinas",
    description="API para predição de falhas em máquinas usando Stacking Ensemble",
    version="1.0.0"
)

class MachineData(BaseModel):
    temperatura_ar: float
    temperatura_processo: float
    umidade_relativa: float = 90.0
    velocidade_rotacional: float
    torque: float
    desgaste_da_ferramenta: float
    tipo: str  # L, M, ou H
    fdf_falha_desgaste_ferramenta: int = 0
    fdc_falha_dissipacao_calor: int = 0
    fp_falha_potencia: int = 0
    fte_falha_tensao_excessiva: int = 0
    fa_falha_aleatoria: int = 0

class PredictionResponse(BaseModel):
    falha_prevista: bool
    probabilidade_falha: float
    confianca: str

@app.get("/")
async def root():
    return {"message": "API de Predição de Falhas em Máquinas", "status": "ativo"}

@app.get("/health")
async def health_check():
    if model is None:
        return {"status": "erro", "message": "Modelos não carregados"}
    return {"status": "ok", "message": "API funcionando corretamente"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_failure(data: MachineData):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")

    try:
        # Preparar dados de entrada
        # Codificar tipo de máquina
        if data.tipo not in ['L', 'M', 'H']:
            raise HTTPException(status_code=400, detail="Tipo deve ser L, M ou H")

        tipo_encoded = label_encoder.transform([data.tipo])[0]

        # Criar array de features
        features = np.array([[
            data.temperatura_ar,
            data.temperatura_processo,
            data.umidade_relativa,
            data.velocidade_rotacional,
            data.torque,
            data.desgaste_da_ferramenta,
            tipo_encoded,
            data.fdf_falha_desgaste_ferramenta,
            data.fdc_falha_dissipacao_calor,
            data.fp_falha_potencia,
            data.fte_falha_tensao_excessiva,
            data.fa_falha_aleatoria
        ]])

        # Normalizar features
        features_scaled = scaler.transform(features)

        # Fazer predição
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        # Calcular confiança
        prob_falha = probability[1]
        if prob_falha > 0.8:
            confianca = "Alta"
        elif prob_falha > 0.5:
            confianca = "Média"
        else:
            confianca = "Baixa"

        return PredictionResponse(
            falha_prevista=bool(prediction),
            probabilidade_falha=float(prob_falha),
            confianca=confianca
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(data_list: list[MachineData]):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")

    try:
        results = []
        for data in data_list:
            # Preparar dados
            tipo_encoded = label_encoder.transform([data.tipo])[0]

            features = np.array([[
                data.temperatura_ar,
                data.temperatura_processo,
                data.umidade_relativa,
                data.velocidade_rotacional,
                data.torque,
                data.desgaste_da_ferramenta,
                tipo_encoded,
                data.fdf_falha_desgaste_ferramenta,
                data.fdc_falha_dissipacao_calor,
                data.fp_falha_potencia,
                data.fte_falha_tensao_excessiva,
                data.fa_falha_aleatoria
            ]])

            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            prob_falha = probability[1]

            if prob_falha > 0.8:
                confianca = "Alta"
            elif prob_falha > 0.5:
                confianca = "Média"
            else:
                confianca = "Baixa"

            results.append({
                "falha_prevista": bool(prediction),
                "probabilidade_falha": float(prob_falha),
                "confianca": confianca
            })

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição em lote: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
