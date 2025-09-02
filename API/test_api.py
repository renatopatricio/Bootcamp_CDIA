import requests
import json

# URL base da API (ajuste conforme necessário)
BASE_URL = "http://localhost:8000"

def test_health():
    """Testa o endpoint de health check"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())

def test_single_prediction():
    """Testa predição única"""
    # Dados de exemplo
    data = {
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

    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("Predição única:", response.json())

def test_batch_prediction():
    """Testa predição em lote"""
    data_list = [
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
        },
        {
            "temperatura_ar": 304.0,
            "temperatura_processo": 313.0,
            "umidade_relativa": 90.0,
            "velocidade_rotacional": 1200.0,
            "torque": 70.0,
            "desgaste_da_ferramenta": 250.0,
            "tipo": "H",
            "fdf_falha_desgaste_ferramenta": 1,
            "fdc_falha_dissipacao_calor": 1,
            "fp_falha_potencia": 0,
            "fte_falha_tensao_excessiva": 1,
            "fa_falha_aleatoria": 0
        }
    ]

    response = requests.post(f"{BASE_URL}/predict_batch", json=data_list)
    print("Predição em lote:", response.json())

if __name__ == "__main__":
    print("Testando API de Predição de Falhas...")
    print("=" * 50)

    try:
        test_health()
        print()
        test_single_prediction()
        print()
        test_batch_prediction()
    except requests.exceptions.ConnectionError:
        print("Erro: Não foi possível conectar à API. Certifique-se de que ela está rodando.")
    except Exception as e:
        print(f"Erro durante os testes: {e}")
