from pydantic import BaseModel 
from fastapi import FastAPI
import uvicorn 
import joblib

# Criar uma instância do FastAPI
app = FastAPI()

# Criar uma classe que terá os dados do request body para a API 
class request_body(BaseModel):
  horas_estudos: float

# Carregar modelo para realizar a predição 
modelo_pontuacao = joblib.load('./modelo_regressao.pkl')

def predict(data: request_body):
  # Preparar os dados para predição
  input_feature = [[data.horas_estudos]]

  # Realizar a predição
  y_pred = modelo_pontuacao.predict(input_feature)[0].astype(int)

  return {'pontuacao_teste': y_pred.toList()}

# uvicorn api_modelo_regressao:app --reload
