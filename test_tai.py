import pytest
from fastapi.testclient import TestClient
from main import app, transformar_parametros, EAP, parar_teste, criterio_parada, maxima_informacao_th, proximo_item_criterio
import numpy as np

client = TestClient(app)

# Fixtures para reutilização de dados
@pytest.fixture
def sample_PAR():
    return np.array([[1.0, 250.0, 0.2], [2.0, 300.0, 0.3]])

@pytest.fixture
def sample_TAIRequest():
    return {
        "ESTUDANTE": "Aluno1",
        "AnoEscolarEstudante": 8,
        "proficiencia": 500.0,
        "profic_inic": 500.0,
        "idItem": ["ITEM1", "ITEM2"],
        "parA": [1.0, 2.0],
        "parB": [250.0, 300.0],
        "parC": [0.2, 0.3],
        "administrado": ["ITEM1"],
        "respostas": ["A"],
        "gabarito": ["A"],
        "erropadrao": 0.5,
        "n_Ij": 45,
        "componente": "LP",
        "idEixo": [1],
        "idHabilidade": [2]
    }

# Testes para transformar_parametros
def test_transformar_parametros_LP(sample_PAR):
    PAR = sample_PAR.copy()
    transformed = transformar_parametros(PAR, "LP")
    assert np.allclose(transformed[:, 0], [55.093, 110.186], rtol=1e-4)
    assert np.allclose(transformed[:, 1], [0.000272266894, 0.907828581], rtol=1e-4)

def test_transformar_parametros_MT(sample_PAR):
    PAR = sample_PAR.copy()
    transformed = transformar_parametros(PAR, "MT")
    assert np.allclose(transformed[:, 0], [55.892, 111.784], rtol=1e-4)
    assert np.allclose(transformed[:, 1], [0.000644099334, 0.895226508], rtol=1e-4)

# Testes para EAP
def test_EAP_basic():
    U = [1, 0]
    PAR = np.array([[1.5, 0.5, 0.2], [2.0, 1.0, 0.3]])
    administrado = [0, 1]
    theta, ep = EAP(U, PAR, administrado)
    assert isinstance(theta, float)
    assert isinstance(ep, float)

# Testes para parar_teste
def test_parar_teste_continuar():
    pontos_corte = [-1.0, 0.0, 1.0]
    assert parar_teste(0.5, 0.6, pontos_corte) == 0  # Intervalo cruzando pontos

def test_parar_teste_parar():
    pontos_corte = [-1.0, 0.0, 1.0]
    assert parar_teste(0.5, 0.3, pontos_corte) == 1  # Intervalo dentro de um segmento

# Testes para criterio_parada
def test_criterio_ep_atingido():
    assert criterio_parada(0.0, 0.4, parada="EP", EP=0.5, n_resp=10, n_min=8) == True

def test_criterio_max_itens():
    assert criterio_parada(0.0, 1.0, n_resp=43, n_Ij=45) == True

# Testes para maxima_informacao_th
def test_maxima_informacao_th():
    PAR = np.array([[1.0, 0.5, 0.2]])
    info = maxima_informacao_th(0.5, PAR)
    assert len(info) == 1
    assert info[0] > 0

# Testes para proximo_item_criterio
def test_proximo_item_criterio():
    INFO = np.array([0.1, 0.9, 0.5])
    administrado = [1]
    pos = proximo_item_criterio(INFO, administrado)
    assert pos == 2

# Testes para endpoints
def test_ping():
    response = client.post("/pingR")
    assert response.status_code == 200
    assert response.json() == {"status": "200"}

def test_proximo_item_primeiro_item(sample_TAIRequest):
    sample_TAIRequest["respostas"] = []
    response = client.post("/proximo", json=sample_TAIRequest)
    assert response.status_code == 200
    assert "proximo" in response.json()

def test_proximo_item_parada(sample_TAIRequest):
    # Configurar 45 itens e parâmetros para suportar n_Ij=45
    n_itens = 45
    sample_TAIRequest["idItem"] = [f"ITEM{i}" for i in range(1, n_itens + 1)]
    sample_TAIRequest["parA"] = [1.0] * n_itens
    sample_TAIRequest["parB"] = [250.0] * n_itens
    sample_TAIRequest["parC"] = [0.2] * n_itens

    # Simular 43 itens administrados (n_resp = n_Ij - 2)
    n_respostas = 43
    sample_TAIRequest["administrado"] = sample_TAIRequest["idItem"][:n_respostas]
    sample_TAIRequest["respostas"] = ["A"] * n_respostas
    sample_TAIRequest["gabarito"] = ["A"] * n_respostas
    sample_TAIRequest["n_Ij"] = 45

    response = client.post("/proximo", json=sample_TAIRequest)
    assert response.json()["proximo"] == -1  # Deve retornar -1 (parada)