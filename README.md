# 📘 SME Próximo API

API desenvolvida com [FastAPI](https://fastapi.tiangolo.com/) utilizada na Prova Saberes e Aprendizagens.

Escopo: Recebe respostas e parâmetros, estima proficiência, verifica critério de parada e retorna o próximo item.

## 🥞 Stack
- [FastApi 0.115.12](https://fastapi.tiangolo.com/)
- [Uvicorn 0.34.2](https://www.uvicorn.org/)
- [Httpx 0.28.1](https://www.python-httpx.org/)
- [Pytest 8.3.5](https://docs.pytest.org/en/stable/)
- [Pydantic 2.11.4](https://docs.pydantic.dev/latest/)


## 🧱 Estrutura do Projeto

```
prova-tai-api/
├── .gitignore                # Arquivos e pastas ignorados pelo Git
├── docker-compose.yaml       # Orquestração de containers com Docker
├── Dockerfile                # Imagem da aplicação
├── main.py                   # Ponto de entrada da aplicação FastAPI
├── test_tai.py               # Arquivo de testes com Pytest
├── requirements.txt          # Dependências do projeto
└── README.md                 # Documentação do projeto (este arquivo)
```

## 🚀 Como Executar

### Instalação local com ambiente virtual

```bash
# Crie o ambiente virtual
python -m venv venv

# Ative o ambiente virtual
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt

# Execute a aplicação
uvicorn main:app --reload
```

### Execução com Docker

```bash
docker compose up --build
```

Acesse a API no navegador:

```
http://localhost:8000
```

Documentação automática:
- Swagger UI: [`/docs`](http://localhost:8000/docs)
- Redoc: [`/redoc`](http://localhost:8000/redoc)

## 🧪 Executando os Testes
```bash
python -m pytest
```

