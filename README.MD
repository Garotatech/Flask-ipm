# IPM-IA: API Flask com Jupyter Notebook e SQLite

Este projeto é uma API REST desenvolvida com Flask, utilizando banco de dados SQLite e executada via Jupyter Notebook.  

O objetivo faz parte de um desafio técnico, onde a obrigatoriedade é manter a aplicação em formato `.ipynb` (notebook Jupyter).

---

## 🚀 Tecnologias utilizadas

- Python 3.11
- Flask
- Flask-SQLAlchemy
- Flask-JWT-Extended
- Jupyter Notebook
- SQLite
- Docker & Docker Compose

---

## 📦 Estrutura de Pastas
├── .venv/ # Ambiente virtual (local)
├── tests/ # Pasta de testes
├── main.ipynb # Notebook principal da aplicação
├── requirements.txt # Dependências da aplicação
├── Dockerfile # Dockerfile para build da aplicação
├── docker-compose.yml # Orquestração com Docker Compose
└── README.md # Este arquivo


---

## ✅ Instruções para rodar o projeto via Docker

### Pré-requisitos:

- Docker
- Docker Compose

---

### Passo 1 - Build da Imagem Docker:

```bash
docker-compose build
```


### Passo 2 - Subir os containers:

```bash
docker-compose up
```

### Passo 3 - Acessar o Jupyter Notebook:

Abra no navegador:
http://localhost:8888


### Passo 3 - Rodar a aplicação Flask

Dentro do Jupyter:

1. Abra o arquivo `main.ipynb`.

2. No menu superior, clique em `Kernel → Restart & Run All`.

 Isso executará todas as células do notebook, incluindo o servidor Flask, que irá rodar na porta `5000`.
 

---

## 📡 Endpoints disponíveis - Postman

### 1. Login (Gerar Token JWT)

- **Método:** `POST`
- **Endpoint:** `http://localhost:5000/login`

**Body (JSON):**
```json
{
  "username": "admin",
  "password": "admin"
}
``` 
Resposta esperada:
```json
{
  "access_token": "<seu_token_jwt>"
}
```


### 2. Criar usuário 

- **Método:** `POST`
- **Endpoint:** `http://localhost:5000/users`

**Body (JSON):**

**Exemplo**
```json
{
  "username": "garotaatech",
  "password": "garotatech@outlook.com"
}
```

Resposta esperada:
```json
{
    "created_at": "2025-06-22T23:10:35.063394",
    "email": "garotatech@outlook.com",
    "id": 1,
    "nome": "garotaatech"
}
```

### 3. Listar Usuários 

- **Método:** `GET`
- **Endpoint:** `http://localhost:5000/users`

Resposta esperada:
```json
[
    {
        "created_at": "2025-06-22T23:10:35.063394",
        "email": "garotatech@outlook.com",
        "id": 1,
        "nome": "garotaatech"
    }
]
```

### 4. Atualizar Usuários

- **Método:** `PUT`
- **Endpoint:** `http://localhost:5000/users/1` *(exemplo com id=1)*


**Body (JSON):**

**Exemplo**
```json
{
  "nome": "new name",
  "email": "novoemail@exemplo.com"
}
``` 

Resposta esperada:
```json
{
    "created_at": "2025-06-22T23:10:35.063394",
    "email": "novoemail@exemplo.com",
    "id": 1,
    "nome": "new name"
}
```


### 6. Deletar Usuário

- **Método:** `DELETE`  
- **Endpoint:** `http://localhost:5000/users/1`  *(exemplo com id=1)*