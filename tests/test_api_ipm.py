import pytest
import requests

BASE_URL = 'http://localhost:5000'


# Função que cria o token
@pytest.fixture(scope='session')
def token():
    login_data = {'username': 'admin', 'password': 'admin'}
    response = requests.post(f'{BASE_URL}/login', json=login_data)
    assert response.status_code == 200, f"Erro no login: {response.text}"
    return response.json()['access_token']


# função que cria um novo usuário (pois até o momento, só existe o usuário admin)
def test_create_user(token):
    headers = {'Authorization': f'Bearer {token}'}
    user_data = {'nome': 'garotaatech', 'email': 'garotatech@outlook.com'}
    response = requests.post(f'{BASE_URL}/users', json=user_data, headers=headers)
    assert response.status_code == 201
    data = response.json()
    assert data['nome'] == 'garotaatech'
    assert data['email'] == 'garotatech@outlook.com'
    # Salva o ID para os próximos testes
    global user_id
    user_id = data['id']


def test_get_user(token):
    """
    Testa a recuperação de um usuário específico (GET /users/{id}).

    Pré-requisitos:
        - Um usuário com o ID contido na variável global 'user_id' deve existir previamente no banco de dados.
        - Um token JWT válido deve ser fornecido pela fixture 'token'.

    Procedimento do teste:
        1. Envia uma requisição GET para o endpoint /users/{user_id}, incluindo o token de autenticação no header.
        2. Verifica se o status da resposta é 200 OK.
        3. Valida se o JSON de resposta contém o campo 'id' com o valor esperado (user_id).

    Assertions:
        - O status HTTP deve ser 200.
        - O campo 'id' no corpo da resposta deve ser igual a 'user_id'.

    Parâmetros:
        token (str): Token JWT válido, fornecido automaticamente pela fixture do pytest.

    Retorno:
        None. O teste falhará se as condições esperadas não forem atendidas.
    """
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(f'{BASE_URL}/users/{user_id}', headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data['id'] == user_id


def test_update_user(token):
    headers = {'Authorization': f'Bearer {token}'}
    update_data = {'nome': 'garotatechupdate', 'email': 'garotatechupdate@gmail.com'}
    response = requests.put(f'{BASE_URL}/users/{user_id}', json=update_data, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data['nome'] == 'garotatechupdate'
    assert data['email'] == 'garotatechupdate@gmail.com'


def test_list_users(token):
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(f'{BASE_URL}/users', headers=headers)
    assert response.status_code == 200
    users = response.json()
    assert any(user['id'] == user_id for user in users)


def test_delete_user(token):
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.delete(f'{BASE_URL}/users/{user_id}', headers=headers)
    assert response.status_code == 204


def test_user_deleted(token):
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(f'{BASE_URL}/users/{user_id}', headers=headers)
    assert response.status_code == 404


def test_predict_route(token):
    """
    Testa a rota de predição (/predict) da API Flask.

    Pré-requisitos:
        - A API deve estar em execução e com o modelo 'modelo_regressao.pkl' carregado.
        - Um token JWT válido deve ser fornecido (gerado pela fixture 'token').

    Procedimento do teste:
        1. Envia uma requisição POST para o endpoint /predict com dados de entrada numéricos.
        2. Inclui o token JWT no header Authorization.
        3. Valida se a resposta tem status 200 OK.
        4. Verifica se o campo 'predicao' está presente no corpo da resposta.
        5. Confirma se o resultado da predição é uma lista de valores numéricos.

    Assertions:
        - O status HTTP deve ser 200.
        - O JSON de resposta deve conter a chave 'predicao'.
        - O resultado da predição deve ser uma lista numérica.

    Parâmetros:
        token (str): Token JWT válido, fornecido automaticamente pela fixture do pytest.

    Retorno:
        None. O teste falhará se qualquer condição esperada não for atendida.
    """
    headers = {'Authorization': f'Bearer {token}'}
    input_data = {"entrada": [5]}  # Exemplo de entrada para o modelo

    response = requests.post(f'{BASE_URL}/predict', json=input_data, headers=headers)

    # Verifica o status HTTP
    assert response.status_code == 200, f"Erro: {response.text}"

    # Verifica se o JSON tem o campo 'predicao'
    data = response.json()
    assert 'predicao' in data, "Resposta não contém o campo 'predicao'"

    # Verifica se o resultado é uma lista e contém número
    assert isinstance(data['predicao'], list), "Predição não é uma lista"
    assert all(isinstance(valor, (int, float)) for valor in data['predicao']), "Predição contém valores não numéricos"
