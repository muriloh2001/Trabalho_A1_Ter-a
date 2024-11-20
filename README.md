# Trabalho_A1_Ter-a

# Análise e Predição de Modelos de Machine Learning

Este projeto tem como objetivo treinar e re-treinar modelos de **machine learning** para análise de dados, utilizando algoritmos como **Árvore de Decisão**, **Floresta Aleatória**, **K-Nearest Neighbors (KNN)** e **Support Vector Machine (SVM)**. A aplicação oferece uma interface para o upload de dados, seleção de modelo, treinamento e visualização de resultados.

## Índice

1. [Descrição](#descrição)
2. [Funcionalidades](#funcionalidades)
3. [Tecnologias Usadas](#tecnologias-usadas)
4. [Como Rodar o Projeto](#como-rodar-o-projeto)
5. [Estrutura do Projeto](#estrutura-do-projeto)
6. [Contribuição](#contribuição)
7. [Licença](#licença)

## Descrição

Este projeto permite que os usuários façam upload de conjuntos de dados e treinem modelos de machine learning, selecionando entre **Árvore de Decisão**, **Floresta Aleatória**, **KNN**, e **SVM**. O sistema também possibilita o **re-treinamento** do modelo com novos dados, a visualização de gráficos de **importância das características** e o cálculo da **acurácia**.

Além disso, o sistema é desenvolvido com **Flask**, permitindo que os usuários interajam com a aplicação por meio de uma interface web simples.

## Funcionalidades

- **Upload de Dados**: O usuário pode carregar arquivos CSV para treinar o modelo.
- **Escolha do Modelo**: O usuário pode escolher entre quatro modelos de machine learning: **Árvore de Decisão**, **Floresta Aleatória**, **KNN**, e **SVM**.
- **Parâmetros Ajustáveis**: É possível ajustar parâmetros como a profundidade da árvore, número de estimadores e número de vizinhos.
- **Re-treinamento de Modelo**: O sistema permite re-treinar o modelo com novos dados sem a necessidade de enviar um novo arquivo.
- **Visualização da Acurácia**: Exibição da acurácia do modelo treinado com 2 casas decimais.
- **Gráfico de Importância das Características**: Exibe a importância das características para modelos como **Árvore de Decisão** e **Floresta Aleatória**.

## Tecnologias Usadas

Este projeto é construído com as seguintes tecnologias:

- **Python**: Linguagem de programação principal.
- **Flask**: Framework web para criar a interface de usuário e gerenciar as requisições HTTP.
- **Scikit-learn**: Biblioteca para machine learning, usada para treinar e avaliar os modelos de machine learning (DecisionTree, RandomForest, KNN, SVM).
- **Matplotlib** e **Seaborn**: Bibliotecas para visualização de dados, usadas para gerar gráficos de importância das características.
- **Pandas**: Biblioteca para manipulação de dados, usada para carregar e preparar os dados para treinamento.
- **Joblib**: Usada para salvar e carregar os modelos treinados.

## Como Rodar o Projeto

### Pré-requisitos

Antes de rodar o projeto, é necessário ter o **Python 3** instalado. Você pode usar um ambiente virtual para isolar as dependências.

1. **Instalar dependências**:
   O projeto usa o **Python 3**. Para instalar as dependências, basta usar o `pip`:

   ```bash
   pip install -r requirements.txt

O requirements.txt contem as bibliotecas necessárias, como:

flask
scikit-learn
matplotlib
seaborn
pandas
joblib

Rodando o Servidor Flask: Após instalar as dependências, execute o seguinte comando para rodar o servidor Flask `pip`: 
   ```bash
   pip install -r requirements.txt
````
Estrutura do Projeto
A estrutura do projeto segue o padrão recomendado para aplicativos Flask:
```bash
project/
│
├── app.py                 # Arquivo principal do Flask (ponto de entrada)
├── models.py              # Código relacionado ao treinamento e re-treinamento de modelos
├── custom_joblib.py       # (Opcional) Código para salvar e carregar modelos (se usado)
├── templates/             # Templates HTML (Flask)
│   ├── index.html
│   ├── predict.html
│   └── result.html
├── static/                # Arquivos estáticos (CSS, JS, imagens)
│   └── images/
├── requirements.txt       # Dependências do Python
└── README.md              # Documentação do projeto
````
Arquivos principais:
app.py: Contém as rotas e lógica principal do Flask. A aplicação lida com o upload de arquivos, treinamento de modelos e visualização dos resultados.
models.py: Contém a lógica para treinar os modelos de machine learning (Árvore de Decisão, Floresta Aleatória, KNN, SVM).
templates/: Diretório contendo os templates HTML para as páginas do aplicativo.
static/images/: Diretório onde são salvos os gráficos gerados pela aplicação (ex: gráficos de importância das características).





