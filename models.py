import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load  # Substituindo custom_joblib por joblib
from sklearn.neighbors import KNeighborsClassifier


# Salvar o modelo treinado
def save_model(model, filename):
    dump(model, filename)


# Carregar o modelo treinado
def load_model(filename):
    return load(filename)


# Função para gerar gráficos com base nas colunas fornecidas
def generate_plots(data):
    plots = []
    sns.set(style="whitegrid", palette="muted")

    # Verifica se a pasta static/images existe; caso contrário, cria
    if not os.path.exists('static/images'):
        os.makedirs('static/images')

    # Gráfico de distribuição das idades (Histograma)
    if 'Age' in data.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data['Age'], kde=True, color='skyblue', bins=20)
        plt.title('Distribuição das Idades', fontsize=16)
        plt.xlabel('Idade', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plot_path = 'static/images/plot_age_distribution.png'
        plt.savefig(plot_path)
        plt.close()
        plots.append('images/plot_age_distribution.png')

    # Gráfico de contagem de Gênero (Barplot)
    if 'Gender' in data.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=data['Gender'], palette='pastel')
        plt.title('Distribuição de Gênero', fontsize=16)
        plt.xlabel('Gênero', fontsize=12)
        plt.ylabel('Quantidade', fontsize=12)
        plot_path = 'static/images/plot_gender_distribution.png'
        plt.savefig(plot_path)
        plt.close()
        plots.append('images/plot_gender_distribution.png')

    # Gráfico de contagem para Diagnosis (Barplot)
    if 'Diagnosis' in data.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=data['Diagnosis'], palette='viridis')
        plt.title('Distribuição de Diagnósticos', fontsize=16)
        plt.xlabel('Diagnóstico', fontsize=12)
        plt.ylabel('Quantidade', fontsize=12)
        plot_path = 'static/images/plot_diagnosis.png'
        plt.savefig(plot_path)
        plt.close()
        plots.append('images/plot_diagnosis.png')

    # Gráfico de Severidade dos Sintomas (Symptom Severity) vs Idade (Scatterplot)
    if 'Symptom Severity (1-10)' in data.columns and 'Age' in data.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data['Age'], y=data['Symptom Severity (1-10)'], color='orange')
        plt.title('Severidade dos Sintomas vs Idade', fontsize=16)
        plt.xlabel('Idade', fontsize=12)
        plt.ylabel('Severidade dos Sintomas (1-10)', fontsize=12)
        plot_path = 'static/images/plot_symptom_severity_vs_age.png'
        plt.savefig(plot_path)
        plt.close()
        plots.append('images/plot_symptom_severity_vs_age.png')

    # Gráfico de Qualidade do Sono (Histograma)
    if 'Sleep Quality (1-10)' in data.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data['Sleep Quality (1-10)'], kde=True, color='green', bins=10)
        plt.title('Distribuição da Qualidade do Sono', fontsize=16)
        plt.xlabel('Qualidade do Sono (1-10)', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plot_path = 'static/images/plot_sleep_quality.png'
        plt.savefig(plot_path)
        plt.close()
        plots.append('images/plot_sleep_quality.png')

    # Gráfico de Nível de Estresse (Stress Level) (Boxplot)
    if 'Stress Level (1-10)' in data.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=data['Stress Level (1-10)'], color='purple')
        plt.title('Distribuição do Nível de Estresse', fontsize=16)
        plt.xlabel('Nível de Estresse (1-10)', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plot_path = 'static/images/plot_stress_level.png'
        plt.savefig(plot_path)
        plt.close()
        plots.append('images/plot_stress_level.png')

    # Gráfico de Aderência ao Tratamento (Adherence to Treatment %) (Violinplot)
    if 'Adherence to Treatment (%)' in data.columns:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=data['Adherence to Treatment (%)'], color='brown')
        plt.title('Aderência ao Tratamento (%)', fontsize=16)
        plt.xlabel('Aderência ao Tratamento (%)', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plot_path = 'static/images/plot_adherence_to_treatment.png'
        plt.savefig(plot_path)
        plt.close()
        plots.append('images/plot_adherence_to_treatment.png')

    return plots

def train_model(data, model_type='DecisionTree', max_depth=5, n_estimators=100, n_neighbors=5, kernel='linear'):
    # Selecionar colunas numéricas
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_columns) < 2:
        raise ValueError("Não há colunas suficientes para treinar o modelo. O dataset precisa de ao menos duas colunas numéricas.")

    X = data[numeric_columns[:-1]]  # Features (exceto a última)
    y = data[numeric_columns[-1]]   # Target (última coluna)

    # Divisão dos dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Seleção do modelo
    if model_type == 'DecisionTree':
        model = DecisionTreeClassifier(max_depth=max_depth)
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    elif model_type == 'KNN':
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_type == 'SVM':
        model = SVC(kernel=kernel)
    else:
        raise ValueError("Modelo não suportado.")

    # Treinamento do modelo
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    # Gráfico de importância das características (apenas para DecisionTree e RandomForest)
    plot_path = None
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        feature_importances = model.feature_importances_
        sns.barplot(x=feature_importances, y=X.columns, palette='coolwarm')
        plt.title('Importância das Características', fontsize=16)
        plt.xlabel('Importância', fontsize=12)
        plt.ylabel('Características', fontsize=12)
        plot_path = 'static/images/feature_importance.png'
        plt.savefig(plot_path)
        plt.close()

    return accuracy, plot_path

def retrain_model(data, model_type='DecisionTree', max_depth=5, n_estimators=100, n_neighbors=5, kernel='linear', model_filename='model.pkl'):
    # Seleção das colunas numéricas
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_columns) < 2:
        raise ValueError("Não há colunas suficientes para treinar o modelo. O dataset precisa de ao menos duas colunas numéricas.")

    X = data[numeric_columns[:-1]]  # Features
    y = data[numeric_columns[-1]]   # Target

    # Divisão dos dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Seleção e treinamento do modelo
    if model_type == 'DecisionTree':
        model = DecisionTreeClassifier(max_depth=max_depth)
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    elif model_type == 'KNN':
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_type == 'SVM':
        model = SVC(kernel=kernel)
    else:
        raise ValueError("Modelo não suportado.")

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    # Salvar o modelo treinado
    save_model(model, model_filename)

    return accuracy
