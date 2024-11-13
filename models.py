import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from custom_joblib import save_model


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

# Função para treinar o modelo e retornar a precisão e o gráfico de importância das características
def train_model(data, model_type='DecisionTree', max_depth=5):
    if 'Age' in data.columns and 'Symptom Severity (1-10)' in data.columns and 'Diagnosis' in data.columns:
        X = data[['Age', 'Symptom Severity (1-10)']]  # Exemplo de seleção de features
        y = data['Diagnosis']  # Target

        # Divisão dos dados em treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Treinamento do modelo com base no tipo selecionado
        if model_type == 'DecisionTree':
            model = DecisionTreeClassifier(max_depth=max_depth)
        elif model_type == 'RandomForest':
            model = RandomForestClassifier(max_depth=max_depth, n_estimators=100)
        else:
            raise ValueError("Modelo não suportado: selecione um modelo válido.")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Gráfico de importância das características (Barplot)
        plt.figure(figsize=(10, 6))
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            sns.barplot(x=feature_importances, y=X.columns, palette='coolwarm')
            plt.title('Importância das Características', fontsize=16)
            plt.xlabel('Importância', fontsize=12)
            plt.ylabel('Características', fontsize=12)
            plot_path = 'static/images/feature_importance.png'  # Salvando o gráfico na pasta correta
            plt.savefig(plot_path)
            plt.close()  # Fecha o gráfico atual para liberar memória
        else:
            plot_path = None  # Caso o modelo não tenha 'feature_importances'

        return accuracy, 'images/feature_importance.png' if plot_path else None
    else:
        raise ValueError("As colunas necessárias não foram encontradas no CSV.")
    

# Função para treinar/re-treinar o modelo
def retrain_model(data, model_type='DecisionTree', max_depth=5, model_filename='model.pkl'):
    # Verifica se as colunas necessárias estão presentes no DataFrame
    if 'Age' in data.columns and 'Symptom Severity (1-10)' in data.columns and 'Diagnosis' in data.columns:
        X = data[['Age', 'Symptom Severity (1-10)']]  # Exemplo de seleção de features
        y = data['Diagnosis']  # Target

        # Divisão dos dados em treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Seleção do modelo a ser utilizado
        if model_type == 'DecisionTree':
            model = DecisionTreeClassifier(max_depth=max_depth)
        elif model_type == 'RandomForest':
            model = RandomForestClassifier(max_depth=max_depth, n_estimators=100)
        else:
            raise ValueError("Modelo não suportado: selecione um modelo válido.")

        # Treinamento do modelo
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Salva o modelo treinado
        save_model(model, model_filename)
        
        # Retorna a acurácia do modelo treinado
        return accuracy
    else:
        raise ValueError("As colunas necessárias não foram encontradas no CSV.")
