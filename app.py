from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename
from models import retrain_model, train_model, generate_plots

app = Flask(__name__)

# Definindo o diretório de uploads
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}  # Definir extensões permitidas

# Função para verificar se o arquivo tem a extensão permitida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Criar o diretório de uploads, caso não exista
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Rota inicial para upload do arquivo CSV
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('analyze', filename=filename))
        else:
            return "Arquivo inválido. Apenas arquivos CSV são permitidos."
    return render_template('upload.html')

# Rota para análise de dados e visualização de gráficos
@app.route('/analyze/<filename>')
def analyze(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)
    plots = generate_plots(data)  # Gera os gráficos com base nos dados
    return render_template('analyze.html', plots=plots, filename=filename)

@app.route('/retrain', methods=['POST'])
def retrain():
    if request.method == 'POST':
        filename = request.form.get('filename')  # Supondo que o nome do arquivo seja enviado no formulário

        if filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                # Carregar os novos dados do arquivo CSV
                data = pd.read_csv(filepath)
                # Re-treinar o modelo
                accuracy = retrain_model(data)
                return render_template('retrain_result.html', accuracy=accuracy)
            except Exception as e:
                return f"Erro ao processar o arquivo: {e}"
        else:
            return "Nenhum arquivo selecionado."

# Rota para predições com machine learning
@app.route('/predict/<filename>', methods=['GET', 'POST'])
def predict(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)
    
    if request.method == 'POST':
        model_type = request.form.get('model_type', 'DecisionTree')
        max_depth = int(request.form.get('max_depth', 5))
        
        accuracy, plot_url = train_model(data, model_type, max_depth)  # Treina o modelo e gera o gráfico de importância
        return render_template('predict.html', accuracy=accuracy, plot_url=plot_url, filename=filename)
    
    return render_template('predict.html', accuracy=None, plot_url=None, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
