import custom_joblib

# Função para salvar o modelo treinado
def save_model(model, filename):
    with open(filename, 'wb') as f:
        custom_joblib.dump(model, f)

# Função para carregar o modelo treinado
def load_model(filename):
    with open(filename, 'rb') as f:
        model = custom_joblib.load(f)
    return model
