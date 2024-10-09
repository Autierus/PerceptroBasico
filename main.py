import random
import matplotlib.pyplot as plt

# Definir a função de ativação (degrau)
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron com treinamento até classificação correta
def perceptron_training(data, labels, learning_rate=0.1, epochs=100):
    num_samples, num_features = len(data), len(data[0])
    
    # Inicializar pesos e viés
    weights = [random.uniform(-1, 1) for _ in range(num_features)]
    bias = random.uniform(-1, 1)
    
    errors = []  # Para rastrear o erro por época

    for epoch in range(epochs):
        total_error = 0
        for i in range(num_samples):
            # Calcular soma ponderada
            weighted_sum = sum([weights[j] * data[i][j] for j in range(num_features)]) + bias
            
            # Obter saída pela função de ativação
            output = step_function(weighted_sum)
            
            # Calcular erro (diferença entre saída prevista e saída real)
            error = labels[i] - output
            
            # Atualizar pesos e viés se houver erro
            if error != 0:
                for j in range(num_features):
                    weights[j] += learning_rate * error * data[i][j]
                bias += learning_rate * error
            
            total_error += abs(error)
        
        # Salvar o erro total por época
        errors.append(total_error)
        
        # Se não houver erros, o treinamento está completo
        if total_error == 0:
            print(f"Treinamento completo na época {epoch + 1}")
            break

    return weights, bias, errors

# Plotar o erro por época
def plot_errors(errors):
    plt.plot(errors)
    plt.title('Erro por época')
    plt.xlabel('Época')
    plt.ylabel('Erro total')
    plt.show()

# Exemplo de treinamento com a tabela verdade AND
data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
labels = [0, 0, 0, 1]  # Tabela verdade AND

# Treinar o Perceptron
weights, bias, errors = perceptron_training(data, labels)

# Exibir resultados
print("Pesos finais:", weights)
print("Viés final:", bias)

# Plotar erros
plot_errors(errors)
