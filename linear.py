import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        print("Fichier lu avec succès")
        print("Colonnes dans le fichier :", data.columns)
        print("Nombre de lignes :", len(data))
        if 'km' in data.columns and 'price' in data.columns:
            X = data['km'].values.reshape(-1, 1)
            y = data['price'].values.reshape(-1, 1)
            print("Données chargées avec succès.")
            return X, y
        else:
            print("Les colonnes 'km' ou 'price' n'existent pas dans le fichier")
            return None, None
    except FileNotFoundError:
        print("Le fichier '" + filepath + "' n'a pas été trouvé")
        return None, None
    except pd.errors.EmptyDataError:
        print("Le fichier '" + filepath + "' est vide")
        return None, None
    except Exception as e:
        print("Une erreur s'est produite :", str(e))
        return None, None

def plot_data(X, y):
    plt.scatter(X, y)
    plt.xlabel('Kilométrage (km)')
    plt.ylabel('Prix (€)')
    plt.title('Prix vs Kilométrage')
    plt.show()

def initialize_theta(n_features):
    return np.random.randn(n_features, 1)

def model(X, theta):
    return np.dot(X, theta)

def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y) ** 2)

def grad(X, y, theta):
    m = len(y)
    error = model(X, theta) - y
    return (1/m) * np.dot(X.T, error)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
    cost_history = []
    for i in range(n_iterations):
        gradient = grad(X, y, theta)
        theta -= learning_rate * gradient
        cost = cost_function(X, y, theta)
        cost_history.append(cost)
        if i % 10 == 0:
            print(f"Iteration {i}, Cost: {cost}, Theta: {theta.flatten()}")
    return theta, cost_history

def main():
    print("Répertoire de travail actuel :", os.getcwd())
    X, y = load_data('data.csv')
    if X is not None and y is not None:
        plot_data(X, y)
        
        # Normalize X
        X_normalized = (X - np.mean(X)) / np.std(X)
        X_b = np.hstack((X_normalized, np.ones((X.shape[0], 1))))  # add bias term
        
        theta = initialize_theta(2)
        theta, cost_history = gradient_descent(X_b, y, theta, 0.1, 200)
        
        y_pred = model(X_b, theta)
        plt.scatter(X, y, color='blue', label='Original data')
        plt.plot(X, y_pred, color='red', label='Fitted line')
        plt.xlabel('Kilométrage (km)')
        plt.ylabel('Prix (€)')
        plt.title('Fit with Gradient Descent')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
