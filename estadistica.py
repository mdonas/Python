import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Configuración
np.random.seed(42)  # Establece una semilla para el generador de números aleatorios para que cada vez que ejecutemos el código obtengamos exactamente los mismos resultados
n_simulaciones = 50000 # Haremos 50000 simulaciones

# Método Box-Muller
def box_muller(n):
    # Como tenemos que simular 50000 valores, tenemos 25000 pares
    # Generamos 25000 pares aleatorios entre 0 y 1
    u1 = np.random.uniform(0, 1, n//2)
    u2 = np.random.uniform(0, 1, n//2)
    
    """
    Usamos la formula basica del metood Box-Muller para transformar u1 y u2 en z0 y z1 
    Siendo estas variables aleatorias independientes con distribucción normal y desviciación típica 1
    """
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    
    # Unimos los dos arrays para obtener n valores normales
    return np.concatenate([z0, z1])

# Generar las simulaciones
z_simuladas = box_muller(n_simulaciones)

# Crear histograma
# Definimos una figura de 10 X 6 pulgadas
plt.figure(figsize=(10, 6))
"""
    bins=50; Dividimos el rango de datos en 50 intervalos
    density=True; Normalizamos el historiograma para que el area total sea 1
    alpha=0.7; Hacemos las barras semi transparentes
    color=blue & edgecolor=black; Establecemos el color de las barras en azul y el de los ejes en negro
"""
plt.hist(z_simuladas, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')

# Crea 1000 puntos entre -4 y 4 para graficar la curva esperada
x = np.linspace(-4, 4, 1000)
# Establecemos la formula de la densidad N(0,1)
plt.plot(x, 1/np.sqrt(2*np.pi) * np.exp(-x**2/2), 'r-', linewidth=2)
plt.title('Histograma de valores simulados vs. Densidad teórica N(0,1)')
plt.xlabel('Valores')
plt.ylabel('Densidad')
plt.grid(True)
plt.show()

# Calcular la probabilidad P(Z > 1.645)
"""
 Obtemos la probabilidad para 1.645 de nuestras simulaciones
 Convertimos z_simuladas en True o False si son mayores o no
 Al tener un array = [True, False, False, True, False, True] → [0, 0, 1, 0, 1] porque se tratan el True como 1 y False como 0
 Se cuentan los 1 y se divide por el total = % de 1 ,es decir, % que cumple la condición
"""
prob = np.mean(z_simuladas > 1.645) 
prob1=(norm.cdf(1.645))  # Obtenemos la probabilidad para 1.645
prob_teorica = 1 -prob1  # Valor teórico para P(Z > 1.645) = 1 - P(Z <= 1.645)
# print(f"Probabilidad simulada P(Z > 1.645): {prob:.5f}")
# print(f"Probabilidad teórica P(Z > 1.645): {prob_teorica:.5f}")