import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

"""
PROGRAMA DE SIMULACIÓN ESTADÍSTICA
==================================
Este programa implementa el método Box-Muller para generar números aleatorios 
con distribución normal estándar N(0,1). El objetivo principal es:
1. Demostrar cómo transformar números aleatorios uniformes en normales
2. Visualizar la distribución de los valores generados comparándolos con la distribución teórica
3. Calcular probabilidades empíricas y compararlas con los valores teóricos

Autor: [Tu nombre]
Fecha: Abril 2025
Asignatura: Estadística - Ingeniería
"""
# Configuración
np.random.seed(42)  # Establece una semilla para el generador de números aleatorios para que cada vez que ejecutemos el código obtengamos exactamente los mismos resultados
n_simulaciones = 50000 # Haremos 50000 simulaciones

# Método Box-Muller
def box_muller(n):
    """
    Implementación del método Box-Muller para generar variables aleatorias con distribución normal estándar.
    
    El método Box-Muller es una técnica para generar pares de variables aleatorias independientes
    con distribución normal estándar, a partir de variables uniformes entre 0 y 1.
    
    Este algoritmo:
    1. Genera dos secuencias de números aleatorios uniformes (U1, U2)
    2. Los transforma mediante fórmulas trigonométricas específicas
    3. Produce dos secuencias de números con distribución normal estándar (Z0, Z1)
    
    Args:
        n (int): Número de valores normales a generar (debe ser par)
        
    Returns:
        numpy.ndarray: Array de n elementos con distribución normal estándar N(0,1)
    """
    # Como tenemos que simular 50000 valores, tenemos 25000 pares
    # Generamos 25000 pares aleatorios entre 0 y 1
    u1 = np.random.uniform(0, 1, n//2)
    u2 = np.random.uniform(0, 1, n//2)
    
    """
    Usamos la fórmula básica del método Box-Muller para transformar u1 y u2 en z0 y z1.
    La transformación se basa en las siguientes ecuaciones:
    
    Z0 = sqrt(-2*ln(U1)) * cos(2π*U2)
    Z1 = sqrt(-2*ln(U1)) * sin(2π*U2)
    
    Siendo Z0 y Z1 variables aleatorias independientes con distribución 
    normal estándar (media 0 y desviación típica 1)
    """
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    
    # Unimos los dos arrays para obtener n valores normales
    return np.concatenate([z0, z1])

# Generar las simulaciones
# Aplicamos la función box_muller para obtener n_simulaciones valores normales
z_simuladas = box_muller(n_simulaciones)

# Los valores generados deberían seguir aproximadamente una distribución normal estándar N(0,1)
# con media 0 y desviación típica 1
# VISUALIZACIÓN DE RESULTADOS
# ============================

# Crear histograma para visualizar la distribución de los valores simulados
# Definimos una figura de 10 X 6 pulgadas para mejor visualización
plt.figure(figsize=(10, 6))
"""
Configuración del histograma:
    bins=50: Dividimos el rango de datos en 50 intervalos para una visualización detallada
    density=True: Normalizamos el histograma para que el área total sea 1, permitiendo
                 la comparación directa con la función de densidad teórica
    alpha=0.7: Hacemos las barras semi-transparentes para poder ver la curva teórica
    color='blue' & edgecolor='black': Establecemos el color de las barras en azul y 
                                     el borde en negro para mejor contraste visual
"""
plt.hist(z_simuladas, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')

# Creamos 1000 puntos equiespaciados entre -4 y 4 para graficar la curva de densidad teórica
x = np.linspace(-4, 4, 1000)

# Establecemos la fórmula de la función de densidad de probabilidad (PDF) de la N(0,1)
# La fórmula es: f(x) = (1/√(2π)) * e^(-x²/2)
plt.plot(x, 1/np.sqrt(2*np.pi) * np.exp(-x**2/2), 'r-', linewidth=2, 
         label='Densidad teórica N(0,1)')
plt.title('Histograma de valores simulados vs. Densidad teórica N(0,1)', fontsize=14)
plt.xlabel('Valores de Z', fontsize=12)
plt.ylabel('Densidad de probabilidad', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)  # Agregamos una cuadrícula punteada para facilitar la lectura
plt.legend()  # Mostramos la leyenda para identificar la curva teórica
# plt.savefig('distribucion_normal_simulada.png', dpi=300)  # Opcional: guardar la imagen
# plt.show()  # Descomentar para mostrar el gráfico durante la ejecución

# CÁLCULO DE PROBABILIDADES
# =========================

# Calcular la probabilidad P(Z > 1.645) usando nuestras simulaciones
"""
Cálculo de probabilidad empírica:
1. Obtenemos la probabilidad para P(Z > 1.645) de nuestras simulaciones
2. La expresión 'z_simuladas > 1.645' crea un array de valores booleanos (True/False)
   donde True significa que el valor es mayor que 1.645
3. np.mean() convierte este array a valores numéricos (True=1, False=0) y calcula el promedio
4. Este promedio representa la proporción de valores que cumplen la condición,
   es decir, la probabilidad empírica P(Z > 1.645)

Nota: El valor 1.645 es importante en estadística ya que P(Z > 1.645) ≈ 0.05,
      y es comúnmente usado en pruebas de hipótesis con nivel de significancia del 5%.
"""
prob = np.mean(z_simuladas > 1.645) 
prob1 = norm.cdf(1.645)  # Obtenemos la probabilidad acumulada P(Z ≤ 1.645) usando scipy.stats
prob_teorica = 1 - prob1  # Valor teórico para P(Z > 1.645) = 1 - P(Z ≤ 1.645)

# Mostramos los resultados con formato de 5 decimales para comparar precisión
print(f"Probabilidad acumulada teórica P(Z ≤ 1.645): {prob1:.5f}")
print(f"Probabilidad simulada P(Z > 1.645): {prob:.5f}")
print(f"Probabilidad teórica P(Z > 1.645): {prob_teorica:.5f}")

# Calculamos y mostramos el error absoluto entre el valor teórico y el simulado
error_absoluto = abs(prob - prob_teorica)
print(f"Error absoluto entre probabilidad teórica y simulada: {error_absoluto:.7f}")

# Evaluación de la calidad de la simulación
if error_absoluto < 0.001:
    print("La simulación aproxima muy bien la probabilidad teórica.")
else:
    print("La simulación puede requerir más iteraciones para una mejor aproximación.")
