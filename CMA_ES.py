import math
import numpy as np
from matplotlib import pyplot
from cmaes import CMA

A = 10
n = 100
lower_bound = -5.12
upper_bound = 5.12

population_size = 500
sigma = 3
start_point = np.random.rand(n) * (upper_bound - lower_bound) + lower_bound

def Rastrigin(x):
    return A*n + sum([xi**2 - A*math.cos(2*math.pi*xi) for xi in x])

def grad(x):
    return [2*xi + 2*A*math.pi*math.sin(2*math.pi*xi) for xi in x]
    
def norm(x):
    return math.sqrt(sum([xi**2 for xi in x]))
    
def main():
    bounds = []
    for i in range(n):
        bounds.append([lower_bound, upper_bound])
    bounds = np.array(bounds)
    
    print(f"Начальное среднее: {start_point}")
    optimizer = CMA(mean=start_point, sigma=sigma, bounds=bounds, population_size=population_size)
    
    min_values = []
    max_values = []
    mean_values = []
    
    x = []
    while not optimizer.should_stop():
        x_s = []
        values = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            x_s.append(x)
            value = Rastrigin(x)
            values.append(value)
        min_values.append(min(values))
        max_values.append(max(values))
        mean = sum(values)/optimizer.population_size
        print(f"Поколение: {optimizer.generation}, среднее значение функции: {mean}")
        mean_values.append(mean)
        optimizer.tell(list(zip(x_s, values)))
    print(f"Итоговая точка: {x}")
    print(f"Норма градиента в итоговой точке: {norm(grad(x)):0.5f}")
    pyplot.plot(min_values, "g", max_values, "r", mean_values, "y")
    pyplot.show()
        
if __name__ == "__main__":
    main()