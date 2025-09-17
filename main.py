import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
import time

# --- 1. ИСХОДНЫЕ ДАННЫЕ И ПАРАМЕТРЫ АЛГОРИТМА ---
N = 5  # Количество пунктов производства
K = 8  # Количество городов
Y_max = 100  # Максимальное производство на пункт
X_need = 60  # Необходимый объем потребления на город
price_per_unit_distance = 10  # Стоимость доставки за ед. продукта на ед. расстояния
penalty_per_excess_unit = 50  # Штраф за ед. превышения
penalty_per_shortage_unit = 100  # Штраф за недостаток продуктов (в два раза больше, чем за превышение)

POPULATION_SIZE = 100  # Размер популяции
GENERATIONS = 200  # Количество поколений
MUTATION_RATE = 0.05  # Вероятность мутации

# Генерация случайных координат и объемов
np.random.seed(42)
production_points = [
    {'id': i, 'x': random.randint(0, 100), 'y': random.randint(0, 100), 'supply': random.randint(Y_max - 20, Y_max)} for
    i in range(N)]
cities = [{'id': i, 'x': random.randint(0, 100), 'y': random.randint(0, 100),
           'demand': random.randint(X_need - 10, X_need + 10)} for i in range(K)]

# Вычисляем матрицу расстояний (евклидово расстояние)
distance_matrix = np.zeros((N, K))
for i in range(N):
    for j in range(K):
        dist = np.sqrt(
            (production_points[i]['x'] - cities[j]['x']) ** 2 + (production_points[i]['y'] - cities[j]['y']) ** 2)
        distance_matrix[i][j] = dist


# --- 2. ФУНКЦИЯ ПРИСПОСОБЛЕННОСТИ ---
def calculate_fitness(individual):
    total_cost = 0
    excess_penalty = 0
    shortage_penalty = 0

    if any(individual[i, :].sum() > production_points[i]['supply'] for i in range(N)):
        return 0

    for i in range(N):
        for j in range(K):
            total_cost += individual[i, j] * distance_matrix[i, j] * price_per_unit_distance

    for j in range(K):
        total_supply_to_city = individual[:, j].sum()
        if total_supply_to_city > cities[j]['demand']:
            excess = total_supply_to_city - cities[j]['demand']
            excess_penalty += excess * penalty_per_excess_unit
        elif total_supply_to_city < cities[j]['demand']:
            shortage = cities[j]['demand'] - total_supply_to_city
            shortage_penalty += shortage * penalty_per_shortage_unit

    total_penalty = total_cost + excess_penalty + shortage_penalty
    return 1 / (1 + total_penalty)


# --- 3. ОПЕРАТОРЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА ---

def create_initial_population(size):
    population = []
    for _ in range(size):
        individual = np.zeros((N, K))
        for i in range(N):
            supply_left = production_points[i]['supply']

            flows = np.random.randint(0, supply_left + 1, K)

            if flows.sum() > supply_left:
                flows = flows * supply_left / flows.sum()

            individual[i, :] = flows.astype(int)

        population.append(individual)
    return population


def selection(population, num_parents):
    selected = []
    for _ in range(num_parents):
        tournament_pool = random.sample(population, 3)
        winner = max(tournament_pool, key=calculate_fitness)
        selected.append(winner)
    return selected


def single_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, N - 1)
    child1 = np.vstack((parent1[:crossover_point, :], parent2[crossover_point:, :]))
    child2 = np.vstack((parent2[:crossover_point, :], parent1[crossover_point:, :]))
    return child1, child2


def two_point_crossover(parent1, parent2):
    p1, p2 = sorted(random.sample(range(1, N - 1), 2))
    child1 = np.vstack((parent1[:p1, :], parent2[p1:p2, :], parent1[p2:, :]))
    child2 = np.vstack((parent2[:p1, :], parent1[p1:p2, :], parent2[p2:, :]))
    return child1, child2


def uniform_crossover(parent1, parent2):
    child1 = np.zeros_like(parent1)
    child2 = np.zeros_like(parent2)
    for i in range(N):
        for j in range(K):
            if random.random() < 0.5:
                child1[i, j] = parent1[i, j]
                child2[i, j] = parent2[i, j]
            else:
                child1[i, j] = parent2[i, j]
                child2[i, j] = parent1[i, j]
    return child1, child2


def single_point_mutation(individual):
    if random.random() < MUTATION_RATE:
        row = random.randint(0, N - 1)
        col = random.randint(0, K - 1)
        change = random.randint(-20, 20)
        individual[row, col] = max(0, individual[row, col] + change)
    return individual


def reset_mutation(individual):
    if random.random() < MUTATION_RATE:
        row = random.randint(0, N - 1)
        col = random.randint(0, K - 1)
        individual[row, col] = random.randint(0, production_points[row]['supply'])
    return individual


def redistribution_mutation(individual):
    if random.random() < MUTATION_RATE:
        row = random.randint(0, N - 1)
        supply_left = production_points[row]['supply']

        individual[row, :] = 0  # Сбрасываем все потоки из этого пункта

        # Распределяем их заново случайным образом
        flows = np.random.randint(0, supply_left + 1, K)
        if flows.sum() > supply_left:
            flows = flows * supply_left / flows.sum()
        individual[row, :] = flows.astype(int)
    return individual


# --- 4. ОСНОВНОЙ ЦИКЛ ГЕНЕТИЧЕСКОГО АЛГОРИТМА ---
def genetic_algorithm(crossover_type='single_point', mutation_type='single_point'):
    population = create_initial_population(POPULATION_SIZE)
    best_fitness_history = []

    for generation in range(GENERATIONS):
        fitness_scores = [calculate_fitness(ind) for ind in population]
        best_fitness_history.append(max(fitness_scores))

        parents = selection(population, POPULATION_SIZE // 2)

        next_generation = []
        while len(next_generation) < POPULATION_SIZE:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            # Выбор оператора скрещивания
            if crossover_type == 'single_point':
                child1, child2 = single_point_crossover(parent1, parent2)
            elif crossover_type == 'two_point':
                child1, child2 = two_point_crossover(parent1, parent2)
            elif crossover_type == 'uniform':
                child1, child2 = uniform_crossover(parent1, parent2)

            # Выбор оператора мутации
            if mutation_type == 'single_point':
                next_generation.append(single_point_mutation(child1))
                next_generation.append(single_point_mutation(child2))
            elif mutation_type == 'reset':
                next_generation.append(reset_mutation(child1))
                next_generation.append(reset_mutation(child2))
            elif mutation_type == 'redistribution':
                next_generation.append(redistribution_mutation(child1))
                next_generation.append(redistribution_mutation(child2))

        population = next_generation

    best_individual = max(population, key=calculate_fitness)
    best_fitness = calculate_fitness(best_individual)

    return best_individual, best_fitness, best_fitness_history


# --- 5. ФУНКЦИЯ ПОЛНОГО ПЕРЕБОРА (УПРОЩЕННАЯ) ---
def brute_force_solver_simplified():
    print("--- Запуск полного перебора для упрощенной задачи (2x2) ---")

    N_brute = 2
    K_brute = 2
    Y_max_brute = 10

    brute_production_points = [{'supply': 10}, {'supply': 10}]
    brute_cities = [{'demand': 10}, {'demand': 10}]
    brute_distance_matrix = np.array([[10, 20], [30, 15]])
    brute_price_per_unit_distance = 1
    brute_penalty_per_excess_unit = 5
    brute_penalty_per_shortage_unit = 10

    def calculate_penalty_brute(individual):
        total_cost = 0
        excess_penalty = 0
        shortage_penalty = 0

        if any(individual[i, :].sum() > brute_production_points[i]['supply'] for i in range(N_brute)):
            return float('inf')

        for i in range(N_brute):
            for j in range(K_brute):
                total_cost += individual[i, j] * brute_distance_matrix[i, j] * brute_price_per_unit_distance

        for j in range(K_brute):
            total_supply_to_city = individual[:, j].sum()
            if total_supply_to_city > brute_cities[j]['demand']:
                excess = total_supply_to_city - brute_cities[j]['demand']
                excess_penalty += excess * brute_penalty_per_excess_unit
            elif total_supply_to_city < brute_cities[j]['demand']:
                shortage = brute_cities[j]['demand'] - total_supply_to_city
                shortage_penalty += shortage * brute_penalty_per_shortage_unit

        return total_cost + excess_penalty + shortage_penalty

    min_penalty = float('inf')
    best_solution = None

    start_time = time.time()

    all_flows = range(Y_max_brute + 1)

    for combo in itertools.product(all_flows, repeat=N_brute * K_brute):
        individual = np.array(combo).reshape(N_brute, K_brute)
        penalty = calculate_penalty_brute(individual)

        if penalty < min_penalty:
            min_penalty = penalty
            best_solution = individual.copy()

    end_time = time.time()

    return best_solution, min_penalty, end_time - start_time


# --- 6. ЗАПУСК АЛГОРИТМОВ И ВЫВОД РЕЗУЛЬТАТОВ ---
if __name__ == "__main__":
    # 1. Запуск с мутацией "single_point"
    print("--- Запуск генетического алгоритма (Одноточечная мутация) ---")
    ga_start_time_sp = time.time()
    best_solution_sp, best_fit_sp, history_sp = genetic_algorithm(mutation_type='single_point')
    ga_end_time_sp = time.time()
    print(f"Время выполнения: {ga_end_time_sp - ga_start_time_sp:.4f} сек.")
    print("Лучшее найденное решение (матрица потоков):")
    print(best_solution_sp)
    print("\nЗначение функции приспособленности (Fitness):", best_fit_sp)
    print("Общая стоимость (с учетом штрафов):", 1 / best_fit_sp - 1)
    print("-" * 40)

    # 2. Запуск с мутацией "reset"
    print("--- Запуск генетического алгоритма (Мутация сброса) ---")
    ga_start_time_re = time.time()
    best_solution_re, best_fit_re, history_re = genetic_algorithm(mutation_type='reset')
    ga_end_time_re = time.time()
    print(f"Время выполнения: {ga_end_time_re - ga_start_time_re:.4f} сек.")
    print("Лучшее найденное решение (матрица потоков):")
    print(best_solution_re)
    print("\nЗначение функции приспособленности (Fitness):", best_fit_re)
    print("Общая стоимость (с учетом штрафов):", 1 / best_fit_re - 1)
    print("-" * 40)

    # 3. Запуск с мутацией "redistribution"
    print("--- Запуск генетического алгоритма (Мутация перераспределения) ---")
    ga_start_time_rd = time.time()
    best_solution_rd, best_fit_rd, history_rd = genetic_algorithm(mutation_type='redistribution')
    ga_end_time_rd = time.time()
    print(f"Время выполнения: {ga_end_time_rd - ga_start_time_rd:.4f} сек.")
    print("Лучшее найденное решение (матрица потоков):")
    print(best_solution_rd)
    print("\nЗначение функции приспособленности (Fitness):", best_fit_rd)
    print("Общая стоимость (с учетом штрафов):", 1 / best_fit_rd - 1)
    print("-" * 40)

    # Запуск полного перебора
    best_solution_brute, min_penalty_brute, brute_time = brute_force_solver_simplified()
    print("\n=== Результаты ПОЛНОГО ПЕРЕБОРА (упрощенная задача) ===")
    print(f"Время выполнения: {brute_time:.4f} сек.")
    print("Лучшая матрица потоков (оптимальное решение):")
    print(best_solution_brute)
    print("Минимальная стоимость (штраф):", min_penalty_brute)
    print("-" * 40)

    # Построение графика для сравнения мутаций
    plt.plot(history_sp, label='Одноточечная')
    plt.plot(history_re, label='Сброса')
    plt.plot(history_rd, label='Перераспределения')
    plt.title("Сравнение мутаций")
    plt.xlabel("Поколение")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()