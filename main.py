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

    # Штраф за превышение производства
    if any(individual[i, :].sum() > production_points[i]['supply'] for i in range(N)):
        return 0

    # 1. Считаем транспортные расходы
    for i in range(N):
        for j in range(K):
            total_cost += individual[i, j] * distance_matrix[i, j] * price_per_unit_distance

    # 2. Считаем штраф за превышение потребления в городах
    for j in range(K):
        total_supply_to_city = individual[:, j].sum()
        if total_supply_to_city > cities[j]['demand']:
            excess = total_supply_to_city - cities[j]['demand']
            excess_penalty += excess * penalty_per_excess_unit

    total_penalty = total_cost + excess_penalty
    return 1 / (1 + total_penalty)


# --- 3. ОПЕРАТОРЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА ---

def create_initial_population(size):
    population = []
    for _ in range(size):
        individual = np.zeros((N, K))
        for i in range(N):
            supply_left = production_points[i]['supply']

            # Распределяем производство
            demand_ratios = np.array([cities[j]['demand'] for j in range(K)])
            # Добавляем маленькую константу, чтобы избежать деления на ноль
            demand_ratios = demand_ratios / (demand_ratios.sum() + 1e-6)

            flows = np.random.multinomial(supply_left, demand_ratios)
            for j in range(K):
                individual[i, j] = flows[j]
        population.append(individual)
    return population


def selection(population, num_parents):
    # Турнирный отбор
    selected = []
    for _ in range(num_parents):
        # Выбираем 3 случайных "участника"
        tournament_pool = random.sample(population, 3)
        # Выбираем лучшего из них
        winner = max(tournament_pool, key=calculate_fitness)
        selected.append(winner)
    return selected


def crossover(parent1, parent2):
    # Одноточечное скрещивание по строкам (пунктам производства)
    crossover_point = random.randint(1, N - 1)
    child1 = np.vstack((parent1[:crossover_point, :], parent2[crossover_point:, :]))
    child2 = np.vstack((parent2[:crossover_point, :], parent1[crossover_point:, :]))
    return child1, child2


def mutation(individual):
    # Случайное изменение одного гена (потока)
    if random.random() < MUTATION_RATE:
        row = random.randint(0, N - 1)
        col = random.randint(0, K - 1)

        # Случайно изменяем значение
        change = random.randint(-20, 20)
        individual[row, col] = max(0, individual[row, col] + change)
    return individual


# --- 4. ОСНОВНОЙ ЦИКЛ ГЕНЕТИЧЕСКОГО АЛГОРИТМА ---
def genetic_algorithm():
    population = create_initial_population(POPULATION_SIZE)
    best_fitness_history = []

    for generation in range(GENERATIONS):
        # Оценка приспособленности
        fitness_scores = [calculate_fitness(ind) for ind in population]
        best_fitness_history.append(max(fitness_scores))

        # Выбор родителей
        parents = selection(population, POPULATION_SIZE // 2)

        # Создание нового поколения
        next_generation = []
        while len(next_generation) < POPULATION_SIZE:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            # Скрещивание
            child1, child2 = crossover(parent1, parent2)

            # Мутация
            next_generation.append(mutation(child1))
            next_generation.append(mutation(child2))

        population = next_generation

    best_individual = max(population, key=calculate_fitness)
    best_fitness = calculate_fitness(best_individual)

    return best_individual, best_fitness, best_fitness_history


# --- 5. ФУНКЦИЯ ПОЛНОГО ПЕРЕБОРА (УПРОЩЕННАЯ) ---
def brute_force_solver_simplified():
    print("--- Запуск полного перебора для упрощенной задачи (2x2) ---")

    # Создаем упрощенные данные для демонстрации
    N_brute = 2
    K_brute = 2
    Y_max_brute = 10

    brute_production_points = [{'supply': 10}, {'supply': 10}]
    brute_cities = [{'demand': 10}, {'demand': 10}]
    brute_distance_matrix = np.array([[10, 20], [30, 15]])
    brute_price_per_unit_distance = 1
    brute_penalty_per_excess_unit = 5

    def calculate_penalty_brute(individual):
        total_cost = 0
        excess_penalty = 0

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

        return total_cost + excess_penalty

    min_penalty = float('inf')
    best_solution = None

    start_time = time.time()

    # Генерируем все возможные комбинации потоков для 2x2 матрицы
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
    # --- Запуск генетического алгоритма ---
    ga_start_time = time.time()
    best_solution_ga, best_fit_ga, history = genetic_algorithm()
    ga_end_time = time.time()

    print("=== Результаты ГЕНЕТИЧЕСКОГО АЛГОРИТМА ===")
    print(f"Время выполнения: {ga_end_time - ga_start_time:.4f} сек.")
    print("Лучшее найденное решение (матрица потоков):")
    print(best_solution_ga)
    print("\nЗначение функции приспособленности (Fitness):", best_fit_ga)
    print("Общая стоимость (с учетом штрафов):", 1 / best_fit_ga - 1)
    print("-" * 40)

    # --- Запуск полного перебора (упрощенная версия) ---
    best_solution_brute, min_penalty_brute, brute_time = brute_force_solver_simplified()

    print("\n=== Результаты ПОЛНОГО ПЕРЕБОРА (упрощенная задача) ===")
    print(f"Время выполнения: {brute_time:.4f} сек.")
    print("Лучшая матрица потоков (оптимальное решение):")
    print(best_solution_brute)
    print("Минимальная стоимость (штраф):", min_penalty_brute)
    print("-" * 40)

    # Построение графика
    plt.plot(history)
    plt.title("Изменение лучшей приспособленности по поколениям (ГА)")
    plt.xlabel("Поколение")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.show()