from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import random

app = Flask(__name__)

# Load and preprocess the dataset
dataset_path = r"C:\ai_smartargi\crop_yield.csv"
df = pd.read_csv(dataset_path)

# Ensure column names match the dataset
unique_crops = df['Crop'].unique()
unique_states = df['State'].unique()
unique_seasons = df['Season'].unique()  # Get unique seasons
crop_type_encoding = {crop: i + 1 for i, crop in enumerate(unique_crops)}
state_encoding = {state: i + 1 for i, state in enumerate(unique_states)}
season_encoding = {season: i + 1 for i, season in enumerate(unique_seasons)}  # Encode seasons

# Prepare data array: [Rainfall, Fertilizer, Pesticide, Area, CropType, State, Season, Yield]
data = []
for _, row in df.iterrows():
    features = [
        row['Annual_Rainfall'],
        row['Fertilizer'],
        row['Pesticide'],
        row['Area'],
        crop_type_encoding[row['Crop']],
        state_encoding[row['State']],
        season_encoding[row['Season']]  # Add encoded season
    ]
    target = row['Yield']
    data.append(features + [target])
data = np.array(data)
num_features = data.shape[1] - 1  # Now 7 features (added Season)

# Genetic Algorithm Parameters
pop_size = 10
chromosome_length = num_features  # Update to 7
epsilon = 1e-6

# Optimized Genetic Algorithm Functions
def fitness(chromosome):
    features = data[:, :-1]
    actual_yields = data[:, -1]
    weighted_sums = np.dot(features, chromosome)
    weight_total = np.sum(chromosome) + epsilon
    predicted_yields = weighted_sums / weight_total
    errors = np.abs(predicted_yields - actual_yields)
    avg_error = np.mean(errors)
    return 1 / (1 + avg_error)

def selection(population):
    fitness_scores = np.array([fitness(chromo) for chromo in population])
    top_indices = np.argsort(fitness_scores)[-2:]
    return population[top_indices]

def crossover(parent1, parent2):
    point = random.randint(1, chromosome_length - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def mutate(chromosome, mutation_rate=0.1):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] += np.random.uniform(-0.1, 0.1)
            chromosome[i] = np.clip(chromosome[i], 0, 1)
    return chromosome

def genetic_algorithm(generations=50):
    population = np.random.uniform(low=0, high=1, size=(pop_size, chromosome_length))
    for gen in range(generations):
        selected = selection(population)
        offspring = []
        for _ in range(pop_size // 2):
            p1, p2 = selected
            c1, c2 = crossover(p1, p2)
            offspring.append(mutate(c1))
            offspring.append(mutate(c2))
        population = np.array(offspring)
    return max(population, key=fitness)

# Precompute the best chromosome at startup
print("Training genetic algorithm...")
best_chromosome = genetic_algorithm()
print("Training complete.")

# Flask Routes
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/form')
def form():
    return render_template('form.html', crop_types=unique_crops, states=unique_states, seasons=unique_seasons)

@app.route('/predict', methods=['POST'])
def predict():
    global best_chromosome
    try:
        rainfall = float(request.form['rainfall'])
        fertilizer = float(request.form['fertilizer'])
        pesticide = float(request.form['pesticide'])
        area = float(request.form['area'])
        crop_type = request.form['crop_type']
        state = request.form['state']
        season = request.form['season']  # Get season from form

        crop_type_value = crop_type_encoding.get(crop_type, 0)
        state_value = state_encoding.get(state, 0)
        season_value = season_encoding.get(season, 0)  # Encode season

        # Update user_input to include season
        user_input = np.array([rainfall, fertilizer, pesticide, area, crop_type_value, state_value, season_value])
        weighted_sum = np.dot(best_chromosome, user_input)
        weight_total = np.sum(best_chromosome) + epsilon
        predicted_yield = weighted_sum / weight_total

        return render_template('result.html', prediction=round(predicted_yield, 2))

    except ValueError:
        error_message = "Invalid input! Please enter valid numerical values."
        return render_template('form.html', error=error_message, crop_types=unique_crops, states=unique_states, seasons=unique_seasons)

@app.route('/back')
def back():
    return redirect(url_for('welcome'))

if __name__ == '__main__':
    app.run(debug=True)