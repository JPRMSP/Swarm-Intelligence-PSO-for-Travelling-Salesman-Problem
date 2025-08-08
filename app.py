import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

# --------------------
# Particle Swarm Optimization for TSP
# --------------------

# Function to calculate total path distance
def path_distance(path, cities):
    dist = 0
    for i in range(len(path)):
        city_a = cities[path[i]]
        city_b = cities[path[(i + 1) % len(path)]]
        dist += np.linalg.norm(np.array(city_a) - np.array(city_b))
    return dist

# Function to generate random cities
def generate_cities(num_cities):
    return [(random.random(), random.random()) for _ in range(num_cities)]

# PSO algorithm
def pso_tsp(cities, num_particles=30, max_iter=100, w=0.8, c1=1.5, c2=1.5):
    num_cities = len(cities)
    
    # Initialize particles
    particles = [random.sample(range(num_cities), num_cities) for _ in range(num_particles)]
    pbest = particles.copy()
    pbest_scores = [path_distance(p, cities) for p in particles]
    gbest = pbest[np.argmin(pbest_scores)]
    gbest_score = min(pbest_scores)

    progress = []
    
    for _ in range(max_iter):
        for i in range(num_particles):
            # Swap mutation as velocity update
            if random.random() < w:
                a, b = random.sample(range(num_cities), 2)
                particles[i][a], particles[i][b] = particles[i][b], particles[i][a]
            
            # Personal best update
            dist = path_distance(particles[i], cities)
            if dist < pbest_scores[i]:
                pbest[i] = particles[i].copy()
                pbest_scores[i] = dist
            
            # Global best update
            if dist < gbest_score:
                gbest = particles[i].copy()
                gbest_score = dist

        progress.append(gbest_score)
    
    return gbest, gbest_score, progress

# --------------------
# Streamlit App
# --------------------
st.title("🐜 Swarm Intelligence - PSO for Travelling Salesman Problem")
st.write("This demo uses **Particle Swarm Optimization** to solve a randomly generated TSP without any dataset.")

# User inputs
num_cities = st.slider("Number of Cities", 5, 30, 10)
num_particles = st.slider("Number of Particles", 5, 50, 20)
iterations = st.slider("Number of Iterations", 10, 200, 50)

if st.button("Run PSO"):
    cities = generate_cities(num_cities)
    best_path, best_distance, progress = pso_tsp(cities, num_particles=num_particles, max_iter=iterations)

    st.subheader("Best Path Distance: {:.4f}".format(best_distance))

    # Plot best path
    fig, ax = plt.subplots()
    path_coords = [cities[i] for i in best_path] + [cities[best_path[0]]]
    xs, ys = zip(*path_coords)
    ax.plot(xs, ys, marker="o")
    ax.set_title("Best Path Found")
    st.pyplot(fig)

    # Plot progress
    fig2, ax2 = plt.subplots()
    ax2.plot(progress)
    ax2.set_title("Progress Over Iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best Distance")
    st.pyplot(fig2)
