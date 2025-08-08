# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import io
import base64
import requests
from datetime import datetime

# -------------------------
# Utility functions
# -------------------------
def set_seed(s):
    random.seed(s)
    np.random.seed(s)

def generate_cities(n, width=100, height=100, seed=None):
    if seed is not None:
        set_seed(seed)
    return np.random.rand(n, 2) * [width, height]

def distance_matrix(cities):
    n = len(cities)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dm[i, j] = np.linalg.norm(cities[i] - cities[j])
    return dm

def route_length(route, dm):
    n = len(route)
    L = 0.0
    for i in range(n):
        a = route[i]
        b = route[(i+1) % n]
        L += dm[a, b]
    return L

def plot_route(cities, route, title=None, figsize=(5,5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(cities[:,0], cities[:,1])
    r = list(route) + [route[0]]
    pts = cities[r]
    ax.plot(pts[:,0], pts[:,1], '-o')
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

# -------------------------
# Ant Colony Optimization
# -------------------------
def aco_tsp(cities, num_ants=20, alpha=1.0, beta=3.0, evaporation=0.5, Q=1.0, iterations=100, seed=None, verbose=False):
    if seed is not None:
        set_seed(seed)
    n = len(cities)
    dm = distance_matrix(cities)
    # To avoid division by zero:
    eta = np.zeros((n,n))
    with np.errstate(divide='ignore'):
        eta = 1.0 / dm
    eta[eta == np.inf] = 1e9
    pheromone = np.ones((n,n))
    best_route = None
    best_length = float('inf')
    history = []

    for it in range(iterations):
        all_routes = []
        all_lengths = []

        for a in range(num_ants):
            # build a tour
            start = random.randrange(n)
            tour = [start]
            visited = set(tour)
            while len(tour) < n:
                current = tour[-1]
                probs = np.zeros(n)
                for j in range(n):
                    if j in visited:
                        probs[j] = 0.0
                    else:
                        probs[j] = (pheromone[current,j] ** alpha) * (eta[current,j] ** beta)
                s = probs.sum()
                if s == 0:
                    # fallback: choose random unvisited
                    choices = [j for j in range(n) if j not in visited]
                    next_city = random.choice(choices)
                else:
                    probs = probs / s
                    next_city = np.random.choice(range(n), p=probs)
                tour.append(next_city)
                visited.add(next_city)

            L = route_length(tour, dm)
            all_routes.append(tour)
            all_lengths.append(L)
            if L < best_length:
                best_length = L
                best_route = tour.copy()

        # evaporate
        pheromone = pheromone * (1.0 - evaporation)
        # deposit
        for tour, L in zip(all_routes, all_lengths):
            deposit = Q / L
            for i in range(n):
                a = tour[i]
                b = tour[(i+1) % n]
                pheromone[a,b] += deposit
                pheromone[b,a] += deposit

        history.append(best_length)
        if verbose and (it % max(1, iterations//10) == 0):
            st.write(f"ACO iter {it+1}/{iterations} best {best_length:.3f}")

    return {"best_route": best_route, "best_length": best_length, "history": history}

# -------------------------
# Permutation PSO for TSP (simple swap-based velocity)
# -------------------------
# Particle representation: permutation (list of city indices)
# Velocity representation: list of swaps (tuples (i,j))
def apply_swaps(perm, swaps):
    perm = perm.copy()
    for (i,j) in swaps:
        perm[i], perm[j] = perm[j], perm[i]
    return perm

def swaps_from_perm(a, b):
    # find a sequence of swaps to transform a -> b (simple greedy)
    a = a.copy()
    swaps = []
    pos = {val:i for i,val in enumerate(a)}
    for i in range(len(a)):
        if a[i] != b[i]:
            j = pos[b[i]]
            swaps.append((i,j))
            # perform swap in a and update positions
            a[i], a[j] = a[j], a[i]
            pos[a[j]] = j
            pos[a[i]] = i
    return swaps

def combine_swaps(s1, s2, w1=1.0, w2=1.0, prob_keep=0.5):
    # simple combination: take random subset proportional to weights
    out = []
    # keep some from s1
    for s in s1:
        if random.random() < (prob_keep * w1):
            out.append(s)
    for s in s2:
        if random.random() < (prob_keep * w2):
            out.append(s)
    # unique-ify (naive) by keeping order but removing duplicates of positions
    used_positions = set()
    filtered = []
    for (i,j) in out:
        key = tuple(sorted((i,j)))
        if key not in used_positions:
            filtered.append((i,j))
            used_positions.add(key)
    return filtered

def p_pso_tsp(cities, num_particles=30, w=0.8, c1=1.2, c2=1.2, iterations=100, seed=None, verbose=False):
    if seed is not None:
        set_seed(seed)
    n = len(cities)
    dm = distance_matrix(cities)
    # initialize particles as random permutations
    particles = [list(np.random.permutation(n)) for _ in range(num_particles)]
    pbest = [p.copy() for p in particles]
    pbest_scores = [route_length(p, dm) for p in particles]
    gbest_idx = int(np.argmin(pbest_scores))
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]
    # velocities stored as list of swaps
    velocities = [[] for _ in range(num_particles)]
    history = []

    for it in range(iterations):
        for i in range(num_particles):
            # generate swaps toward personal best and global best
            swaps_to_pbest = swaps_from_perm(particles[i], pbest[i])
            swaps_to_gbest = swaps_from_perm(particles[i], gbest)

            # probabilistic combination with inertia-like effect
            new_velocity = combine_swaps(velocities[i], swaps_to_pbest, w1=w, w2=c1, prob_keep=0.6)
            new_velocity = combine_swaps(new_velocity, swaps_to_gbest, w1=1.0, w2=c2, prob_keep=0.6)

            # apply some randomness
            # small chance to apply a random swap
            if random.random() < 0.05:
                a = random.randrange(n)
                b = random.randrange(n)
                new_velocity.append((a,b))

            # limit velocity length to keep things reasonable
            if len(new_velocity) > n:
                new_velocity = new_velocity[:n]

            # update particle
            particles[i] = apply_swaps(particles[i], new_velocity)
            velocities[i] = new_velocity

            # evaluate
            score = route_length(particles[i], dm)
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest[i] = particles[i].copy()
                if score < gbest_score:
                    gbest_score = score
                    gbest = particles[i].copy()

        history.append(gbest_score)
        if verbose and (it % max(1, iterations//10) == 0):
            st.write(f"P-PSO iter {it+1}/{iterations} best {gbest_score:.3f}")

    return {"best_route": gbest, "best_length": gbest_score, "history": history}

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(layout="wide", page_title="Swarm Playground (TSP)")

st.title("🐜 Swarm Playground — TSP (ACO vs Permutation-PSO)")
st.markdown(
    "Interactive playground that runs **Ant Colony Optimization (ACO)** and a simple **Permutation-PSO** "
    "on the same randomly generated TSP instance. No datasets, everything generated at runtime."
)

col1, col2 = st.columns([1,1])

with col1:
    st.header("Problem & Randomness")
    num_cities = st.slider("Number of cities", 5, 30, 12)
    seed = st.number_input("Random seed (leave 0 for random)", value=0, step=1)
    seed_val = None if seed == 0 else int(seed)
    width = st.number_input("Canvas width", value=100, step=10)
    height = st.number_input("Canvas height", value=100, step=10)
    regenerate = st.button("Regenerate Cities")

with col2:
    st.header("Global Controls")
    iterations = st.slider("Iterations (both algos)", 10, 500, 100)
    show_convergence = st.checkbox("Show convergence plots", value=True)
    run_battle = st.button("Run Battle (ACO vs P-PSO)")
    run_single = st.button("Run ACO only")

# parameters area
st.markdown("---")
st.subheader("Algorithm Parameters")

aco_col, pso_col = st.columns(2)
with aco_col:
    st.markdown("**Ant Colony (ACO)**")
    aco_num_ants = st.slider("Number of ants", 5, 100, 30)
    aco_alpha = st.slider("Alpha (pheromone importance)", 0.1, 5.0, 1.0)
    aco_beta = st.slider("Beta (heuristic importance)", 0.1, 10.0, 2.0)
    aco_evap = st.slider("Evaporation rate", 0.01, 1.0, 0.3)
    aco_iters = iterations

with pso_col:
    st.markdown("**Permutation PSO (P-PSO)**")
    pso_particles = st.slider("Number of particles", 5, 100, 40)
    pso_w = st.slider("Inertia (w)", 0.0, 1.5, 0.8)
    pso_c1 = st.slider("Cognitive (c1)", 0.0, 2.0, 1.2)
    pso_c2 = st.slider("Social (c2)", 0.0, 2.0, 1.2)
    pso_iters = iterations

# storage for cities
if 'cities' not in st.session_state or regenerate:
    st.session_state.cities = generate_cities(num_cities, width, height, seed_val)

# update when number of cities changes
if len(st.session_state.cities) != num_cities:
    st.session_state.cities = generate_cities(num_cities, width, height, seed_val)

cities = st.session_state.cities

# show cities
st.subheader("City Map")
fig_cities = plt.figure(figsize=(4,4))
plt.scatter(cities[:,0], cities[:,1])
plt.title("Cities")
plt.xticks([])
plt.yticks([])
st.pyplot(fig_cities)

# run ACO only or battle
if run_single:
    with st.spinner("Running ACO..."):
        res_aco = aco_tsp(cities, num_ants=aco_num_ants, alpha=aco_alpha, beta=aco_beta,
                          evaporation=aco_evap, iterations=aco_iters, seed=seed_val, verbose=True)
    st.success(f"ACO done — best length {res_aco['best_length']:.3f}")
    fig = plot_route(cities, res_aco['best_route'], title=f"ACO best ({res_aco['best_length']:.2f})")
    st.pyplot(fig)
    if show_convergence:
        st.line_chart({"ACO": res_aco['history']})

if run_battle:
    # run both
    with st.spinner("Running ACO and P-PSO (this may take a moment)..."):
        res_aco = aco_tsp(cities, num_ants=aco_num_ants, alpha=aco_alpha, beta=aco_beta,
                          evaporation=aco_evap, iterations=aco_iters, seed=seed_val, verbose=False)
        res_pso = p_pso_tsp(cities, num_particles=pso_particles, w=pso_w, c1=pso_c1, c2=pso_c2,
                           iterations=pso_iters, seed=seed_val, verbose=False)

    colA, colB = st.columns(2)
    with colA:
        st.subheader(f"ACO Result — best {res_aco['best_length']:.3f}")
        st.pyplot(plot_route(cities, res_aco['best_route'], title=f"ACO ({res_aco['best_length']:.2f})"))
    with colB:
        st.subheader(f"P-PSO Result — best {res_pso['best_length']:.3f}")
        st.pyplot(plot_route(cities, res_pso['best_route'], title=f"P-PSO ({res_pso['best_length']:.2f})"))

    if show_convergence:
        st.subheader("Convergence comparison")
        import pandas as pd
        # align lengths
        maxlen = max(len(res_aco['history']), len(res_pso['history']))
        h_aco = res_aco['history'] + [res_aco['history'][-1]]*(maxlen - len(res_aco['history']))
        h_pso = res_pso['history'] + [res_pso['history'][-1]]*(maxlen - len(res_pso['history']))
        df = pd.DataFrame({"ACO": h_aco, "P-PSO": h_pso})
        st.line_chart(df)

    # show numeric comparison
    if res_aco['best_length'] < res_pso['best_length']:
        st.success(f"Winner: ACO ( {res_aco['best_length']:.3f} < {res_pso['best_length']:.3f} )")
    elif res_pso['best_length'] < res_aco['best_length']:
        st.success(f"Winner: P-PSO ( {res_pso['best_length']:.3f} < {res_aco['best_length']:.3f} )")
    else:
        st.info("Tie!")

    # allow saving results to GitHub
    st.markdown("---")
    st.subheader("Save results to GitHub (optional)")
    st.markdown(
        "If you want to save the best-route image + route text to a GitHub repo, provide: "
        "`owner/repo` and a Personal Access Token (repo scope). **Be careful with your token.**"
    )
    gh_token = st.text_input("GitHub token (keep secret!)", type="password")
    gh_repo = st.text_input("Repo (owner/repo) e.g. username/swarm-playground")
    if st.button("Save ACO + P-PSO results to GitHub"):
        if not gh_token or not gh_repo:
            st.error("Provide token and repo")
        else:
            # prepare image for ACO
            buf = io.BytesIO()
            fig_aco = plot_route(cities, res_aco['best_route'], title=f"ACO_{res_aco['best_length']:.2f}")
            fig_aco.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_bytes = buf.read()
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            img_path_aco = f"results/aco_route_{ts}.png"
            code, resp = save_image_to_github(gh_token, gh_repo, img_path_aco, img_bytes, message=f"ACO route {ts}")
            if code in (200,201):
                st.success(f"Saved ACO image to {gh_repo}/{img_path_aco}")
            else:
                st.error(f"Failed to save ACO image: {resp}")

            # save route text
            route_txt = f"ACO best length: {res_aco['best_length']}\nroute: {res_aco['best_route']}\n\n"
            route_txt += f"P-PSO best length: {res_pso['best_length']}\nroute: {res_pso['best_route']}\n"
            txt_path = f"results/routes_{ts}.txt"
            code2, resp2 = save_text_to_github(gh_token, gh_repo, txt_path, route_txt, message=f"routes {ts}")
            if code2 in (200,201):
                st.success(f"Saved routes to {gh_repo}/{txt_path}")
            else:
                st.error(f"Failed to save routes: {resp2}")

# footer/help
st.markdown("---")
st.markdown(
    "### Tips\n"
    "- Increase iterations to get better results (but runtime increases).\n"
    "- Reduce number of cities for faster runs during demos.\n"
    "- For class demo: set seed for reproducible runs so you can show the professor the same behavior repeatedly.\n"
    "- This is intentionally simple and educational — many improvements (elitism, better PSO velocity encoding) are possible."
)
