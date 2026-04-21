import numpy as np

# Configuration
hp = [0.7298, 2.05, 2.05] #Chi, c1, c2
num_particles = 12
max_iterations = 500

def pso(cost_fn, dimension):

    particles_position = np.random.uniform(-np.pi, np.pi, size=(num_particles, dimension))
    particles_velocity = np.random.uniform(-np.pi, np.pi, size=(num_particles, dimension))

    personal_best_positions = particles_position.copy()
    personal_best_values = np.array([cost_fn(p) for p in personal_best_positions])

    global_best_index = np.argmin(personal_best_values)
    global_best_value = personal_best_values[global_best_index]
    global_best_position = personal_best_positions[global_best_index]

    iteration = 0
    while iteration < max_iterations:
        r1, r2 = np.random.rand(num_particles, dimension), np.random.rand(num_particles, dimension)
        particles_velocity = hp[0] * (particles_velocity + hp[1] * r1 * (personal_best_positions - particles_position) + hp[2] * r2 * (global_best_position - particles_position))
        particles_position = (particles_position + particles_velocity)  # Update positions


        current_values = np.array([cost_fn(p) for p in particles_position])

        # Update personal bests
        improved = current_values < personal_best_values
        personal_best_values[improved] = current_values[improved]
        personal_best_positions[improved] = particles_position[improved]

        # Update global best
        global_best_index = np.argmin(personal_best_values)
        global_best_value = personal_best_values[global_best_index]
        global_best_position = personal_best_positions[global_best_index]
        iteration += 1

    return global_best_value, global_best_position