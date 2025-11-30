import numpy as np
def velocities_for_target_amplitude(positions, attractors, A_target, omega=1.0, lamb=0.05, rng=None):
    """
    positions, attractors: shape (num_particles, dimension)
    A_target: scalar or array broadcastable to positions.shape (radians)
    """
    if rng is None:
        rng = np.random.default_rng()
    dx  = positions - attractors
    rad = np.sqrt(np.maximum(A_target**2 - dx**2, 0.0))
    sign = np.where(rng.random(positions.shape) < 0.5, -1.0, 1.0)
    v = -lamb*dx + sign * omega * rad
    return v

def circ_diff(x, y):
    """Signed shortest angular difference in (-π, π]."""
    return (x - y + np.pi) % (2*np.pi) - np.pi

def circ_dist(x, y):
    """Unsigned shortest angular distance in [0, π]."""
    return np.abs(circ_diff(x, y))

def wrap_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def attractor_calc(p, g, w1, w2):
    alpha = w1 / (w1 + w2)
    diff  = wrap_pi(p - g)     # (-π, π]
    a     = g + alpha * diff
    return wrap_pi(a)          # keep in (-π, π]

def hopso(cost_fn, hp, num_particles, runs, dimension, max_cut, e_min, init_positions=None, max_iterations=500):
    """
    Demonstration: each particle updates its attractor/amplitude/theta
    IMMEDIATELY after it finds a better personal best.
    Then, at the end of each iteration, we check global best and do
    a SWARM-WIDE attractor/amplitude/theta update if there's a new global best.
    """

    import numpy as np
    from tqdm import tqdm

    # Unpack hyperparameters
    w1, w2, tm, lamb = hp

    omega = 1.0

    for run_idx in range(runs):

        # Initialize positions and velocities
        positions = np.random.uniform(-np.pi, np.pi, size=(num_particles, dimension))
        A_target = np.pi 
        #particle_vels = np.random.uniform(-np.pi/2, np.pi/2, size=(num_particles, dimension))
        

        # Personal best
        personal_best_positions = positions.copy()
        personal_best_values = np.array([cost_fn(p) for p in positions])

        # Global best
        gbest_idx = np.argmin(personal_best_values)
        global_best_value = personal_best_values[gbest_idx]
        global_best_position = personal_best_positions[gbest_idx].copy()

        # Histories
        gb_run_history = []


        # Particle time, amplitude, attractors, theta
        t        = np.zeros((num_particles, dimension))
        dead     = np.zeros(num_particles, dtype=bool)

        # Compute initial attractors

        attractors = attractor_calc(personal_best_positions, global_best_position, w1, w2)
        particle_vels = velocities_for_target_amplitude(positions, attractors, A_target, omega=1.0, lamb=hp[3])


        # Initial amplitude
        A = np.sqrt((positions - attractors)**2 + (1/omega)**2 * (particle_vels + lamb*(positions - attractors))**2)
        d_wrap = circ_dist(personal_best_positions, global_best_position)  # [0, π]
        a_floor_1 = (d_wrap/2.0) * max_cut
        A = np.maximum(A, a_floor_1)

        # Initial theta
        cos_theta = (positions - attractors)/A
        theta = np.zeros_like(cos_theta)
        for i in range(num_particles):
            # If out of bounds => kill
            if (np.any(cos_theta[i] < -1) or np.any(cos_theta[i] > 1) or np.isnan(cos_theta[i]).any()):
                personal_best_values[i] = np.inf
                dead[i] = True
                print(f"[Run {run_idx}] Particle {i} killed at init.")
            else:
                theta[i] = np.arccos(cos_theta[i])

        for iteration in range(max_iterations):   
            # -------------
            # (A) Evolve each particle
            # -------------
            for i in range(num_particles):
                if dead[i]:
                    continue

                # Evolve time & amplitude
                delta_t = np.random.rand(dimension)*tm
                t[i] += delta_t
                A[i] *= np.exp(-lamb * delta_t)

                # Enforce min distance
                # Based on that particle's pbest vs global best
                d_j = circ_dist(personal_best_positions[i], global_best_position)
                a_floor_j = (d_j/2.0)*max_cut
                A[i] = np.maximum(A[i], a_floor_j)

                # Update position & velocity
                positions[i] = A[i]*np.cos(omega*t[i] + theta[i]) + attractors[i]
                particle_vels[i] = A[i]*(-omega*np.sin(omega*t[i] + theta[i])
                                         - lamb*np.cos(omega*t[i] + theta[i]))

                # Evaluate cost
                current_value = cost_fn(positions[i])
                #print(current_value,i)

                # -------------
                # (A.1) If there's a personal best update, 
                # we IMMEDIATELY recalc attractors/amplitude/theta 
                # for that single particle
                # -------------
                if current_value < personal_best_values[i]: 
                    personal_best_values[i] = current_value       
                    personal_best_positions[i] = wrap_pi(positions[i])
                    t[i] = 0  # reset time
                    

                    # Now update attractor dimensionwise for this particle
                    attractors[i] =attractor_calc(personal_best_positions[i], global_best_position, w1, w2)

                    # Recompute amplitude for that particle
                    A1_i = np.sqrt(
                        (positions[i] - attractors[i])**2
                        + (1/omega)**2 * (particle_vels[i] + lamb*(positions[i] - attractors[i]))**2
                    )
                    # also enforce min distance again
                    d_i = circ_dist(personal_best_positions[i], global_best_position)
                    a_floor_i = (d_i/2.0) * max_cut
                    A[i] = np.maximum(np.maximum(A[i], A1_i), a_floor_i)

                    # Recompute cos_theta for that particle
                    cos_th_i = (positions[i] - attractors[i])/A[i]
                    # Kill if invalid
                    if (np.any(cos_th_i < -1) or np.any(cos_th_i > 1) or np.isnan(cos_th_i).any()):
                        personal_best_values[i] = np.inf
                        dead[i] = True
                        print(f"[Run {run_idx}] Particle {i} killed at iteration {iteration} after pbest update.")
                    else:
                        theta[i] = np.arccos(cos_th_i)

            # -------------
            # (B) After all personal best updates, check global best
            # -------------
            current_best_idx = np.argmin(personal_best_values)
            gbest_idx = current_best_idx
            current_best_val = personal_best_values[current_best_idx]
            if current_best_val < global_best_value:
                global_best_value    = current_best_val
                global_best_position = personal_best_positions[current_best_idx].copy()
                # Reset all times
                t[:] = 0

                # -------------
                # (B.1) Now do a swarm-wide attractor & amplitude update
                # because the global best changed
                # -------------
                attractors = attractor_calc(personal_best_positions, global_best_position, w1, w2)
                
                # Recompute amplitude & kill invalid
                A_all = np.sqrt(
                    (positions - attractors)**2
                    + (1/omega)**2 * (particle_vels + lamb*(positions - attractors))**2
                )
                d = circ_dist(personal_best_positions, global_best_position)
                a_floor = (d/2.0) * max_cut
                A = np.maximum(np.maximum(A, A_all), a_floor)

                # Update cos_theta for all
                cos_theta = (positions - attractors)/A
                for i in range(num_particles):
                    if dead[i]:
                        continue
                    if (np.any(cos_theta[i] < -1) or np.any(cos_theta[i] > 1) or np.isnan(cos_theta[i]).any()):
                        personal_best_values[i] = np.inf
                        dead[i] = True
                        print(f"[Run {run_idx}] Particle {i} killed after global best updated.")
                    else:
                        theta[i] = np.arccos(cos_theta[i])

            gb_run_history.append(global_best_value)

        e_min.append(global_best_value)