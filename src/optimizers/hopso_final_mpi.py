import numpy as np
from mpi4py import MPI

def velocity_for_target_amplitude(position, attractor, A_target, omega=1.0, lamb=0.05, rng=None):
    """
    position, attractor: shape (dimension)
    A_target: scalar or array broadcastable to position.shape (radians)
    """
    if rng is None:
        rng = np.random.default_rng()
    dx = position - attractor
    rad = np.sqrt(np.maximum(A_target**2 - dx**2, 0.0))
    sign = np.where(rng.random(position.shape) < 0.5, -1.0, 1.0)
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
    a = g + alpha * diff
    return wrap_pi(a)          # keep in (-π, π]

def invalid_theta_test(cos_th_i, dead, personal_best_value, theta, run_id, rank, iteration):
    if (np.any(cos_th_i < -1) or np.any(cos_th_i > 1) or np.isnan(cos_th_i).any()):
        personal_best_value = np.inf
        dead = True
        print(f"[Run {run_id}] Particle {rank} killed at iteration {iteration} after pbest update.")
    else:
        theta = np.arccos(cos_th_i)
    return theta, dead, personal_best_value

def hopso(cost_fn, hp, run_id, dimension, max_cut, e_min, max_iterations=500, comm=None):
    """
    Demonstration: each particle updates its attractor/amplitude/theta
    IMMEDIATELY after it finds a better personal best.
    Then, at the end of each iteration, we check global best and do
    a SWARM-WIDE attractor/amplitude/theta update if there's a new global best.
    """
    # Unpack hyperparameters
    w1, w2, tm, lamb = hp # why lamb = hp #

    omega = 1.0

    # Initialize position and velocity
    position = np.random.uniform(-np.pi, np.pi, size = dimension)
    A_target = np.pi 
    #particle_vels = np.random.uniform(-np.pi/2, np.pi/2, size =  dimension)
    

    # Personal best
    personal_best_position = position.copy()
    personal_best_value = cost_fn(position)

    # Define rank
    rank = comm.Get_rank()

    # Initialize global best for the run
    all_personal_best_values = comm.allgather(personal_best_value)
    all_personal_best_positions = comm.allgather(personal_best_position)

    # Global best
    global_best_idx = np.argmin(all_personal_best_values)
    global_best_value = all_personal_best_values[global_best_idx]
    global_best_position = all_personal_best_positions[global_best_idx] #.copy()

    # Particle time, amplitude, attractor, theta
    t = np.zeros(dimension)
    dead = False

    # Compute initial attractor

    attractor = attractor_calc(personal_best_position, global_best_position, w1, w2)
    particle_vel = velocity_for_target_amplitude(position, attractor, A_target, omega=1.0, lamb=hp[3])


    # Initial amplitude
    A = np.sqrt((position - attractor)**2 + (1/omega)**2 * (particle_vel + lamb*(position - attractor))**2)
    d_wrap = circ_dist(personal_best_position, global_best_position)  # [0, π]
    a_floor_1 = (d_wrap/2.0) * max_cut
    A = np.maximum(A, a_floor_1)

    # Initial theta
    cos_theta = (position - attractor)/A
    theta = np.zeros_like(cos_theta)

    # If out of bounds => kill

    iteration = 0

    theta, dead, personal_best_value = invalid_theta_test(cos_theta, dead, personal_best_value, theta, run_id, rank, iteration)

    while iteration < max_iterations:  
        # -------------
        # (A) Evolve each particle
        # -------------
        if not dead:
            # Evolve time & amplitude
            delta_t = np.random.rand(dimension)*tm
            t += delta_t
            A *= np.exp(-lamb * delta_t)

            # Enforce min distance
            # Based on that particle's pbest vs global best
            d_j = circ_dist(personal_best_position, global_best_position)
            a_floor_j = (d_j/2.0)*max_cut
            A = np.maximum(A, a_floor_j)

            # Update position & velocity
            position = A*np.cos(omega*t + theta) + attractor
            particle_vel = A*(-omega*np.sin(omega*t + theta) - lamb*np.cos(omega*t + theta))

            # Evaluate cost
            current_value = cost_fn(position)
            #print(current_value,i)

            # -------------
            # (A.1) If there's a personal best update, 
            # we IMMEDIATELY recalc attractor/amplitude/theta 
            # for that single particle
            # -------------
            if current_value < personal_best_value: 
                personal_best_value = current_value       
                personal_best_position = wrap_pi(position)
                t = 0  # reset time
                

                # Now update attractor dimensionwise for this particle
                attractor = attractor_calc(personal_best_position, global_best_position, w1, w2)

                # Recompute amplitude for that particle
                A1_i = np.sqrt((position - attractor)**2 + (1/omega)**2 * (particle_vel + lamb*(position - attractor))**2)

                # also enforce min distance again
                d_i = circ_dist(personal_best_position, global_best_position)
                a_floor_i = (d_i/2.0) * max_cut
                A = np.maximum(np.maximum(A, A1_i), a_floor_i)

                # Recompute cos_theta for that particle
                cos_th_i = (position - attractor)/A
                # Kill if invalid
                theta, dead, personal_best_value = invalid_theta_test(cos_th_i, dead, personal_best_value, theta, run_id, rank, iteration)


        # -------------
        # (B) After all personal best updates, check global best
        # -------------
        # Gather personal bests from all processes 
        all_personal_best_values = comm.allgather(personal_best_value)
        all_personal_best_positions = comm.allgather(personal_best_position)

        current_best_idx = np.argmin(all_personal_best_values)
        current_best_val = all_personal_best_values[current_best_idx]

        if current_best_val < global_best_value:
            global_best_value = current_best_val
            global_best_position = all_personal_best_positions[current_best_idx].copy()
            # Reset all times
            t = np.zeros(dimension)

            # -------------
            # (B.1) Now do a swarm-wide attractor & amplitude update
            # because the global best changed
            # -------------
            attractor = attractor_calc(personal_best_position, global_best_position, w1, w2)
            
            # Recompute amplitude & kill invalid
            A_all = np.sqrt((position - attractor)**2 + (1/omega)**2 * (particle_vel + lamb*(position - attractor))**2)
            d = circ_dist(personal_best_position, global_best_position)
            a_floor = (d/2.0) * max_cut
            A = np.maximum(np.maximum(A, A_all), a_floor)

            # Update cos_theta for all
            cos_theta = (position - attractor)/A
            if dead:
                continue
            theta, dead, personal_best_value = invalid_theta_test(cos_theta, dead, personal_best_value, theta, run_id, rank, iteration)
        
        iteration += 1

               
    e_min.append(global_best_value)