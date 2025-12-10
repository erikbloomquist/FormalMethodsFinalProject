# tom_jerry_mdp.py
# Temporal Logic–Constrained Control in MDPs
# Spec: Avoid all traps and Tom while maximizing probability of reaching cheese.
# This script compares Jerry's optimal policy/value under (1) random Tom, (2) heuristic-chasing Tom.

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Grid setup and indexing
# -----------------------------
GRID_SIZE = 5
NUM_CELLS = GRID_SIZE * GRID_SIZE

def rc_to_pos(r, c):
    return r * GRID_SIZE + c

def pos_to_rc(pos):
    return divmod(pos, GRID_SIZE)

# Environment layout: traps and cheese (code coordinates: row 0 = top)
TRAP_CELLS = {
    rc_to_pos(0, 0),  # top-left
    rc_to_pos(3, 2),  # lower trap
}
CHEESE_CELLS = {
    rc_to_pos(2, 1),  # left cheese
    rc_to_pos(2, 4),  # right cheese
}

is_trap = np.zeros(NUM_CELLS, dtype=bool)
is_cheese = np.zeros(NUM_CELLS, dtype=bool)
for p in TRAP_CELLS:
    is_trap[p] = True
for p in CHEESE_CELLS:
    is_cheese[p] = True

# -----------------------------
# Actions and single-agent dynamics
# -----------------------------
NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3
ACTIONS = [NORTH, SOUTH, EAST, WEST]
NUM_ACTIONS = len(ACTIONS)

ACTION_DELTAS = {
    NORTH: (-1, 0),
    SOUTH: ( 1, 0),
    EAST:  ( 0, 1),
    WEST:  ( 0,-1),
}
LEFT_OF = {NORTH: WEST, SOUTH: EAST, EAST: NORTH, WEST: SOUTH}
RIGHT_OF = {NORTH: EAST, SOUTH: WEST, EAST: SOUTH, WEST: NORTH}

def move_cell(pos, action):
    """Deterministic move; if off-grid, stay."""
    r, c = pos_to_rc(pos)
    dr, dc = ACTION_DELTAS[action]
    nr, nc = r + dr, c + dc
    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
        return rc_to_pos(nr, nc)
    else:
        return pos

def jerry_transition_probs(jerry_pos, action):
    """
    Jerry: 0.8 intended direction, 0.1 left, 0.1 right.
    Returns dict[next_jerry_pos] = prob.
    """
    intended = action
    left = LEFT_OF[action]
    right = RIGHT_OF[action]
    moves = [intended, left, right]
    probs = [0.8, 0.1, 0.1]
    out = {}
    for a, p in zip(moves, probs):
        nxt = move_cell(jerry_pos, a)
        out[nxt] = out.get(nxt, 0.0) + p
    return out

# -----------------------------
# Tom policies
# -----------------------------
def tom_transition_probs_random(tom_pos):
    """Tom: uniform random N,S,E,W."""
    out = {}
    for a in ACTIONS:
        nxt = move_cell(tom_pos, a)
        out[nxt] = out.get(nxt, 0.0) + 1.0 / NUM_ACTIONS
    return out

def tom_heuristic_action(tom_pos, jerry_pos):
    """
    Manhattan heuristic toward Jerry.
    Tie-breaking: if multiple actions give same min distance, break uniformly.
    """
    jr, jc = pos_to_rc(jerry_pos)
    best_actions = []
    best_dist = float('inf')
    for a in ACTIONS:
        nxt = move_cell(tom_pos, a)
        nr, nc = pos_to_rc(nxt)
        dist = abs(nr - jr) + abs(nc - jc)
        if dist < best_dist:
            best_dist = dist
            best_actions = [a]
        elif dist == best_dist:
            best_actions.append(a)
    # uniform tie-break
    return np.random.choice(best_actions) if len(best_actions) > 1 else best_actions[0]

def tom_transition_probs_heuristic(tom_pos, jerry_pos, chase_p=0.7):
    """
    Tom: chase Jerry with probability chase_p, otherwise uniform random (1 - chase_p).
    Returns dict[next_tom_pos] = prob.
    """
    chase_p = float(chase_p)
    assert 0.0 <= chase_p <= 1.0
    out = {}
    # chase branch
    best_action = tom_heuristic_action(tom_pos, jerry_pos)
    nxt = move_cell(tom_pos, best_action)
    out[nxt] = out.get(nxt, 0.0) + chase_p
    # random branch
    rand_p = (1.0 - chase_p) / NUM_ACTIONS
    for a in ACTIONS:
        nxt = move_cell(tom_pos, a)
        out[nxt] = out.get(nxt, 0.0) + rand_p
    return out

# -----------------------------
# Joint state space
# -----------------------------
NUM_STATES = NUM_CELLS * NUM_CELLS

def encode_state(jerry_pos, tom_pos):
    return jerry_pos * NUM_CELLS + tom_pos

def decode_state(s):
    j = s // NUM_CELLS
    t = s % NUM_CELLS
    return j, t

def is_bad_state(j, t):
    return is_trap[j] or (j == t)

def is_goal_state(j, t):
    return is_cheese[j]

# -----------------------------
# Joint transition with caching
# -----------------------------
_TRANSITION_CACHE = {}

def joint_transition(s, a, tom_policy='random', chase_p=0.7):
    """
    Return dict[next_state] = prob under given action and Tom policy.
    tom_policy: 'random' or 'heuristic'; chase_p only used for 'heuristic'.
    """
    key = (s, a, tom_policy, round(float(chase_p), 6))
    if key in _TRANSITION_CACHE:
        return _TRANSITION_CACHE[key]

    j, t = decode_state(s)
    if is_goal_state(j, t) or is_bad_state(j, t):
        out = {s: 1.0}
        _TRANSITION_CACHE[key] = out
        return out

    out = {}
    j_probs = jerry_transition_probs(j, a)
    if tom_policy == 'random':
        t_probs = tom_transition_probs_random(t)
    elif tom_policy == 'heuristic':
        t_probs = tom_transition_probs_heuristic(t, j, chase_p=chase_p)
    else:
        raise ValueError(f"Unknown tom_policy: {tom_policy}")

    for j_next, pj in j_probs.items():
        for t_next, pt in t_probs.items():
            s_next = encode_state(j_next, t_next)
            out[s_next] = out.get(s_next, 0.0) + pj * pt

    _TRANSITION_CACHE[key] = out
    return out

# -----------------------------
# Value iteration (reachability)
# -----------------------------
def value_iteration_reachability(tom_policy='random', chase_p=0.7, tol=1e-7, max_iter=10_000, verbose=True):
    """
    Solve V(s) = max_a E[V(s')] under boundary conditions:
        V(goal) = 1, V(bad) = 0 (absorbing).
    tom_policy: 'random' or 'heuristic'; chase_p for heuristic aggressiveness.
    Returns (V, policy).
    """
    V = np.zeros(NUM_STATES, dtype=float)

    # Terminal masks
    goal_mask = np.zeros(NUM_STATES, dtype=bool)
    bad_mask = np.zeros(NUM_STATES, dtype=bool)
    for s in range(NUM_STATES):
        j, t = decode_state(s)
        if is_goal_state(j, t):
            goal_mask[s] = True
        if is_bad_state(j, t):
            bad_mask[s] = True

    # boundary conditions
    V[goal_mask] = 1.0
    V[bad_mask] = 0.0

    for it in range(max_iter):
        delta = 0.0
        V_old = V.copy()

        for s in range(NUM_STATES):
            if goal_mask[s] or bad_mask[s]:
                continue
            best_q = -1.0
            for a in ACTIONS:
                q = 0.0
                trans = joint_transition(s, a, tom_policy=tom_policy, chase_p=chase_p)
                for s_next, p in trans.items():
                    q += p * V_old[s_next]
                if q > best_q:
                    best_q = q
            V[s] = best_q
            delta = max(delta, abs(V[s] - V_old[s]))

        if verbose:
            print(f"Iter {it}, delta = {delta:.3e}")
        if delta < tol:
            if verbose:
                print(f"Converged in {it} iterations.")
            break

    # greedy policy extraction
    policy = np.full(NUM_STATES, -1, dtype=int)
    for s in range(NUM_STATES):
        if goal_mask[s] or bad_mask[s]:
            continue
        best_a = NORTH
        best_q = -1.0
        for a in ACTIONS:
            q = 0.0
            trans = joint_transition(s, a, tom_policy=tom_policy, chase_p=chase_p)
            for s_next, p in trans.items():
                q += p * V[s_next]
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a

    return V, policy

# -----------------------------
# Aggregation and ASCII policy
# -----------------------------
def jerry_value_marginal_over_tom(V, tom_initial_dist=None):
    """Aggregate V over a prior distribution on Tom's initial cell."""
    if tom_initial_dist is None:
        tom_initial_dist = np.ones(NUM_CELLS) / NUM_CELLS
    tom_initial_dist = np.asarray(tom_initial_dist)
    assert tom_initial_dist.shape == (NUM_CELLS,)
    assert np.allclose(tom_initial_dist.sum(), 1.0)

    jerry_vals = np.zeros(NUM_CELLS)
    for j in range(NUM_CELLS):
        val = 0.0
        for t in range(NUM_CELLS):
            s = encode_state(j, t)
            val += tom_initial_dist[t] * V[s]
        jerry_vals[j] = val
    return jerry_vals.reshape(GRID_SIZE, GRID_SIZE)

def jerry_policy_from_joint_ascii(V, tom_policy='random', chase_p=0.7, tom_initial_dist=None):
    """
    Aggregated arrows for Jerry by averaging Q over Tom's initial distribution.
    Returns (arrows_code, arrows_figure).
    """
    if tom_initial_dist is None:
        tom_initial_dist = np.ones(NUM_CELLS) / NUM_CELLS
    tom_initial_dist = np.asarray(tom_initial_dist)
    assert tom_initial_dist.shape == (NUM_CELLS,)
    assert np.allclose(tom_initial_dist.sum(), 1.0)

    arrow_for_action = {NORTH: "↑", SOUTH: "↓", EAST: "→", WEST: "←"}
    arrows_code = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)

    for j in range(NUM_CELLS):
        best_a = NORTH
        best_q = -1.0
        for a in ACTIONS:
            q = 0.0
            for t in range(NUM_CELLS):
                s = encode_state(j, t)
                trans = joint_transition(s, a, tom_policy=tom_policy, chase_p=chase_p)
                for s_next, p in trans.items():
                    q += tom_initial_dist[t] * p * V[s_next]
            if q > best_q:
                best_q = q
                best_a = a
        r, c = pos_to_rc(j)
        arrows_code[r, c] = arrow_for_action[best_a]

    arrows_figure = np.flipud(arrows_code)  # match figure convention (row 0 at bottom)
    return arrows_code, arrows_figure

# -----------------------------
# Visualization helpers
# -----------------------------
def add_markers(ax):
    """Overlay traps (X) and cheese (C) on a heatmap axes."""
    for p in TRAP_CELLS:
        r, c = pos_to_rc(p)
        ax.text(c, r, 'X', color='black', fontsize=14, ha='center', va='center', fontweight='bold')
    for p in CHEESE_CELLS:
        r, c = pos_to_rc(p)
        ax.text(c, r, 'C', color='black', fontsize=14, ha='center', va='center', fontweight='bold')

def plot_comparison(grid_vals_random, grid_vals_heuristic, save_path="comparison.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axes[0].imshow(grid_vals_random, cmap='RdYlGn', vmin=0, vmax=1)
    axes[0].set_title("Random Tom")
    add_markers(axes[0])
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(grid_vals_heuristic, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1].set_title("Heuristic Tom")
    add_markers(axes[1])
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    diff = grid_vals_random - grid_vals_heuristic
    vlim = max(abs(diff.min()), abs(diff.max()))
    im3 = axes[2].imshow(diff, cmap='coolwarm', vmin=-vlim, vmax=vlim)
    axes[2].set_title("Advantage of Random Tom (Δ)")
    add_markers(axes[2])
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("Col")
        ax.set_ylabel("Row")
        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

# -----------------------------
# Main experiment
# -----------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    print("Solving with random Tom...")
    V_random, policy_random = value_iteration_reachability(tom_policy='random', tol=1e-7, verbose=True)
    grid_vals_random = jerry_value_marginal_over_tom(V_random)
    print("\nJerry success probabilities (random Tom):")
    print(grid_vals_random)

    print("\nSolving with heuristic Tom (chase_p=0.7)...")
    V_heuristic, policy_heuristic = value_iteration_reachability(tom_policy='heuristic', chase_p=0.7, tol=1e-7, verbose=True)
    grid_vals_heuristic = jerry_value_marginal_over_tom(V_heuristic)
    print("\nJerry success probabilities (heuristic Tom):")
    print(grid_vals_heuristic)

    # difference
    diff = grid_vals_random - grid_vals_heuristic
    print("\nSuccess probability decrease with heuristic Tom (random - heuristic):")
    print(diff)

    # sample initial state value
    jerry_start = rc_to_pos(4, 0)
    tom_start = rc_to_pos(0, 2)
    s0 = encode_state(jerry_start, tom_start)
    print(f"\nSample V_random at (Jerry={jerry_start}, Tom={tom_start}): {V_random[s0]:.4f}")
    print(f"Sample V_heuristic at (Jerry={jerry_start}, Tom={tom_start}): {V_heuristic[s0]:.4f}")

    # ASCII arrows (aggregated policy) under each Tom policy
    arrows_code_rand, _ = jerry_policy_from_joint_ascii(V_random, tom_policy='random')
    print("\nJerry aggregated optimal directions (random Tom):")
    for r in range(GRID_SIZE):
        print(" ".join(arrows_code_rand[r]))

    arrows_code_heur, _ = jerry_policy_from_joint_ascii(V_heuristic, tom_policy='heuristic', chase_p=0.7)
    print("\nJerry aggregated optimal directions (heuristic Tom):")
    for r in range(GRID_SIZE):
        print(" ".join(arrows_code_heur[r]))

    # Visualization
    print("\nRendering comparison heatmaps...")
    plot_comparison(grid_vals_random, grid_vals_heuristic, save_path="comparison.png")

    # Optional: save CSVs
    np.savetxt("jerry_success_random.csv", grid_vals_random, delimiter=",", fmt="%.4f")
    np.savetxt("jerry_success_heuristic.csv", grid_vals_heuristic, delimiter=",", fmt="%.4f")
    np.savetxt("jerry_success_diff.csv", diff, delimiter=",", fmt="%.4f")