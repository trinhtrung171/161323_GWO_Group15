import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# HÀM GWO 
# =============================================================================

def GWO(fobj, lb, ub, dim, SearchAgents_no, Max_iter, variant='original', record_weights=False):
    """
    variant: 'original', 'modified_param', 'weighted_static', 'weighted_dynamic'
    record_weights: nếu True và variant == 'weighted_dynamic' -> trả lại lịch sử w1,w2,w3
    Returns:
        Alpha_score, Alpha_pos, Convergence_curve, optional weight_history (dict)
    """
    # Khởi tạo leaders
    Alpha_pos = np.zeros(dim); Alpha_score = float('inf')
    Beta_pos  = np.zeros(dim); Beta_score  = float('inf')
    Delta_pos = np.zeros(dim); Delta_score = float('inf')

    if not isinstance(lb, (list, np.ndarray)): lb = np.full(dim, lb)
    if not isinstance(ub, (list, np.ndarray)): ub = np.full(dim, ub)

    Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    Convergence_curve = np.zeros(Max_iter)
    epsilon = 1e-12

    # prepare recording
    weight_history = {'w1': [], 'w2': [], 'w3': []} if record_weights else None

    for t in range(Max_iter):
        # --- evaluate population & update leaders
        for i in range(SearchAgents_no):
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            fitness = fobj(Positions[i, :])

            if fitness < Alpha_score:
                Alpha_score, Beta_score, Delta_score = fitness, Alpha_score, Beta_score
                Alpha_pos, Beta_pos, Delta_pos = Positions[i, :].copy(), Alpha_pos.copy(), Beta_pos.copy()
            elif fitness < Beta_score:
                Beta_score, Delta_score = fitness, Beta_score
                Beta_pos, Delta_pos = Positions[i, :].copy(), Beta_pos.copy()
            elif fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()

        # --- a(t): slower decay (keeps exploration longer)
        # a in [0,2]; we make it decay sub-linearly so exploration persists
        frac = t / (Max_iter - 1)
        a = 2 * (1 - frac**0.6)  # exponent <1 => slower drop

        # --- compute X1,X2,X3 (matrix form) ---
        r1_alpha, r2_alpha = np.random.rand(SearchAgents_no, dim), np.random.rand(SearchAgents_no, dim)
        A1 = 2 * a * r1_alpha - a
        C1 = 2 * r2_alpha
        D_alpha = np.abs(C1 * Alpha_pos - Positions)
        X1 = Alpha_pos - A1 * D_alpha

        r1_beta, r2_beta = np.random.rand(SearchAgents_no, dim), np.random.rand(SearchAgents_no, dim)
        A2 = 2 * a * r1_beta - a
        C2 = 2 * r2_beta
        D_beta = np.abs(C2 * Beta_pos - Positions)
        X2 = Beta_pos - A2 * D_beta

        r1_delta, r2_delta = np.random.rand(SearchAgents_no, dim), np.random.rand(SearchAgents_no, dim)
        A3 = 2 * a * r1_delta - a
        C3 = 2 * r2_delta
        D_delta = np.abs(C3 * Delta_pos - Positions)
        X3 = Delta_pos - A3 * D_delta

        # --- variants ---
        if variant == 'weighted_static':
            w1, w2, w3 = 0.6, 0.3, 0.1
            new_Positions = w1 * X1 + w2 * X2 + w3 * X3

        elif variant == 'weighted_dynamic':
            # -----------------------
            # Dynamic Weight 4.1 (improved)
            # -----------------------
            # 1) compute fitness gaps (smaller = better)
            f_min = min(Alpha_score, Beta_score, Delta_score)
            f_max = max(Alpha_score, Beta_score, Delta_score)

            if abs(f_max - f_min) < 1e-14:
                gap_alpha, gap_beta, gap_delta = 1.0, 0.7, 0.3
            else:
                # larger gap => leader is relatively better
                gap_alpha = (f_max - Alpha_score) / (f_max - f_min + epsilon)
                gap_beta  = (f_max - Beta_score)  / (f_max - f_min + epsilon)
                gap_delta = (f_max - Delta_score) / (f_max - f_min + epsilon)

            # 2) amplify differences (power >1)
            power = 2.8  # increased a bit for stronger distinction
            w1_raw = gap_alpha ** power
            w2_raw = gap_beta  ** power
            w3_raw = gap_delta ** power

            # 3) normalize
            S = w1_raw + w2_raw + w3_raw + epsilon
            w1_raw /= S; w2_raw /= S; w3_raw /= S

            # 4) schedule explore/exploit (exponential-like)
            exploit_factor = (1 - a/2) ** 3    # when a small -> exploit_factor ~1
            explore_factor = 1 - exploit_factor

            # 5) blend with uniform to keep diversity early
            w1 = w1_raw * exploit_factor + (1/3) * explore_factor
            w2 = w2_raw * exploit_factor + (1/3) * explore_factor
            w3 = w3_raw * exploit_factor + (1/3) * explore_factor

            # 6) optional small jitter to break symmetry/stuck
            jitter = 1e-4
            w1 += np.random.uniform(-jitter, jitter)
            w2 += np.random.uniform(-jitter, jitter)
            w3 += np.random.uniform(-jitter, jitter)

            # renormalize to sum=1
            Wsum = w1 + w2 + w3 + epsilon
            w1 /= Wsum; w2 /= Wsum; w3 /= Wsum

            # record weights (average across agents for this iteration)
            if record_weights:
                weight_history['w1'].append(w1)
                weight_history['w2'].append(w2)
                weight_history['w3'].append(w3)

            new_Positions = w1 * X1 + w2 * X2 + w3 * X3

        else:
            # original or modified_param
            new_Positions = (X1 + X2 + X3) / 3.0

        # update population
        Positions = np.clip(new_Positions, lb, ub)

        # store best fitness
        Convergence_curve[t] = Alpha_score

    if record_weights and variant == 'weighted_dynamic':
        return Alpha_score, Alpha_pos, Convergence_curve, weight_history
    else:
        return Alpha_score, Alpha_pos, Convergence_curve

# =============================================================================
# HÀM MỤC TIÊU: ROSENBROCK (dễ quan sát, hội tụ chậm)
# =============================================================================
def fobj_rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)

# =============================================================================
# SCRIPT CHẠY (cấu hình để dễ quan sát)
# =============================================================================
if __name__ == "__main__":
    # simulation parameters (chọn kích thước vừa phải để dễ quan sát)
    SearchAgents_no = 30
    Max_iter = 800         # nhiều vòng để quan sát chậm
    dim = 10               # nhỏ hơn để dễ theo dõi
    lb = -2.0
    ub = 2.0
    N_RUNS = 12            # ít lần chạy để từng run dễ quan sát

    variants_to_run = ['original', 'weighted_static', 'weighted_dynamic']

    all_final_scores = {v: [] for v in variants_to_run}
    best_run_curves = {v: None for v in variants_to_run}
    best_run_scores = {v: float('inf') for v in variants_to_run}
    best_weight_history = None

    print("Running experiments (Rosenbrock) ...")
    for run in range(N_RUNS):
        print(f" Run {run+1}/{N_RUNS}")
        # ensure same initial seed per run for fair comparison? we randomize each variant separately
        for v in variants_to_run:
            if v == 'weighted_dynamic':
                score, pos, curve, w_hist = GWO(fobj_rosenbrock, lb, ub, dim,
                                                SearchAgents_no, Max_iter,
                                                variant=v, record_weights=True)
            else:
                score, pos, curve = GWO(fobj_rosenbrock, lb, ub, dim,
                                        SearchAgents_no, Max_iter, variant=v, record_weights=False)

            all_final_scores[v].append(score)
            if score < best_run_scores[v]:
                best_run_scores[v] = score
                best_run_curves[v] = curve
                if v == 'weighted_dynamic':
                    best_weight_history = w_hist  # keep weight history of best dynamic run

    # print summary
    print("\nRESULTS SUMMARY (Lower = Better):")
    for v in variants_to_run:
        arr = np.array(all_final_scores[v])
        print(f" {v:16s} | best: {arr.min():.3e} | mean: {arr.mean():.3e} | median: {np.median(arr):.3e} | std: {arr.std():.3e}")

    # ---------------- Plot convergence curves (best runs) ----------------
    plt.figure(figsize=(11,6))
    for v in variants_to_run:
        curve = best_run_curves[v]
        if curve is not None:
            plt.plot(curve, label=f"{v} (best run)", lw=2)
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness so far (log)")
    plt.title("Convergence (best runs) - Rosenbrock")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------- Boxplot of final scores ----------------
    plt.figure(figsize=(9,6))
    data_to_plot = [all_final_scores[v] for v in variants_to_run]
    plt.boxplot(data_to_plot, labels=variants_to_run)
    plt.yscale('log')
    plt.title("Final fitness distribution (Rosenbrock)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # ---------------- If we have weight history, plot it ----------------
    if best_weight_history is not None:
        plt.figure(figsize=(10,4))
        iters = np.arange(len(best_weight_history['w1']))
        plt.plot(iters, best_weight_history['w1'], label='w1 (Alpha)', lw=2)
        plt.plot(iters, best_weight_history['w2'], label='w2 (Beta)',  lw=2)
        plt.plot(iters, best_weight_history['w3'], label='w3 (Delta)', lw=2)
        plt.xlabel("Iteration")
        plt.ylabel("Weight (blended)")
        plt.title("Weight evolution during best weighted_dynamic run")
        plt.legend()
        plt.grid(True)
        plt.show()