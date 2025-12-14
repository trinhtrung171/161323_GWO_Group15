import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# HÀM GWO 
# =============================================================================
def GWO(fobj, lb, ub, dim, SearchAgents_no, Max_iter, variant='original', record_weights=False):
    # Khởi tạo
    Alpha_pos = np.zeros(dim); Alpha_score = float('inf')
    Beta_pos  = np.zeros(dim); Beta_score  = float('inf')
    Delta_pos = np.zeros(dim); Delta_score = float('inf')

    if not isinstance(lb, (list, np.ndarray)): lb = np.full(dim, lb)
    if not isinstance(ub, (list, np.ndarray)): ub = np.full(dim, ub)

    Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    Convergence_curve = np.zeros(Max_iter)
    epsilon = 1e-12

    # Lưu lịch sử trọng số (chỉ dùng cho weighted_dynamic)
    weight_history = {'w1': [], 'w2': [], 'w3': []}

    for t in range(Max_iter):
        for i in range(SearchAgents_no):
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            fitness = fobj(Positions[i, :]) # Hàm fobj bây giờ sẽ giải mã TSP

            if fitness < Alpha_score:
                Alpha_score, Beta_score, Delta_score = fitness, Alpha_score, Beta_score
                Alpha_pos, Beta_pos, Delta_pos = Positions[i, :].copy(), Alpha_pos.copy(), Beta_pos.copy()
            elif fitness < Beta_score:
                Beta_score, Delta_score = fitness, Beta_score
                Beta_pos, Delta_pos = Positions[i, :].copy(), Beta_pos.copy()
            elif fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()

        # a giảm chậm để thăm dò lâu hơn
        frac = t / (Max_iter - 1)
        a = 2 * (1 - frac**0.6)

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

        if variant == 'weighted_static':
            w1, w2, w3 = 0.6, 0.3, 0.1
            new_Positions = w1 * X1 + w2 * X2 + w3 * X3

        elif variant == 'weighted_dynamic':
            # Logic Dynamic 4.1
            f_min = min(Alpha_score, Beta_score, Delta_score)
            f_max = max(Alpha_score, Beta_score, Delta_score)

            if abs(f_max - f_min) < 1e-14:
                gap_alpha, gap_beta, gap_delta = 1.0, 0.7, 0.3
            else:
                gap_alpha = (f_max - Alpha_score) / (f_max - f_min + epsilon)
                gap_beta  = (f_max - Beta_score)  / (f_max - f_min + epsilon)
                gap_delta = (f_max - Delta_score) / (f_max - f_min + epsilon)

            power = 2.8
            w1_raw = gap_alpha ** power
            w2_raw = gap_beta  ** power
            w3_raw = gap_delta ** power

            S = w1_raw + w2_raw + w3_raw + epsilon
            w1_raw /= S; w2_raw /= S; w3_raw /= S

            exploit_factor = (1 - a/2) ** 3
            explore_factor = 1 - exploit_factor

            w1 = w1_raw * exploit_factor + (1/3) * explore_factor
            w2 = w2_raw * exploit_factor + (1/3) * explore_factor
            w3 = w3_raw * exploit_factor + (1/3) * explore_factor

            jitter = 1e-4
            w1 += np.random.uniform(-jitter, jitter)
            w2 += np.random.uniform(-jitter, jitter)
            w3 += np.random.uniform(-jitter, jitter)

            Wsum = w1 + w2 + w3 + epsilon
            w1 /= Wsum; w2 /= Wsum; w3 /= Wsum

            # Ghi lại lịch sử trọng số
            if record_weights:
                weight_history['w1'].append(w1)
                weight_history['w2'].append(w2)
                weight_history['w3'].append(w3)

            new_Positions = w1 * X1 + w2 * X2 + w3 * X3
        else:
            new_Positions = (X1 + X2 + X3) / 3.0

        Positions = np.clip(new_Positions, lb, ub)
        Convergence_curve[t] = Alpha_score

    if record_weights and variant == 'weighted_dynamic':
        return Alpha_score, Alpha_pos, Convergence_curve, weight_history
    return Alpha_score, Alpha_pos, Convergence_curve

# =============================================================================
# HÀM MỤC TIÊU: TSP (MÃ HÓA KHÓA NGẪU NHIÊN)
# =============================================================================

# Tạo bản đồ thành phố
NUM_CITIES = 20
np.random.seed(42) # Cố định bản đồ để công bằng
city_coords = np.random.rand(NUM_CITIES, 2) * 100 # Tọa độ x,y ngẫu nhiên

def calculate_distance(path_indices):
    """Tính tổng quãng đường của một lộ trình"""
    dist = 0
    for i in range(len(path_indices) - 1):
        c1 = city_coords[path_indices[i]]
        c2 = city_coords[path_indices[i+1]]
        dist += np.sqrt(np.sum((c1 - c2)**2))
    # Quay về điểm đầu
    c_last = city_coords[path_indices[-1]]
    c_first = city_coords[path_indices[0]]
    dist += np.sqrt(np.sum((c_last - c_first)**2))
    return dist

def fobj_tsp(x):
    """
    x: Vector số thực (độ ưu tiên)
    return: Tổng quãng đường
    """
    # 1. Giải mã: Sắp xếp x để lấy thứ tự thành phố
    # argsort trả về chỉ số của các phần tử sau khi sắp xếp
    path_indices = np.argsort(x)
    
    # 2. Tính quãng đường
    return calculate_distance(path_indices)

# =============================================================================
# CHẠY MÔ PHỎNG
# =============================================================================
if __name__ == "__main__":
    SearchAgents_no = 50
    Max_iter = 500
    dim = NUM_CITIES # Số chiều = Số thành phố
    lb = 0
    ub = 1 
    
    variants = ['original', 'weighted_static', 'weighted_dynamic']
    results = {}
    best_dynamic_weights = None # Để lưu lịch sử trọng số của lần chạy tốt nhất

    print(f"Đang giải bài toán TSP với {NUM_CITIES} thành phố...")

    # Chạy từng biến thể
    for v in variants:
        print(f"  Running {v}...")
        # Reset seed ngẫu nhiên cho thuật toán (nhưng bản đồ thành phố giữ nguyên)
        np.random.seed(10) 
        
        if v == 'weighted_dynamic':
            score, pos, curve, w_hist = GWO(fobj_tsp, lb, ub, dim, SearchAgents_no, Max_iter, variant=v, record_weights=True)
            best_dynamic_weights = w_hist
        else:
            score, pos, curve = GWO(fobj_tsp, lb, ub, dim, SearchAgents_no, Max_iter, variant=v, record_weights=False)
        
        # Lưu kết quả
        best_path = np.argsort(pos)
        results[v] = {'score': score, 'path': best_path, 'curve': curve}
        print(f"    -> Quãng đường ngắn nhất: {score:.2f}")

# --- VẼ KẾT QUẢ (TÁCH THÀNH 3 HÌNH RIÊNG) ---

    # HÌNH 1: TỐC ĐỘ HỘI TỤ
    plt.figure(figsize=(10, 6))
    for v in variants:
        plt.plot(results[v]['curve'], label=f"{v} (Best: {results[v]['score']:.1f})", lw=2)
    plt.title("1. Tốc độ hội tụ (Càng thấp càng tốt)", fontsize=14)
    plt.ylabel("Tổng quãng đường")
    plt.xlabel("Vòng lặp")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() # Hiển thị hình 1

    # HÌNH 2: BIẾN ĐỘNG TRỌNG SỐ (Chỉ Weighted Dynamic)
    if best_dynamic_weights:
        plt.figure(figsize=(12, 5))
        iters = range(len(best_dynamic_weights['w1']))
        plt.plot(iters, best_dynamic_weights['w1'], label='Alpha (w1)', color='green', lw=1.5)
        plt.plot(iters, best_dynamic_weights['w2'], label='Beta (w2)', color='orange', lw=1.5, alpha=0.7)
        plt.plot(iters, best_dynamic_weights['w3'], label='Delta (w3)', color='blue', lw=1.5, alpha=0.7)
        
        plt.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='Mức cân bằng (1/3)')
        
        plt.title("2. Biến động Trọng số trong Weighted Dynamic (Chiến thuật Thích ứng)", fontsize=14)
        plt.ylabel("Giá trị Trọng số")
        plt.xlabel("Vòng lặp")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show() # Hiển thị hình 2

    # HÌNH 3: LỘ TRÌNH CHI TIẾT
    fig3, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, v in enumerate(variants):
        ax = axes[i]
        path_indices = results[v]['path']
        score = results[v]['score']
        
        # Vẽ thành phố
        ax.scatter(city_coords[:, 0], city_coords[:, 1], c='red', s=50, zorder=2)
        for idx, (x, y) in enumerate(city_coords):
            ax.text(x, y+3, str(idx), fontsize=9, ha='center')
            
        # Vẽ đường đi
        path_x = []
        path_y = []
        for idx in path_indices:
            path_x.append(city_coords[idx, 0])
            path_y.append(city_coords[idx, 1])
        path_x.append(city_coords[path_indices[0], 0])
        path_y.append(city_coords[path_indices[0], 1])
        
        ax.plot(path_x, path_y, 'b-', alpha=0.7, lw=2, zorder=1)
        ax.set_title(f"3.{i+1} Lộ trình: {v}\nDist: {score:.2f}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show() # Hiển thị hình 3