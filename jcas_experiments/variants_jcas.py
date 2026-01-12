"""
Các biến thể cải tiến của GWO cho bài toán JCAS:
1. Weighted Dynamic GWO (WDGWO)
2. Chaotic GWO (CGWO)
"""
import numpy as np
from jcas_problem import JCASProblem

class WeightedDynamic_GWO_JCAS:
    """GWO với trọng số động thích ứng theo fitness"""
    def __init__(self, jcas_problem: JCASProblem, n_wolves=30, max_iter=200, seed=42):
        np.random.seed(seed)
        self.problem = jcas_problem
        self.dim = jcas_problem.get_dimension()
        self.lb, self.ub = jcas_problem.get_bounds()
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        
        self.convergence_curve = np.zeros(max_iter)
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def initialize_population(self):
        positions = np.zeros((self.n_wolves, self.dim))
        for i in range(self.n_wolves):
            for d in range(self.dim):
                positions[i, d] = self.lb[d] + np.random.rand() * (self.ub[d] - self.lb[d])
        return positions
    
    def compute_dynamic_weights(self, Alpha_score, Beta_score, Delta_score, all_fitness):
        """Tính trọng số động dựa trên khoảng cách fitness"""
        f_max = np.max(all_fitness)
        f_min = np.min(all_fitness)
        epsilon = 1e-10
        
        # Khoảng cách chuẩn hóa
        gap_alpha = (f_max - Alpha_score) / (f_max - f_min + epsilon)
        gap_beta = (f_max - Beta_score) / (f_max - f_min + epsilon)
        gap_delta = (f_max - Delta_score) / (f_max - f_min + epsilon)
        
        # Áp dụng hàm mũ để tăng phân biệt
        power = 2.8
        w1_raw = gap_alpha ** power
        w2_raw = gap_beta ** power
        w3_raw = gap_delta ** power
        
        # Chuẩn hóa về tổng = 1
        total = w1_raw + w2_raw + w3_raw + epsilon
        w1 = w1_raw / total
        w2 = w2_raw / total
        w3 = w3_raw / total
        
        return w1, w2, w3
    
    def optimize(self):
        positions = self.initialize_population()
        fitness = np.array([self.problem.evaluate(pos) for pos in positions])
        
        sorted_indices = np.argsort(fitness)
        Alpha_pos = positions[sorted_indices[0]].copy()
        Alpha_score = fitness[sorted_indices[0]]
        Beta_pos = positions[sorted_indices[1]].copy()
        Beta_score = fitness[sorted_indices[1]]
        Delta_pos = positions[sorted_indices[2]].copy()
        Delta_score = fitness[sorted_indices[2]]
        
        for t in range(self.max_iter):
            a = 2 - t * (2.0 / self.max_iter)
            
            # Tính trọng số động
            w1, w2, w3 = self.compute_dynamic_weights(Alpha_score, Beta_score, Delta_score, fitness)
            
            for i in range(self.n_wolves):
                for d in range(self.dim):
                    # Alpha
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * Alpha_pos[d] - positions[i, d])
                    X1 = Alpha_pos[d] - A1 * D_alpha
                    
                    # Beta
                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * Beta_pos[d] - positions[i, d])
                    X2 = Beta_pos[d] - A2 * D_beta
                    
                    # Delta
                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * Delta_pos[d] - positions[i, d])
                    X3 = Delta_pos[d] - A3 * D_delta
                    
                    # Cập nhật với trọng số động
                    positions[i, d] = w1 * X1 + w2 * X2 + w3 * X3
                
                positions[i] = np.clip(positions[i], self.lb, self.ub)
            
            fitness = np.array([self.problem.evaluate(pos) for pos in positions])
            
            for i in range(self.n_wolves):
                if fitness[i] < Alpha_score:
                    Delta_score, Delta_pos = Beta_score, Beta_pos.copy()
                    Beta_score, Beta_pos = Alpha_score, Alpha_pos.copy()
                    Alpha_score, Alpha_pos = fitness[i], positions[i].copy()
                elif fitness[i] < Beta_score:
                    Delta_score, Delta_pos = Beta_score, Beta_pos.copy()
                    Beta_score, Beta_pos = fitness[i], positions[i].copy()
                elif fitness[i] < Delta_score:
                    Delta_score, Delta_pos = fitness[i], positions[i].copy()
            
            self.convergence_curve[t] = Alpha_score
            
            if (t + 1) % 20 == 0:
                print(f"  [WDGWO] Iter {t+1}/{self.max_iter}: Best = {Alpha_score:.6f}, Weights = ({w1:.3f}, {w2:.3f}, {w3:.3f})")
        
        self.best_solution = Alpha_pos
        self.best_fitness = Alpha_score
        return Alpha_pos, Alpha_score, self.convergence_curve


class Chaotic_GWO_JCAS:
    """GWO với ánh xạ hỗn loạn (Chaotic map) cho thăm dò tốt hơn"""
    def __init__(self, jcas_problem: JCASProblem, n_wolves=30, max_iter=200, seed=42, chaos_type='tent'):
        np.random.seed(seed)
        self.problem = jcas_problem
        self.dim = jcas_problem.get_dimension()
        self.lb, self.ub = jcas_problem.get_bounds()
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.chaos_type = chaos_type
        
        self.convergence_curve = np.zeros(max_iter)
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # Khởi tạo biến hỗn loạn
        self.chaos_value = np.random.rand()
    
    def chaotic_map(self, x):
        """Ánh xạ hỗn loạn Tent Map"""
        if self.chaos_type == 'tent':
            return 2 * x if x < 0.5 else 2 * (1 - x)
        elif self.chaos_type == 'logistic':
            mu = 4.0  # Tham số hỗn loạn cực đại
            return mu * x * (1 - x)
        else:
            return x
    
    def initialize_population(self):
        positions = np.zeros((self.n_wolves, self.dim))
        for i in range(self.n_wolves):
            for d in range(self.dim):
                positions[i, d] = self.lb[d] + np.random.rand() * (self.ub[d] - self.lb[d])
        return positions
    
    def optimize(self):
        positions = self.initialize_population()
        fitness = np.array([self.problem.evaluate(pos) for pos in positions])
        
        sorted_indices = np.argsort(fitness)
        Alpha_pos = positions[sorted_indices[0]].copy()
        Alpha_score = fitness[sorted_indices[0]]
        Beta_pos = positions[sorted_indices[1]].copy()
        Beta_score = fitness[sorted_indices[1]]
        Delta_pos = positions[sorted_indices[2]].copy()
        Delta_score = fitness[sorted_indices[2]]
        
        for t in range(self.max_iter):
            a = 2 - t * (2.0 / self.max_iter)
            
            # Cập nhật giá trị hỗn loạn
            self.chaos_value = self.chaotic_map(self.chaos_value)
            
            for i in range(self.n_wolves):
                for d in range(self.dim):
                    # Sử dụng chaos thay cho random
                    self.chaos_value = self.chaotic_map(self.chaos_value)
                    r1 = self.chaos_value
                    self.chaos_value = self.chaotic_map(self.chaos_value)
                    r2 = self.chaos_value
                    
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * Alpha_pos[d] - positions[i, d])
                    X1 = Alpha_pos[d] - A1 * D_alpha
                    
                    self.chaos_value = self.chaotic_map(self.chaos_value)
                    r1 = self.chaos_value
                    self.chaos_value = self.chaotic_map(self.chaos_value)
                    r2 = self.chaos_value
                    
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * Beta_pos[d] - positions[i, d])
                    X2 = Beta_pos[d] - A2 * D_beta
                    
                    self.chaos_value = self.chaotic_map(self.chaos_value)
                    r1 = self.chaos_value
                    self.chaos_value = self.chaotic_map(self.chaos_value)
                    r2 = self.chaos_value
                    
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * Delta_pos[d] - positions[i, d])
                    X3 = Delta_pos[d] - A3 * D_delta
                    
                    positions[i, d] = (X1 + X2 + X3) / 3.0
                
                positions[i] = np.clip(positions[i], self.lb, self.ub)
            
            fitness = np.array([self.problem.evaluate(pos) for pos in positions])
            
            for i in range(self.n_wolves):
                if fitness[i] < Alpha_score:
                    Delta_score, Delta_pos = Beta_score, Beta_pos.copy()
                    Beta_score, Beta_pos = Alpha_score, Alpha_pos.copy()
                    Alpha_score, Alpha_pos = fitness[i], positions[i].copy()
                elif fitness[i] < Beta_score:
                    Delta_score, Delta_pos = Beta_score, Beta_pos.copy()
                    Beta_score, Beta_pos = fitness[i], positions[i].copy()
                elif fitness[i] < Delta_score:
                    Delta_score, Delta_pos = fitness[i], positions[i].copy()
            
            self.convergence_curve[t] = Alpha_score
            
            if (t + 1) % 20 == 0:
                print(f"  [CGWO] Iter {t+1}/{self.max_iter}: Best = {Alpha_score:.6f}")
        
        self.best_solution = Alpha_pos
        self.best_fitness = Alpha_score
        return Alpha_pos, Alpha_score, self.convergence_curve
