"""
GWO nguyên bản áp dụng cho bài toán JCAS Multibeam Optimization
"""
import numpy as np
from jcas_problem import JCASProblem

class GWO_JCAS:
    def __init__(self, jcas_problem: JCASProblem, n_wolves=30, max_iter=200, seed=42):
        """
        Parameters:
        -----------
        jcas_problem : JCASProblem instance
        n_wolves : số lượng sói trong bầy
        max_iter : số vòng lặp tối đa
        """
        np.random.seed(seed)
        self.problem = jcas_problem
        self.dim = jcas_problem.get_dimension()
        self.lb, self.ub = jcas_problem.get_bounds()
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        
        # Lưu lịch sử
        self.convergence_curve = np.zeros(max_iter)
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def initialize_population(self):
        """Khởi tạo quần thể ngẫu nhiên trong không gian tìm kiếm"""
        positions = np.zeros((self.n_wolves, self.dim))
        for i in range(self.n_wolves):
            for d in range(self.dim):
                positions[i, d] = self.lb[d] + np.random.rand() * (self.ub[d] - self.lb[d])
        return positions
    
    def optimize(self):
        """Thuật toán GWO nguyên bản"""
        # Khởi tạo quần thể
        positions = self.initialize_population()
        
        # Đánh giá fitness ban đầu
        fitness = np.array([self.problem.evaluate(pos) for pos in positions])
        
        # Khởi tạo Alpha, Beta, Delta
        sorted_indices = np.argsort(fitness)
        Alpha_pos = positions[sorted_indices[0]].copy()
        Alpha_score = fitness[sorted_indices[0]]
        Beta_pos = positions[sorted_indices[1]].copy()
        Beta_score = fitness[sorted_indices[1]]
        Delta_pos = positions[sorted_indices[2]].copy()
        Delta_score = fitness[sorted_indices[2]]
        
        # Vòng lặp chính
        for t in range(self.max_iter):
            # Tuyến tính giảm a từ 2 về 0
            a = 2 - t * (2.0 / self.max_iter)
            
            # Cập nhật vị trí mỗi con sói
            for i in range(self.n_wolves):
                for d in range(self.dim):
                    # Tính A và C
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    
                    # Khoảng cách đến Alpha
                    D_alpha = abs(C1 * Alpha_pos[d] - positions[i, d])
                    X1 = Alpha_pos[d] - A1 * D_alpha
                    
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    
                    # Khoảng cách đến Beta
                    D_beta = abs(C2 * Beta_pos[d] - positions[i, d])
                    X2 = Beta_pos[d] - A2 * D_beta
                    
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    
                    # Khoảng cách đến Delta
                    D_delta = abs(C3 * Delta_pos[d] - positions[i, d])
                    X3 = Delta_pos[d] - A3 * D_delta
                    
                    # Vị trí mới (trung bình của 3 vị trí)
                    positions[i, d] = (X1 + X2 + X3) / 3.0
                
                # Đảm bảo trong biên
                positions[i] = np.clip(positions[i], self.lb, self.ub)
            
            # Đánh giá lại fitness
            fitness = np.array([self.problem.evaluate(pos) for pos in positions])
            
            # Cập nhật Alpha, Beta, Delta
            for i in range(self.n_wolves):
                if fitness[i] < Alpha_score:
                    Delta_score = Beta_score
                    Delta_pos = Beta_pos.copy()
                    Beta_score = Alpha_score
                    Beta_pos = Alpha_pos.copy()
                    Alpha_score = fitness[i]
                    Alpha_pos = positions[i].copy()
                elif fitness[i] < Beta_score:
                    Delta_score = Beta_score
                    Delta_pos = Beta_pos.copy()
                    Beta_score = fitness[i]
                    Beta_pos = positions[i].copy()
                elif fitness[i] < Delta_score:
                    Delta_score = fitness[i]
                    Delta_pos = positions[i].copy()
            
            # Lưu lịch sử hội tụ
            self.convergence_curve[t] = Alpha_score
            
            # In tiến trình
            if (t + 1) % 20 == 0:
                print(f"  Iteration {t+1}/{self.max_iter}: Best Fitness = {Alpha_score:.6f}")
        
        self.best_solution = Alpha_pos
        self.best_fitness = Alpha_score
        
        return Alpha_pos, Alpha_score, self.convergence_curve
