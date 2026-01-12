"""
Two-Step Iterative Least Squares (TSILS) cho bài toán JCAS
Converted từ MATLAB code gốc
"""
import numpy as np
from scipy.signal import find_peaks

class TSILS_JCAS:
    """
    Two-Step Iterative Least Squares algorithm for JCAS beamforming
    """
    def __init__(self, jcas_problem, max_iter=50, convergence_tol=1e-3, seed=42):
        """
        Parameters:
        -----------
        jcas_problem : JCASProblem instance
        max_iter : số vòng lặp tối đa
        convergence_tol : ngưỡng hội tụ
        """
        np.random.seed(seed)
        self.problem = jcas_problem
        self.M = jcas_problem.M
        self.A_q = jcas_problem.A_q
        self.eq_dirs = jcas_problem.eq_dirs
        self.desired_dir = jcas_problem.desired_dir
        self.max_iter = max_iter
        self.convergence_tol = convergence_tol
        
        self.convergence_curve = np.zeros(max_iter)
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def find_nulls(self, pattern, n_nulls=2):
        """
        Tìm các điểm null (cực tiểu) gần main lobe nhất
        
        Parameters:
        -----------
        pattern : array
            Pattern magnitude
        n_nulls : int
            Số lượng nulls cần tìm (mặc định 2: trái và phải main lobe)
            
        Returns:
        --------
        null_indices : array
            Indices của các nulls
        """
        # Flip pattern để tìm peaks (nulls là valleys)
        flipped = -pattern
        peaks, _ = find_peaks(flipped)
        
        if len(peaks) == 0:
            # Fallback: trả về 2 đầu
            return np.array([0, len(pattern)-1])
        
        # Tìm main lobe position
        main_lobe_idx = np.argmax(pattern)
        
        # Sort peaks theo khoảng cách đến main lobe
        distances = np.abs(peaks - main_lobe_idx)
        sorted_indices = np.argsort(distances)
        
        # Lấy n_nulls gần nhất
        closest_nulls = peaks[sorted_indices[:min(n_nulls, len(peaks))]]
        
        return np.sort(closest_nulls)
    
    def generate_desired_pattern(self):
        """
        Tạo desired pattern magnitude bằng Capon's beamforming initialization
        
        Returns:
        --------
        PdM : array
            Desired pattern magnitude (sidelobes = 0)
        W0 : array
            Initial beamforming weights (complex)
        """
        # Find index corresponding to desired direction
        desired_idx = np.argmin(np.abs(self.eq_dirs - np.sin(self.desired_dir)))
        
        # Capon's beamforming initialization
        # R_uu = a(θ) * a^H(θ) + noise
        a_desired = self.A_q[:, desired_idx].reshape(-1, 1)
        noise_power = (np.random.rand() / 1000) ** 2
        R_uu = a_desired @ a_desired.conj().T + noise_power * np.eye(self.M)
        
        # Capon's weight: w = R^{-1} a / (a^H R^{-1} a)
        R_inv = np.linalg.inv(R_uu)
        R_inv_a = R_inv @ a_desired
        denominator = a_desired.conj().T @ R_inv_a
        W0 = R_inv_a / denominator
        W0 = W0.flatten()
        
        # Generate reference pattern
        P_ref = np.abs(W0.conj().T @ self.A_q)
        
        # Find nulls (2 closest to main lobe)
        null_indices = self.find_nulls(P_ref, n_nulls=2)
        
        # Create desired pattern (zero out sidelobes)
        PdM = P_ref.copy()
        PdM[:null_indices[0]] = 0
        PdM[null_indices[1]:] = 0
        
        return PdM, W0
    
    def two_step_ILS(self, W0, PdM):
        """
        Two-Step Iterative Least Squares optimization
        
        Parameters:
        -----------
        W0 : array (complex)
            Initial beamforming weights
        PdM : array
            Desired pattern magnitude
            
        Returns:
        --------
        W_opt : array (complex)
            Optimized beamforming weights
        """
        # Indices to approximate (sample every 4th direction + non-zero desired)
        alpha = np.sort(np.unique(np.concatenate([
            np.arange(0, len(self.eq_dirs), 4),
            np.where(PdM > 0)[0]
        ])))
        
        # Current pattern magnitude
        PM = np.abs(W0.conj().T @ self.A_q)
        
        W_current = W0.copy()
        
        for iteration in range(self.max_iter):
            # Step 2: Extract V matrix (sampled steering vectors)
            V = self.A_q[:, alpha]
            
            # Step 3: Check convergence
            diff_total = np.sum(np.abs(PM[alpha] - PdM[alpha]))
            
            if iteration % 10 == 0:
                print(f"  TSILS Iteration {iteration}/{self.max_iter}, "
                      f"Total difference = {diff_total:.6f}")
            
            # Tính fitness và lưu lịch sử
            # Encode W_current thành vector x để đánh giá
            x_current = np.concatenate([W_current.real, W_current.imag])
            fitness = self.problem.evaluate(x_current)
            self.convergence_curve[iteration] = fitness
            
            if diff_total < self.convergence_tol:
                print(f"  TSILS converged at iteration {iteration}")
                # Fill remaining convergence curve
                self.convergence_curve[iteration:] = fitness
                break
            
            # Step 4: Iterative Least-Squares update
            inner_iter = 0
            max_inner_iter = 100
            
            while inner_iter < max_inner_iter:
                inner_iter += 1
                
                # PdP = W^H * V * (diag(PdM))^{-1}
                # Use pseudoinverse for stability
                PdM_alpha = PdM[alpha]
                PdM_diag_inv = np.diag(1.0 / (PdM_alpha + 1e-10))
                
                PdP = W_current.conj().T @ V @ PdM_diag_inv
                
                # Normalize
                PdP0 = PdP / (np.max(np.abs(PdP)) + 1e-10)
                
                # Update weights: W1 = (V*V^H)^{-1} * V * diag(PdM) * PdP0^H
                VVH = V @ V.conj().T
                VVH_inv = np.linalg.inv(VVH + 1e-6 * np.eye(self.M))  # Regularization
                
                W1 = VVH_inv @ V @ np.diag(PdM_alpha) @ PdP0.conj().T
                W1 = W1.flatten()
                
                # Check inner convergence
                if np.linalg.norm(W1 - W_current) ** 2 < 1e-4:
                    break
                else:
                    W_current = W1
            
            # Update W and PM
            W_current = W1
            PM = np.abs(W_current.conj().T @ self.A_q)
        
        return W_current
    
    def optimize(self):
        """
        Main optimization routine
        
        Returns:
        --------
        best_solution : array
            Best solution encoded as [real; imag]
        best_fitness : float
            Best fitness value
        convergence_curve : array
            Fitness history
        """
        print(f"  Initializing TSILS with Capon's beamforming...")
        
        # Step 1: Generate desired pattern and initial weights
        PdM, W0 = self.generate_desired_pattern()
        
        print(f"  Running Two-Step ILS optimization...")
        
        # Step 2-4: Iterative optimization
        W_opt = self.two_step_ILS(W0, PdM)
        
        # Encode result
        self.best_solution = np.concatenate([W_opt.real, W_opt.imag])
        self.best_fitness = self.problem.evaluate(self.best_solution)
        
        print(f"  TSILS final fitness: {self.best_fitness:.6f}")
        
        return self.best_solution, self.best_fitness, self.convergence_curve
