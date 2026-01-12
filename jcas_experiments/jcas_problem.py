"""
JCAS Problem Wrapper for GWO Optimization
Bài toán tối ưu hóa multibeam cho Joint Communication and Sensing (JCAS)

Mục tiêu: Tối ưu hóa beamforming weights để:
- Maximizing communication beam gain ở hướng mong muốn
- Minimizing interference/sidelobe levels
- Balancing communication-sensing trade-off
"""

import numpy as np

class JCASProblem:
    """
    Bài toán JCAS Multibeam Optimization
    
    Tham số tối ưu: Beamforming weights (complex values)
    Biến quyết định: Real & Imaginary parts của weights
    """
    
    def __init__(self, M=12, Q=160, phi=1, desired_dir=0.0, ro=0.5):
        """
        Parameters:
        -----------
        M : int
            Số lượng phần tử anten (antenna elements)
        Q : int
            Độ phân giải của equivalent directions
        phi : float
            Quantization step
        desired_dir : float
            Hướng mong muốn cho communication beam (rad)
        ro : float
            Communication-sensing trade-off parameter (0-1)
        """
        self.M = M
        self.Q = Q
        self.phi = phi
        self.desired_dir = desired_dir
        self.ro = ro
        self.lambda_wave = 1.0  # Wavelength
        
        # Equivalent directions
        self.eq_dirs = np.arange(-1, 1, phi/Q)
        
        # Generate steering matrix (Array response)
        self.A_q = self._generate_steering_matrix()
        
        # Dimension: 2*M (real + imaginary parts)
        self.dim = 2 * M
        
        # Bounds: normalized weights [-1, 1]
        self.lower_bounds = np.full(self.dim, -1.0)
        self.upper_bounds = np.full(self.dim, 1.0)
        
        print(f"JCAS Problem initialized:")
        print(f"  - Antenna elements (M): {M}")
        print(f"  - Equivalent directions: {len(self.eq_dirs)}")
        print(f"  - Optimization dimension: {self.dim}")
        print(f"  - Desired direction: {desired_dir:.2f} rad")
        print(f"  - Comm-Sensing tradeoff (ro): {ro}")
    
    def _generate_steering_matrix(self):
        """
        Tạo steering matrix A_q cho equivalent directions
        A_q[m, n] = exp(-j * 2π * m * eq_dir[n])
        """
        m_indices = np.arange(self.M).reshape(-1, 1)  # (M, 1)
        eq_dirs_arr = self.eq_dirs.reshape(1, -1)     # (1, N)
        
        # Steering matrix
        A_q = np.exp(-1j * 2 * np.pi * m_indices * eq_dirs_arr)
        return A_q
    
    def decode_weights(self, x):
        """
        Giải mã từ vector tối ưu x sang complex weights
        x[:M] = real parts, x[M:] = imaginary parts
        """
        w_real = x[:self.M]
        w_imag = x[self.M:]
        w = w_real + 1j * w_imag
        
        # Normalize
        w = w / (np.linalg.norm(w) + 1e-10)
        return w
    
    def evaluate(self, x):
        """
        Hàm mục tiêu: Minimize fitness
        
        Fitness bao gồm:
        1. Maximize main lobe gain tại desired direction
        2. Minimize sidelobe levels
        3. Minimize deviation từ desired pattern
        
        Returns:
        --------
        fitness : float
            Giá trị fitness (càng nhỏ càng tốt)
        """
        # Decode beamforming weights
        w = self.decode_weights(x)
        
        # Array response pattern: P = |w^H * A_q|^2
        response = np.abs(w.conj().T @ self.A_q) ** 2
        response_db = 10 * np.log10(response / np.max(response) + 1e-10)
        
        # Find index of desired direction
        desired_idx = np.argmin(np.abs(self.eq_dirs - np.sin(self.desired_dir)))
        
        # Component 1: Maximize main lobe gain (negate for minimization)
        main_lobe_gain = -response_db[desired_idx]
        
        # Component 2: Minimize sidelobe level
        # Exclude main lobe region (±0.1 around desired direction)
        mainlobe_mask = np.abs(self.eq_dirs - self.eq_dirs[desired_idx]) > 0.1
        sidelobe_level = np.max(response_db[mainlobe_mask])
        
        # Component 3: Penalize high average sidelobe
        avg_sidelobe = np.mean(response_db[mainlobe_mask])
        
        # Component 4: Pattern uniformity (variance penalty)
        pattern_variance = np.std(response_db)
        
        # Combined fitness (weighted sum)
        fitness = (
            1.0 * main_lobe_gain +      # Main lobe gain (negative, want high)
            0.5 * sidelobe_level +       # Max sidelobe (want low)
            0.3 * avg_sidelobe +         # Avg sidelobe (want low)
            0.2 * pattern_variance       # Pattern variance (want controlled)
        )
        
        return fitness
    
    def evaluate_metrics(self, x):
        """
        Đánh giá các metrics chi tiết cho phân tích
        """
        w = self.decode_weights(x)
        response = np.abs(w.conj().T @ self.A_q) ** 2
        response_db = 10 * np.log10(response / np.max(response) + 1e-10)
        
        desired_idx = np.argmin(np.abs(self.eq_dirs - np.sin(self.desired_dir)))
        mainlobe_mask = np.abs(self.eq_dirs - self.eq_dirs[desired_idx]) > 0.1
        
        metrics = {
            'main_lobe_gain_db': response_db[desired_idx],
            'max_sidelobe_db': np.max(response_db[mainlobe_mask]),
            'avg_sidelobe_db': np.mean(response_db[mainlobe_mask]),
            'pattern_std': np.std(response_db),
            'fitness': self.evaluate(x)
        }
        
        return metrics, response_db
    
    def get_dimension(self):
        """Trả về số chiều của bài toán"""
        return self.dim
    
    def get_bounds(self):
        """Trả về bounds của biến quyết định"""
        return self.lower_bounds, self.upper_bounds
