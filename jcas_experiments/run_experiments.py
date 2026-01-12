"""
Script chạy thí nghiệm so sánh các thuật toán trên bài toán JCAS
Phần 3-6 của báo cáo
Bao gồm: TSILS (baseline), GWO, WDGWO, CGWO
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

from jcas_problem import JCASProblem
from gwo_jcas import GWO_JCAS
from variants_jcas import WeightedDynamic_GWO_JCAS, Chaotic_GWO_JCAS
from tsils_jcas import TSILS_JCAS

def run_single_trial(algorithm_class, problem, seed, **kwargs):
    """Chạy một lần thử nghiệm"""
    algo = algorithm_class(problem, seed=seed, **kwargs)
    best_sol, best_fit, curve = algo.optimize()
    return best_sol, best_fit, curve

def run_experiments(n_trials=10, n_wolves=30, max_iter=200):
    """Chạy thí nghiệm với nhiều lần thử"""
    
    # Khởi tạo bài toán JCAS
    print("=" * 70)
    print("KHỞI TẠO BÀI TOÁN JCAS MULTIBEAM OPTIMIZATION")
    print("=" * 70)
    problem = JCASProblem(M=12, Q=160, phi=1, desired_dir=0.0, ro=0.5)
    
    algorithms = {
        'TSILS': TSILS_JCAS,  # Baseline method (không dùng n_wolves)
        'GWO': GWO_JCAS,
        'WDGWO': WeightedDynamic_GWO_JCAS,
        'CGWO': Chaotic_GWO_JCAS
    }
    
    results = {}
    convergence_data = {}
    best_solutions = {}
    
    for alg_name, alg_class in algorithms.items():
        print(f"\n{'=' * 70}")
        print(f"ĐANG CHẠY {alg_name} - {n_trials} TRIALS")
        print(f"{'=' * 70}")
        
        best_fitness_list = []
        all_curves = []
        all_solutions = []
        
        for trial in range(n_trials):
            print(f"\n--- Trial {trial + 1}/{n_trials} ---")
            
            # TSILS không cần n_wolves parameter
            if alg_name == 'TSILS':
                best_sol, best_fit, curve = run_single_trial(
                    alg_class, problem, seed=trial, max_iter=max_iter
                )
            else:
                best_sol, best_fit, curve = run_single_trial(
                    alg_class, problem, seed=trial, 
                    n_wolves=n_wolves, max_iter=max_iter
                )
            
            best_fitness_list.append(best_fit)
            all_curves.append(curve)
            all_solutions.append(best_sol)
        
        # Lấy solution tốt nhất
        best_trial_idx = np.argmin(best_fitness_list)
        best_solutions[alg_name] = all_solutions[best_trial_idx]
        
        # Thống kê
        results[alg_name] = {
            'best': np.min(best_fitness_list),
            'worst': np.max(best_fitness_list),
            'mean': np.mean(best_fitness_list),
            'std': np.std(best_fitness_list),
            'median': np.median(best_fitness_list)
        }
        convergence_data[alg_name] = np.array(all_curves)
        
        print(f"\n[{alg_name}] KẾT QUẢ SAU {n_trials} TRIALS:")
        print(f"  Best:   {results[alg_name]['best']:.6f}")
        print(f"  Worst:  {results[alg_name]['worst']:.6f}")
        print(f"  Mean:   {results[alg_name]['mean']:.6f}")
        print(f"  Std:    {results[alg_name]['std']:.6f}")
        print(f"  Median: {results[alg_name]['median']:.6f}")
    
    # Lưu kết quả
    os.makedirs('results', exist_ok=True)
    
    # 1. Bảng kết quả
    df = pd.DataFrame(results).T
    df.to_csv('results/comparison_table.csv')
    print(f"\n✓ Đã lưu bảng kết quả: results/comparison_table.csv")
    
    # In bảng đẹp
    print("\n" + "=" * 70)
    print("BẢNG SO SÁNH KẾT QUẢ")
    print("=" * 70)
    print(df.to_string())
    
    # 2. Vẽ đồ thị hội tụ trung bình
    plt.figure(figsize=(10, 6))
    colors = {'TSILS': 'purple', 'GWO': 'blue', 'WDGWO': 'red', 'CGWO': 'green'}
    linestyles = {'TSILS': '-', 'GWO': '-', 'WDGWO': '-', 'CGWO': '--'}
    markers = {'TSILS': 'o', 'GWO': 's', 'WDGWO': '^', 'CGWO': 'v'}
    
    for alg_name in algorithms.keys():
        mean_curve = np.mean(convergence_data[alg_name], axis=0)
        std_curve = np.std(convergence_data[alg_name], axis=0)
        
        plt.plot(mean_curve, label=alg_name, linewidth=2.5, 
                color=colors[alg_name], linestyle=linestyles[alg_name],
                marker=markers[alg_name], markevery=20, markersize=6)
        # Thêm vùng std
        plt.fill_between(range(len(mean_curve)), 
                        mean_curve - std_curve, 
                        mean_curve + std_curve, 
                        alpha=0.2, color=colors[alg_name])
    
    plt.xlabel('Iteration', fontsize=13, fontweight='bold')
    plt.ylabel('Best Fitness', fontsize=13, fontweight='bold')
    plt.title(f'Convergence Comparison on JCAS Problem ({n_trials} trials)', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('results/convergence_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu đồ thị hội tụ: results/convergence_comparison.png")
    plt.close()
    
    # 3. Vẽ boxplot so sánh
    plt.figure(figsize=(10, 6))
    data_for_box = []
    labels_for_box = []
    
    for alg_name in algorithms.keys():
        final_fitness = convergence_data[alg_name][:, -1]
        data_for_box.append(final_fitness)
        labels_for_box.append(alg_name)
    
    bp = plt.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    plt.ylabel('Final Fitness Value', fontsize=13, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=13, fontweight='bold')
    plt.title(f'Final Fitness Distribution ({n_trials} trials)', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('results/boxplot_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu boxplot: results/boxplot_comparison.png")
    plt.close()
    
    # 4. Vẽ beam pattern của solution tốt nhất từ mỗi thuật toán
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    
    for idx, (alg_name, best_sol) in enumerate(best_solutions.items()):
        metrics, response_db = problem.evaluate_metrics(best_sol)
        
        axes_flat[idx].plot(problem.eq_dirs, response_db, linewidth=2, 
                      color=colors[alg_name])
        axes_flat[idx].axhline(y=-3, color='gray', linestyle='--', alpha=0.5, label='-3dB')
        axes_flat[idx].axhline(y=-10, color='gray', linestyle=':', alpha=0.5, label='-10dB')
        axes_flat[idx].set_xlabel('Equivalent Direction', fontsize=11, fontweight='bold')
        axes_flat[idx].set_ylabel('Normalized Gain (dB)', fontsize=11, fontweight='bold')
        axes_flat[idx].set_title(f'{alg_name} Beam Pattern\nFitness: {metrics["fitness"]:.4f}', 
                           fontsize=12, fontweight='bold')
        axes_flat[idx].grid(True, alpha=0.3, linestyle='--')
        axes_flat[idx].set_xlim([-1, 1])
        axes_flat[idx].set_ylim([-40, 5])
        axes_flat[idx].legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/beam_patterns.png', dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu beam patterns: results/beam_patterns.png")
    plt.close()
    
    # 5. Chi tiết metrics của solution tốt nhất
    print("\n" + "=" * 70)
    print("CHI TIẾT METRICS CỦA SOLUTION TỐT NHẤT")
    print("=" * 70)
    
    metrics_data = {}
    for alg_name, best_sol in best_solutions.items():
        metrics, _ = problem.evaluate_metrics(best_sol)
        metrics_data[alg_name] = metrics
        
        print(f"\n{alg_name}:")
        for key, value in metrics.items():
            print(f"  {key:20s}: {value:10.6f}")
    
    df_metrics = pd.DataFrame(metrics_data).T
    df_metrics.to_csv('results/detailed_metrics.csv')
    print(f"\n✓ Đã lưu metrics chi tiết: results/detailed_metrics.csv")
    
    # 6. Tính improvement (so với TSILS baseline)
    print("\n" + "=" * 70)
    print("PHẦN TRĂM CẢI THIỆN SO VỚI TSILS (PHƯƠNG PHÁP GỐC)")
    print("=" * 70)
    
    baseline = results['TSILS']['mean']
    improvements = {}
    
    for alg_name in ['GWO', 'WDGWO', 'CGWO']:
        improvement = ((baseline - results[alg_name]['mean']) / abs(baseline)) * 100
        improvements[alg_name] = improvement
        print(f"{alg_name}: {improvement:+.2f}%")
    
    # Lưu improvements
    with open('results/improvements.txt', 'w') as f:
        f.write("PHẦN TRĂM CẢI THIỆN SO VỚI TSILS (PHƯƠNG PHÁP GỐC)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Baseline (TSILS): {baseline:.6f}\n\n")
        for alg_name, imp in improvements.items():
            f.write(f"{alg_name}: {imp:+.2f}%\n")
    
    print(f"✓ Đã lưu improvements: results/improvements.txt")
    
    return results, convergence_data, best_solutions

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run GWO experiments on JCAS problem')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials')
    parser.add_argument('--wolves', type=int, default=30, help='Number of wolves')
    parser.add_argument('--iterations', type=int, default=200, help='Max iterations')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("BẮT ĐẦU THÍ NGHIỆM GWO VÀ CÁC BIẾN THỂ TRÊN BÀI TOÁN JCAS")
    print("Bao gồm: TSILS (baseline), GWO, WDGWO, CGWO")
    print("=" * 70)
    print(f"Số trials: {args.trials}")
    print(f"Số wolves: {args.wolves}")
    print(f"Số iterations: {args.iterations}")
    print("=" * 70)
    
    results, convergence, solutions = run_experiments(
        n_trials=args.trials,
        n_wolves=args.wolves,
        max_iter=args.iterations
    )
    
    print("\n" + "=" * 70)
    print("✓ HOÀN TẤT THÍ NGHIỆM!")
    print("=" * 70)
    print("Các file kết quả đã được lưu trong thư mục: results/")
    print("  - comparison_table.csv      : Bảng so sánh thống kê")
    print("  - detailed_metrics.csv      : Metrics chi tiết")
    print("  - convergence_comparison.png: Đồ thị hội tụ")
    print("  - boxplot_comparison.png    : Phân phối fitness")
    print("  - beam_patterns.png         : Các beam pattern tốt nhất (4 thuật toán)")
    print("  - improvements.txt          : Phần trăm cải thiện so với TSILS")
    print("=" * 70)
