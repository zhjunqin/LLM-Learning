import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 设置绘图参数
plt.rcParams["font.size"] = 11
plt.rcParams["figure.figsize"] = (18, 12)

# 参数设置
d = 128
base = 10000
alpha = 1
beta = 32
L_orig = 4096
L_ext = 163840
k = 40

# 维度索引
i_values = np.arange(0, d // 2)


def compute_rope_parameters(i_values, base, d, L):
    """计算RoPE参数"""
    theta_i = base ** (-2 * i_values / d)
    wavelength = 2 * np.pi / theta_i
    r = L / wavelength
    return theta_i, wavelength, r


def compute_gamma(r, alpha, beta):
    """计算γ(r)函数"""
    gamma = np.zeros_like(r)
    gamma[r > beta] = 0
    gamma[r < alpha] = 1
    mask = (r >= alpha) & (r <= beta)
    gamma[mask] = (r[mask] - beta) / (alpha - beta)
    return gamma


def compute_ntk_by_parts_theta(theta_orig, gamma, k):
    """计算NTK-by-parts的theta_i"""
    return (1 - gamma) * theta_orig / k + gamma * theta_orig


def compute_ntk_aware_theta(i_values, base, d, k):
    """计算NTK-aware的theta_i"""
    new_base = base * (k ** (d / (d - 2)))
    return new_base ** (-2 * i_values / d)


# 计算各种方法的参数
theta_orig, wavelength_orig, r_orig = compute_rope_parameters(i_values, base, d, L_orig)
gamma_values = compute_gamma(r_orig, alpha, beta)
theta_ntk_by_parts = compute_ntk_by_parts_theta(theta_orig, gamma_values, k)
theta_ntk_aware = compute_ntk_aware_theta(i_values, base, d, k)

# 创建图表
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])

# 颜色配置
colors = {
    "original": "#1f77b4",
    "pi": "#ff7f0e",
    "ntk_aware": "#2ca02c",
    "ntk_by_parts": "#d62728",
}

# 选择不同的位置进行比较
positions = [512, 2048, 8192, 32768, 65536, 163840]

# 为每个位置创建子图
for idx, pos in enumerate(positions):
    if idx < 6:  # 只绘制前6个位置
        ax = fig.add_subplot(gs[idx // 3, idx % 3])

        # 计算各方法的 m*θ_i
        angles_orig = pos * theta_orig
        angles_pi = (pos * L_orig / L_ext) * theta_orig  # 位置插值
        angles_ntk_aware = pos * theta_ntk_aware
        angles_ntk_by_parts = pos * theta_ntk_by_parts

        # 绘制曲线
        ax.semilogy(
            i_values,
            angles_orig,
            "o-",
            color=colors["original"],
            label="Original RoPE",
            linewidth=2,
            markersize=3,
        )
        ax.semilogy(
            i_values,
            angles_pi,
            "s-",
            color=colors["pi"],
            label="Position Interpolation",
            linewidth=2,
            markersize=3,
        )
        ax.semilogy(
            i_values,
            angles_ntk_aware,
            "^-",
            color=colors["ntk_aware"],
            label="NTK-aware",
            linewidth=2,
            markersize=3,
        )
        ax.semilogy(
            i_values,
            angles_ntk_by_parts,
            "v-",
            color=colors["ntk_by_parts"],
            label="NTK-by-parts",
            linewidth=2,
            markersize=3,
        )

        # 添加重要的角度标记
        ax.axhline(y=np.pi, color="gray", linestyle="--", alpha=0.5, label="π")
        ax.axhline(y=2 * np.pi, color="gray", linestyle=":", alpha=0.5, label="2π")

        ax.set_xlabel("Dimension Index i")
        ax.set_ylabel("m·θ_i (log scale)")
        ax.set_title(f"Position m = {pos}")
        ax.grid(True, alpha=0.3)

        # 只在第一个子图中添加图例
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

# 添加总标题
fig.suptitle(
    "Detailed m·θ_i Comparison: Algorithm Evolution Analysis", fontsize=16, y=0.98
)

plt.tight_layout()
plt.savefig("detailed_m_theta_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# 输出一些关键的分析信息
print("=== 详细的 m·θ_i 分析 ===")
print(f"参数设置: d={d}, L_orig={L_orig}, L_ext={L_ext}, k={k}")
print(f"α={alpha}, β={beta}")
print()

# 分析不同维度在不同位置的表现
print("关键维度在不同位置的旋转角度对比:")
print("位置\\维度", end="")
key_dims = [0, 8, 16, 24, 32, 40, 48, 56, 63]
for dim in key_dims:
    print(f"\t{dim:2d}", end="")
print()

for pos in [1024, 4096, 16384, 65536]:
    print(f"{pos:5d}", end="")
    for dim in key_dims:
        if dim < len(theta_orig):
            angle_orig = pos * theta_orig[dim]
            angle_ntk = pos * theta_ntk_by_parts[dim]
            ratio = angle_ntk / angle_orig
            print(f"\t{ratio:.2f}", end="")
    print()

print("\n维度分类统计:")
high_freq_count = np.sum(r_orig > beta)
mid_freq_count = np.sum((r_orig >= alpha) & (r_orig <= beta))
low_freq_count = np.sum(r_orig < alpha)

print(
    f"高频维度 (r > β={beta}): {high_freq_count} 个 ({high_freq_count/len(i_values)*100:.1f}%)"
)
print(
    f"中频维度 (α ≤ r ≤ β): {mid_freq_count} 个 ({mid_freq_count/len(i_values)*100:.1f}%)"
)
print(
    f"低频维度 (r < α={alpha}): {low_freq_count} 个 ({low_freq_count/len(i_values)*100:.1f}%)"
)

print("\n算法演进总结:")
print("1. 原始 RoPE → 位置插值 (PI):")
print("   - 解决了长度外推问题")
print("   - 但损失了高频信息")

print("2. 位置插值 → NTK-aware:")
print("   - 通过调整基底保持更多信息")
print("   - 但对所有维度统一处理")

print("3. NTK-aware → NTK-by-parts:")
print("   - 根据维度频率特性分类处理")
print("   - 高频维度保持不变，低频维度插值")
print("   - 实现了最优的信息保持与扩展平衡")

print("\n关键优势:")
print("- 保持了高频维度的精细信息")
print("- 对低频维度进行适当的插值")
print("- 在中频维度实现平滑过渡")
print("- 整体性能在长文本处理中表现最佳")
