import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 设置绘图参数
plt.rcParams["figure.figsize"] = (16, 12)
plt.rcParams["font.size"] = 12

# 参数设置
d = 128  # 维度
base = 10000  # 基底
alpha = 1  # NTK-by-parts参数
beta = 32  # NTK-by-parts参数
L_orig = 4096  # 原始上下文长度
L_ext = 163840  # 扩展后上下文长度
k = 40  # 扩展倍数

# 维度索引
i_values = np.arange(0, d // 2)


def compute_original_rope_params(i_values, base, d, L):
    """计算原始RoPE参数"""
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


# 计算各种参数
theta_orig, wavelength_orig, r_orig = compute_original_rope_params(
    i_values, base, d, L_orig
)
gamma_values = compute_gamma(r_orig, alpha, beta)
theta_ntk_by_parts = compute_ntk_by_parts_theta(theta_orig, gamma_values, k)
theta_ntk_aware = compute_ntk_aware_theta(i_values, base, d, k)

# 创建图表
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

# 颜色配置
colors = {
    "original": "#1f77b4",
    "pi": "#ff7f0e",
    "ntk_aware": "#2ca02c",
    "ntk_by_parts": "#d62728",
    "gamma": "#9467bd",
}

# 1. θ_i 对比图
ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogy(
    i_values,
    theta_orig,
    "o-",
    color=colors["original"],
    label="Original RoPE",
    linewidth=2,
    markersize=4,
)
ax1.semilogy(
    i_values,
    theta_orig / k,
    "s-",
    color=colors["pi"],
    label="Position Interpolation (θ/k)",
    linewidth=2,
    markersize=4,
)
ax1.semilogy(
    i_values,
    theta_ntk_aware,
    "^-",
    color=colors["ntk_aware"],
    label="NTK-aware",
    linewidth=2,
    markersize=4,
)
ax1.semilogy(
    i_values,
    theta_ntk_by_parts,
    "v-",
    color=colors["ntk_by_parts"],
    label="NTK-by-parts",
    linewidth=2,
    markersize=4,
)

ax1.set_xlabel("Dimension Index i")
ax1.set_ylabel("θ_i (log scale)")
ax1.set_title("Frequency Parameters θ_i Comparison")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 波长和上下文长度对比
ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogy(
    i_values,
    wavelength_orig,
    "o-",
    color=colors["original"],
    label="Wavelength λ_i",
    linewidth=2,
    markersize=4,
)
ax2.axhline(
    y=L_orig,
    color="red",
    linestyle="--",
    alpha=0.8,
    label=f"Original Context Length ({L_orig})",
    linewidth=2,
)
ax2.axhline(
    y=L_ext,
    color="purple",
    linestyle="--",
    alpha=0.8,
    label=f"Extended Context Length ({L_ext})",
    linewidth=2,
)

# 添加维度分类区域
high_freq_mask = r_orig > beta
mid_freq_mask = (r_orig >= alpha) & (r_orig <= beta)
low_freq_mask = r_orig < alpha

ax2.fill_between(
    i_values[high_freq_mask],
    1,
    1e8,
    alpha=0.2,
    color="red",
    label="High Freq (No Interpolation)",
)
ax2.fill_between(
    i_values[mid_freq_mask],
    1,
    1e8,
    alpha=0.2,
    color="orange",
    label="Mid Freq (Partial Interpolation)",
)
ax2.fill_between(
    i_values[low_freq_mask],
    1,
    1e8,
    alpha=0.2,
    color="green",
    label="Low Freq (Full Interpolation)",
)

ax2.set_xlabel("Dimension Index i")
ax2.set_ylabel("Wavelength λ_i (log scale)")
ax2.set_title("Wavelength vs Context Length")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(1, 1e8)

# 3. γ(r) 函数和维度分类
ax3 = fig.add_subplot(gs[1, 0])
ax3_twin = ax3.twinx()

# 绘制频率比 r(i)
line1 = ax3.semilogy(
    i_values,
    r_orig,
    "o-",
    color="blue",
    label="r(i) = L/λ_i",
    linewidth=2,
    markersize=4,
)
ax3.axhline(
    y=alpha, color="green", linestyle="--", alpha=0.8, label=f"α = {alpha}", linewidth=2
)
ax3.axhline(
    y=beta, color="red", linestyle="--", alpha=0.8, label=f"β = {beta}", linewidth=2
)
ax3.set_xlabel("Dimension Index i")
ax3.set_ylabel("Frequency Ratio r(i)", color="blue")

# 绘制γ(r)
line2 = ax3_twin.plot(
    i_values,
    gamma_values,
    "s-",
    color=colors["gamma"],
    label="γ(r)",
    linewidth=2,
    markersize=4,
)
ax3_twin.set_ylabel("γ(r)", color=colors["gamma"])
ax3_twin.set_ylim(-0.1, 1.1)

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(
    lines
    + [
        plt.Line2D([0], [0], color="green", linestyle="--"),
        plt.Line2D([0], [0], color="red", linestyle="--"),
    ],
    labels + [f"α = {alpha}", f"β = {beta}"],
    loc="center right",
)
ax3.set_title("Frequency Ratio r(i) and γ(r) Function")
ax3.grid(True, alpha=0.3)

# 4. 旋转角度在不同位置的对比
ax4 = fig.add_subplot(gs[1, 1])
sample_positions = [1024, 4096, 16384, 65536]

for i, pos in enumerate(sample_positions):
    if pos <= L_ext:
        # 计算不同方法的旋转角度
        angles_orig = pos * theta_orig
        angles_pi = (pos * L_orig / L_ext) * theta_orig  # 位置插值
        angles_ntk_aware = pos * theta_ntk_aware
        angles_ntk_by_parts = pos * theta_ntk_by_parts

        # 使用不同的透明度来表示不同位置
        alpha_val = 0.3 + 0.7 * i / len(sample_positions)

        if i == 0:  # 只为第一条线添加标签
            ax4.plot(
                i_values,
                angles_orig,
                "-",
                color=colors["original"],
                alpha=alpha_val,
                linewidth=2,
                label="Original RoPE",
            )
            ax4.plot(
                i_values,
                angles_pi,
                "--",
                color=colors["pi"],
                alpha=alpha_val,
                linewidth=2,
                label="Position Interpolation",
            )
            ax4.plot(
                i_values,
                angles_ntk_aware,
                ":",
                color=colors["ntk_aware"],
                alpha=alpha_val,
                linewidth=2,
                label="NTK-aware",
            )
            ax4.plot(
                i_values,
                angles_ntk_by_parts,
                "-.",
                color=colors["ntk_by_parts"],
                alpha=alpha_val,
                linewidth=2,
                label="NTK-by-parts",
            )
        else:
            ax4.plot(
                i_values,
                angles_orig,
                "-",
                color=colors["original"],
                alpha=alpha_val,
                linewidth=2,
            )
            ax4.plot(
                i_values,
                angles_pi,
                "--",
                color=colors["pi"],
                alpha=alpha_val,
                linewidth=2,
            )
            ax4.plot(
                i_values,
                angles_ntk_aware,
                ":",
                color=colors["ntk_aware"],
                alpha=alpha_val,
                linewidth=2,
            )
            ax4.plot(
                i_values,
                angles_ntk_by_parts,
                "-.",
                color=colors["ntk_by_parts"],
                alpha=alpha_val,
                linewidth=2,
            )

ax4.set_xlabel("Dimension Index i")
ax4.set_ylabel("Rotation Angle m·θ_i")
ax4.set_title("Rotation Angles at Different Positions")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ntk_by_parts_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 输出详细的分析信息
print("=== NTK-by-parts 详细分析 ===")
print(f"参数设置:")
print(f"  维度 d = {d}")
print(f"  基底 base = {base}")
print(f"  α = {alpha}, β = {beta}")
print(f"  原始上下文长度 L_orig = {L_orig}")
print(f"  扩展后上下文长度 L_ext = {L_ext}")
print(f"  扩展倍数 k = {k}")
print()

# 分析维度分类
no_interp_mask = r_orig > beta
partial_interp_mask = (r_orig >= alpha) & (r_orig <= beta)
full_interp_mask = r_orig < alpha

print(f"维度分类详情:")
print(
    f"  高频维度 (r > β = {beta}): {np.sum(no_interp_mask)} 个, 占比 {np.sum(no_interp_mask)/len(i_values)*100:.1f}%"
)
print(f"    - 保持原始频率，不做插值")
print(
    f"    - 维度范围: {i_values[no_interp_mask].min()} - {i_values[no_interp_mask].max()}"
)
print(
    f"  中频维度 (α ≤ r ≤ β): {np.sum(partial_interp_mask)} 个, 占比 {np.sum(partial_interp_mask)/len(i_values)*100:.1f}%"
)
print(f"    - 部分插值，介于原始和PI之间")
print(
    f"    - 维度范围: {i_values[partial_interp_mask].min()} - {i_values[partial_interp_mask].max()}"
)
print(
    f"  低频维度 (r < α = {alpha}): {np.sum(full_interp_mask)} 个, 占比 {np.sum(full_interp_mask)/len(i_values)*100:.1f}%"
)
print(f"    - 完全插值，等同于位置插值")
print(
    f"    - 维度范围: {i_values[full_interp_mask].min()} - {i_values[full_interp_mask].max()}"
)
print()

# 算法对比
print("算法特点对比:")
print("1. 原始 RoPE:")
print("   - 所有维度使用相同的位置索引")
print("   - 超出训练长度时性能急剧下降")
print()
print("2. 位置插值 (PI):")
print("   - 将位置压缩到训练范围内")
print("   - 损失高频信息，但保持相对位置关系")
print()
print("3. NTK-aware:")
print("   - 统一调整所有维度的基底")
print("   - 在高频和低频间做平衡")
print()
print("4. NTK-by-parts:")
print("   - 针对不同频率的维度采用不同策略")
print("   - 保持高频信息同时实现长度扩展")
print("   - 在各个维度间达到最优平衡")
