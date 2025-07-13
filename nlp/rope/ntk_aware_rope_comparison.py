import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 参数设置
d = 128  # 维度大小
b = 10000  # 基地
original_max_position = 4096  # 原始上下文长度
extended_max_position = 163840  # 扩展后上下文长度
extension_factor = 40  # 扩展倍数 (163840 / 4096 = 40)

# 计算维度索引
i_values = np.arange(0, d // 2)  # i = 0, 1, 2, ..., d/2-1


def calculate_original_rope(i_values, base, dim):
    """计算原始 RoPE 的 theta_i"""
    return base ** (-2 * i_values / dim)


def calculate_ntk_aware_rope(i_values, base, dim, extension_factor):
    """计算 NTK-Aware Scaled RoPE 的 theta_i"""
    # 根据 NTK-Aware 公式调整基地
    # base = base * extension_factor ** (dim / (dim-2))
    adjusted_base = base * (extension_factor ** (dim / (dim - 2)))
    return adjusted_base ** (-2 * i_values / dim)


def plot_theta_comparison():
    """绘制 theta_i 对比图"""
    # 计算原始和 NTK-Aware 的 theta_i
    theta_original = calculate_original_rope(i_values, b, d)
    theta_ntk = calculate_ntk_aware_rope(i_values, b, d, extension_factor)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 线性尺度对比
    ax1.plot(
        i_values,
        theta_original,
        "b-",
        linewidth=2,
        marker="o",
        markersize=4,
        label="原始 RoPE",
        alpha=0.8,
    )
    ax1.plot(
        i_values,
        theta_ntk,
        "r-",
        linewidth=2,
        marker="s",
        markersize=4,
        label="NTK-Aware Scaled RoPE",
        alpha=0.8,
    )
    ax1.set_xlabel("维度索引 i")
    ax1.set_ylabel("θ_i")
    ax1.set_title("θ_i 对比 (线性尺度)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 对数尺度对比
    ax2.semilogy(
        i_values,
        theta_original,
        "b-",
        linewidth=2,
        marker="o",
        markersize=4,
        label="原始 RoPE",
        alpha=0.8,
    )
    ax2.semilogy(
        i_values,
        theta_ntk,
        "r-",
        linewidth=2,
        marker="s",
        markersize=4,
        label="NTK-Aware Scaled RoPE",
        alpha=0.8,
    )
    ax2.set_xlabel("维度索引 i")
    ax2.set_ylabel("θ_i (log scale)")
    ax2.set_title("θ_i 对比 (对数尺度)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("theta_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_wavelength_comparison():
    """绘制波长对比图"""
    # 计算原始和 NTK-Aware 的 theta_i
    theta_original = calculate_original_rope(i_values, b, d)
    theta_ntk = calculate_ntk_aware_rope(i_values, b, d, extension_factor)

    # 计算波长
    lambda_original = 2 * np.pi / theta_original
    lambda_ntk = 2 * np.pi / theta_ntk

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 线性尺度对比
    ax1.plot(
        i_values,
        lambda_original,
        "b-",
        linewidth=2,
        marker="o",
        markersize=4,
        label="原始 RoPE",
        alpha=0.8,
    )
    ax1.plot(
        i_values,
        lambda_ntk,
        "r-",
        linewidth=2,
        marker="s",
        markersize=4,
        label="NTK-Aware Scaled RoPE",
        alpha=0.8,
    )
    ax1.set_xlabel("维度索引 i")
    ax1.set_ylabel("波长 λ_i")
    ax1.set_title("波长对比 (线性尺度)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 对数尺度对比
    ax2.semilogy(
        i_values,
        lambda_original,
        "b-",
        linewidth=2,
        marker="o",
        markersize=4,
        label="原始 RoPE",
        alpha=0.8,
    )
    ax2.semilogy(
        i_values,
        lambda_ntk,
        "r-",
        linewidth=2,
        marker="s",
        markersize=4,
        label="NTK-Aware Scaled RoPE",
        alpha=0.8,
    )
    ax2.set_xlabel("维度索引 i")
    ax2.set_ylabel("波长 λ_i (log scale)")
    ax2.set_title("波长对比 (对数尺度)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 添加上下文长度参考线
    ax2.axhline(
        y=original_max_position,
        color="blue",
        linestyle="--",
        alpha=0.5,
        label=f"原始上下文长度 = {original_max_position}",
    )
    ax2.axhline(
        y=extended_max_position,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"扩展上下文长度 = {extended_max_position}",
    )
    ax2.legend()

    plt.tight_layout()
    plt.savefig("wavelength_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_rotation_period_comparison():
    """绘制旋转周期对比图"""
    # 计算原始和 NTK-Aware 的 theta_i
    theta_original = calculate_original_rope(i_values, b, d)
    theta_ntk = calculate_ntk_aware_rope(i_values, b, d, extension_factor)

    # 选择几个代表性维度
    selected_dims = [0, 8, 16, 32, 63]
    colors = ["red", "blue", "green", "orange", "purple"]

    # 为了可视化，使用较小的位置范围
    m_values_original = np.arange(0, original_max_position)
    m_values_extended = np.arange(0, extended_max_position)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 原始 RoPE 的 sin 曲线
    for idx, dim in enumerate(selected_dims):
        if dim < len(theta_original):
            ax1.plot(
                m_values_original,
                np.sin(m_values_original * theta_original[dim]),
                color=colors[idx],
                linewidth=1.5,
                label=f"i={dim}",
                alpha=0.8,
            )

    ax1.set_xlabel("序列位置 m")
    ax1.set_ylabel("sin(m * θ_i)")
    ax1.set_title("原始 RoPE 的 sin(m * θ_i)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)

    # 2. NTK-Aware Scaled RoPE 的 sin 曲线
    for idx, dim in enumerate(selected_dims):
        if dim < len(theta_ntk):
            ax2.plot(
                m_values_extended,
                np.sin(m_values_extended * theta_ntk[dim]),
                color=colors[idx],
                linewidth=1.5,
                label=f"i={dim}",
                alpha=0.8,
            )

    ax2.set_xlabel("序列位置 m")
    ax2.set_ylabel("sin(m * θ_i)")
    ax2.set_title("NTK-Aware Scaled RoPE 的 sin(m * θ_i)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)

    # 3. 在相同位置范围内的对比 (使用原始范围)
    for idx, dim in enumerate(selected_dims):
        if dim < len(theta_original):
            ax3.plot(
                m_values_original,
                np.sin(m_values_original * theta_original[dim]),
                color=colors[idx],
                linewidth=2,
                label=f"原始 i={dim}",
                alpha=0.8,
            )
            ax3.plot(
                m_values_original,
                np.sin(m_values_original * theta_ntk[dim]),
                color=colors[idx],
                linewidth=2,
                linestyle="--",
                label=f"NTK-Aware i={dim}",
                alpha=0.8,
            )

    ax3.set_xlabel("序列位置 m")
    ax3.set_ylabel("sin(m * θ_i)")
    ax3.set_title("相同位置范围内的对比 (0-4096)")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.1, 1.1)

    # 4. 周期长度对比
    periods_original = 2 * np.pi / theta_original
    periods_ntk = 2 * np.pi / theta_ntk

    ax4.semilogy(
        i_values,
        periods_original,
        "bo-",
        linewidth=2,
        markersize=4,
        label="原始 RoPE",
        alpha=0.8,
    )
    ax4.semilogy(
        i_values,
        periods_ntk,
        "ro-",
        linewidth=2,
        markersize=4,
        label="NTK-Aware Scaled RoPE",
        alpha=0.8,
    )
    ax4.set_xlabel("维度索引 i")
    ax4.set_ylabel("周期长度 (log scale)")
    ax4.set_title("周期长度对比")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 添加参考线
    ax4.axhline(
        y=original_max_position,
        color="blue",
        linestyle="--",
        alpha=0.5,
        label=f"原始上下文长度 = {original_max_position}",
    )
    ax4.axhline(
        y=extended_max_position,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"扩展上下文长度 = {extended_max_position}",
    )
    ax4.legend()

    plt.tight_layout()
    plt.savefig("rotation_period_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_m_theta_comparison():
    """绘制 m * θ_i 对比图"""
    # 计算原始和 NTK-Aware 的 theta_i
    theta_original = calculate_original_rope(i_values, b, d)
    theta_ntk = calculate_ntk_aware_rope(i_values, b, d, extension_factor)

    # 选择几个代表性维度
    selected_dims = [0, 8, 16, 32, 63]
    colors = ["red", "blue", "green", "orange", "purple"]

    # 位置值 - 为了对比，使用原始范围
    m_values = np.arange(0, original_max_position)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 原始 RoPE 的 m * θ_i 热力图
    M, I = np.meshgrid(m_values, i_values)
    THETA_original = calculate_original_rope(I, b, d)
    Z_original = M * THETA_original

    im1 = ax1.imshow(
        Z_original,
        aspect="auto",
        cmap="viridis",
        extent=[0, original_max_position, 0, d // 2],
    )
    ax1.set_xlabel("位置 m")
    ax1.set_ylabel("维度索引 i")
    ax1.set_title("原始 RoPE 的 m * θ_i")
    plt.colorbar(im1, ax=ax1, label="m * θ_i")

    # 2. NTK-Aware Scaled RoPE 的 m * θ_i 热力图
    THETA_ntk = calculate_ntk_aware_rope(I, b, d, extension_factor)
    Z_ntk = M * THETA_ntk

    im2 = ax2.imshow(
        Z_ntk,
        aspect="auto",
        cmap="viridis",
        extent=[0, original_max_position, 0, d // 2],
        vmin=Z_original.min(),
        vmax=Z_original.max(),
    )  # 使用相同的色彩范围
    ax2.set_xlabel("位置 m")
    ax2.set_ylabel("维度索引 i")
    ax2.set_title("NTK-Aware Scaled RoPE 的 m * θ_i")
    plt.colorbar(im2, ax=ax2, label="m * θ_i")

    # 3. 不同维度下的 m * θ_i 对比
    for idx, dim in enumerate(selected_dims):
        if dim < len(theta_original):
            ax3.plot(
                m_values,
                m_values * theta_original[dim],
                color=colors[idx],
                linewidth=2,
                label=f"原始 i={dim}",
                alpha=0.8,
            )
            ax3.plot(
                m_values,
                m_values * theta_ntk[dim],
                color=colors[idx],
                linewidth=2,
                linestyle="--",
                label=f"NTK-Aware i={dim}",
                alpha=0.8,
            )

    ax3.set_xlabel("位置 m")
    ax3.set_ylabel("m * θ_i")
    ax3.set_title("不同维度下的 m * θ_i 对比")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.grid(True, alpha=0.3)

    # 4. 不同位置下的 m * θ_i 对比
    selected_positions = [0, 1024, 2048, 3072, 4095]
    pos_colors = ["red", "blue", "green", "orange", "purple"]

    for idx, pos in enumerate(selected_positions):
        if pos < len(m_values):
            ax4.plot(
                i_values,
                pos * theta_original,
                color=pos_colors[idx],
                linewidth=2,
                label=f"原始 m={pos}",
                alpha=0.8,
            )
            ax4.plot(
                i_values,
                pos * theta_ntk,
                color=pos_colors[idx],
                linewidth=2,
                linestyle="--",
                label=f"NTK-Aware m={pos}",
                alpha=0.8,
            )

    ax4.set_xlabel("维度索引 i")
    ax4.set_ylabel("m * θ_i")
    ax4.set_title("不同位置下的 m * θ_i 对比")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("m_theta_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_scaled_position_comparison():
    """绘制不同 m 尺度上的 NTK-Aware Scaled RoPE 与原始 RoPE 对比"""
    # 计算原始和 NTK-Aware 的 theta_i
    theta_original = calculate_original_rope(i_values, b, d)
    theta_ntk = calculate_ntk_aware_rope(i_values, b, d, extension_factor)

    # 定义原始位置和对应的扩展位置
    original_positions = [0, 1024, 2048, 4096]
    scaled_positions = [
        pos * extension_factor for pos in original_positions
    ]  # [0, 40960, 81920, 163840]

    # 选择几个代表性维度
    selected_dims = [0, 8, 16, 32, 63]
    colors = ["red", "blue", "green", "orange", "purple"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. m * θ_i 在不同位置的对比 - 按维度分组
    for idx, dim in enumerate(selected_dims):
        if dim < len(theta_original):
            # 原始 RoPE 的 m * θ_i
            original_values = [pos * theta_original[dim] for pos in original_positions]
            # NTK-Aware Scaled RoPE 的 m * θ_i
            scaled_values = [pos * theta_ntk[dim] for pos in scaled_positions]

            ax1.plot(
                original_positions,
                original_values,
                "o-",
                color=colors[idx],
                linewidth=2,
                markersize=6,
                label=f"原始 RoPE i={dim}",
                alpha=0.8,
            )
            ax1.plot(
                original_positions,
                scaled_values,
                "s--",
                color=colors[idx],
                linewidth=2,
                markersize=6,
                label=f"NTK-Aware (扩展位置) i={dim}",
                alpha=0.8,
            )

    ax1.set_xlabel("原始位置 m")
    ax1.set_ylabel("m * θ_i")
    ax1.set_title("不同维度在对应位置的 m * θ_i 对比")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(original_positions)

    # 2. sin(m * θ_i) 在不同位置的对比
    for idx, dim in enumerate(selected_dims):
        if dim < len(theta_original):
            # 原始 RoPE 的 sin(m * θ_i)
            original_sin_values = [
                np.sin(pos * theta_original[dim]) for pos in original_positions
            ]
            # NTK-Aware Scaled RoPE 的 sin(m * θ_i)
            scaled_sin_values = [
                np.sin(pos * theta_ntk[dim]) for pos in scaled_positions
            ]

            ax2.plot(
                original_positions,
                original_sin_values,
                "o-",
                color=colors[idx],
                linewidth=2,
                markersize=6,
                label=f"原始 RoPE i={dim}",
                alpha=0.8,
            )
            ax2.plot(
                original_positions,
                scaled_sin_values,
                "s--",
                color=colors[idx],
                linewidth=2,
                markersize=6,
                label=f"NTK-Aware (扩展位置) i={dim}",
                alpha=0.8,
            )

    ax2.set_xlabel("原始位置 m")
    ax2.set_ylabel("sin(m * θ_i)")
    ax2.set_title("不同维度在对应位置的 sin(m * θ_i) 对比")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(original_positions)
    ax2.set_ylim(-1.1, 1.1)

    # 3. 相对位置编码对比（归一化后）
    # 计算相对位置 (0, 0.25, 0.5, 1.0)
    relative_positions = [pos / original_max_position for pos in original_positions]

    for idx, dim in enumerate(selected_dims):
        if dim < len(theta_original):
            # 原始 RoPE
            original_relative_values = [
                pos * theta_original[dim] for pos in original_positions
            ]
            # NTK-Aware Scaled RoPE (使用扩展位置)
            scaled_relative_values = [pos * theta_ntk[dim] for pos in scaled_positions]

            ax3.plot(
                relative_positions,
                original_relative_values,
                "o-",
                color=colors[idx],
                linewidth=2,
                markersize=6,
                label=f"原始 RoPE i={dim}",
                alpha=0.8,
            )
            ax3.plot(
                relative_positions,
                scaled_relative_values,
                "s--",
                color=colors[idx],
                linewidth=2,
                markersize=6,
                label=f"NTK-Aware (对应相对位置) i={dim}",
                alpha=0.8,
            )

    ax3.set_xlabel("相对位置 (m / max_length)")
    ax3.set_ylabel("m * θ_i")
    ax3.set_title("相对位置编码的对比")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(relative_positions)
    ax3.set_xticklabels(["0", "0.25", "0.5", "1.0"])

    # 4. 编码差异分析
    position_labels = ["0", "1024", "2048", "4096"]
    x_pos = np.arange(len(position_labels))

    # 计算每个位置在不同维度上的平均编码差异
    avg_differences = []
    for pos_idx, (orig_pos, scaled_pos) in enumerate(
        zip(original_positions, scaled_positions)
    ):
        differences = []
        for dim in selected_dims:
            if dim < len(theta_original):
                orig_val = orig_pos * theta_original[dim]
                scaled_val = scaled_pos * theta_ntk[dim]
                diff = abs(orig_val - scaled_val)
                differences.append(diff)
        avg_differences.append(np.mean(differences))

    bars = ax4.bar(x_pos, avg_differences, color="skyblue", alpha=0.7)
    ax4.set_xlabel("位置")
    ax4.set_ylabel("平均编码差异 |原始 - NTK-Aware|")
    ax4.set_title("不同位置的编码差异")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(position_labels)
    ax4.grid(True, alpha=0.3)

    # 在柱状图上添加数值标签
    for bar, diff in zip(bars, avg_differences):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{diff:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("scaled_position_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 打印详细的数值对比
    print("\n" + "=" * 80)
    print("不同位置的详细编码对比:")
    print("=" * 80)
    print("位置\t\t原始位置\t扩展位置\t相对位置")
    for orig_pos, scaled_pos in zip(original_positions, scaled_positions):
        rel_pos = orig_pos / original_max_position
        print(f"{orig_pos}\t\t{orig_pos}\t\t{scaled_pos}\t\t{rel_pos:.2f}")

    print("\n各维度的 m * θ_i 值对比:")
    print("-" * 80)
    for dim in selected_dims:
        if dim < len(theta_original):
            print(f"\n维度 i={dim}:")
            print("位置\t原始 m*θ_i\t\tNTK-Aware m*θ_i\t差异")
            for orig_pos, scaled_pos in zip(original_positions, scaled_positions):
                orig_val = orig_pos * theta_original[dim]
                scaled_val = scaled_pos * theta_ntk[dim]
                diff = abs(orig_val - scaled_val)
                print(f"{orig_pos}\t{orig_val:.4f}\t\t{scaled_val:.4f}\t\t{diff:.4f}")


def plot_scaling_effect_analysis():
    """绘制缩放效果分析图"""
    # 计算原始和 NTK-Aware 的 theta_i
    theta_original = calculate_original_rope(i_values, b, d)
    theta_ntk = calculate_ntk_aware_rope(i_values, b, d, extension_factor)

    # 计算缩放比例
    scaling_ratio = theta_ntk / theta_original

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 缩放比例分析
    ax1.plot(i_values, scaling_ratio, "g-", linewidth=2, marker="o", markersize=4)
    ax1.set_xlabel("维度索引 i")
    ax1.set_ylabel("缩放比例 (θ_ntk / θ_original)")
    ax1.set_title("NTK-Aware Scaled RoPE 的缩放比例")
    ax1.grid(True, alpha=0.3)

    # 2. 缩放比例 (对数尺度)
    ax2.semilogy(i_values, scaling_ratio, "g-", linewidth=2, marker="o", markersize=4)
    ax2.set_xlabel("维度索引 i")
    ax2.set_ylabel("缩放比例 (log scale)")
    ax2.set_title("NTK-Aware Scaled RoPE 的缩放比例 (对数尺度)")
    ax2.grid(True, alpha=0.3)

    # 3. 基地调整效果
    original_base = b
    adjusted_base = b * (extension_factor ** (d / (d - 2)))
    base_scaling = adjusted_base / original_base

    ax3.bar(
        ["原始基地", "NTK-Aware 基地"],
        [original_base, adjusted_base],
        color=["blue", "red"],
        alpha=0.7,
    )
    ax3.set_ylabel("基地值")
    ax3.set_title(f"基地调整 (缩放因子: {base_scaling:.2f})")
    ax3.grid(True, alpha=0.3)

    # 4. 不同维度的频率变化
    selected_dims = [0, 16, 32, 48, 63]
    dim_labels = [f"i={dim}" for dim in selected_dims]

    original_freqs = [
        1 / (2 * np.pi / theta_original[dim]) if dim < len(theta_original) else 0
        for dim in selected_dims
    ]
    ntk_freqs = [
        1 / (2 * np.pi / theta_ntk[dim]) if dim < len(theta_ntk) else 0
        for dim in selected_dims
    ]

    x = np.arange(len(dim_labels))
    width = 0.35

    ax4.bar(x - width / 2, original_freqs, width, label="原始 RoPE", alpha=0.8)
    ax4.bar(x + width / 2, ntk_freqs, width, label="NTK-Aware Scaled RoPE", alpha=0.8)
    ax4.set_xlabel("维度")
    ax4.set_ylabel("频率")
    ax4.set_title("不同维度的频率对比")
    ax4.set_xticks(x)
    ax4.set_xticklabels(dim_labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("scaling_effect_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_comparison_statistics():
    """打印对比统计信息"""
    # 计算原始和 NTK-Aware 的 theta_i
    theta_original = calculate_original_rope(i_values, b, d)
    theta_ntk = calculate_ntk_aware_rope(i_values, b, d, extension_factor)

    # 计算波长
    lambda_original = 2 * np.pi / theta_original
    lambda_ntk = 2 * np.pi / theta_ntk

    print("NTK-Aware Scaled RoPE 对比统计信息:")
    print(f"维度 d = {d}")
    print(f"原始基地 b = {b}")
    print(f"扩展倍数 = {extension_factor}")
    print(f"调整后基地 = {b * (extension_factor ** (d / (d - 2))):.2f}")
    print(f"原始上下文长度 = {original_max_position}")
    print(f"扩展上下文长度 = {extended_max_position}")
    print("\n" + "=" * 100)
    print(
        "维度i\t原始θ_i\t\tNTK-Aware θ_i\t缩放比例\t原始波长\tNTK-Aware波长\t波长比例"
    )
    print("=" * 100)

    for i in range(0, len(i_values), 8):  # 每8个维度打印一次
        if i < len(theta_original):
            theta_orig = theta_original[i]
            theta_ntk_val = theta_ntk[i]
            scaling_ratio = theta_ntk_val / theta_orig
            lambda_orig = lambda_original[i]
            lambda_ntk_val = lambda_ntk[i]
            lambda_ratio = lambda_ntk_val / lambda_orig

            print(
                f"{i}\t{theta_orig:.2e}\t{theta_ntk_val:.2e}\t{scaling_ratio:.2f}\t\t"
                f"{lambda_orig:.1f}\t{lambda_ntk_val:.1f}\t\t{lambda_ratio:.2f}"
            )


def plot_m_theta_scale_comparison():
    """绘制不同 m 尺度上的 m·θ_i 对比图"""
    # 原始位置
    original_positions = [0, 512, 1024, 2048, 4096]
    # NTK-Aware 对应的扩展位置
    scaled_positions = [pos * extension_factor for pos in original_positions]

    # 计算原始和 NTK-Aware 的 theta_i
    theta_original = calculate_original_rope(i_values, b, d)
    theta_ntk = calculate_ntk_aware_rope(i_values, b, d, extension_factor)

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        "不同 m 尺度上的 m·θ_i 对比 (原始 vs NTK-Aware Scaled RoPE)",
        fontsize=16,
        fontweight="bold",
    )

    # 颜色设置
    colors = ["blue", "green", "red", "purple", "orange"]

    # 绘制原始 RoPE 的 m·θ_i
    ax1 = axes[0, 0]
    for j, m in enumerate(original_positions):
        if m == 0:
            continue  # 跳过 m=0 的情况
        m_theta_original = m * theta_original
        ax1.plot(
            i_values,
            m_theta_original,
            color=colors[j],
            linewidth=2,
            label=f"m={m}",
            alpha=0.8,
        )

    ax1.set_xlabel("维度索引 i")
    ax1.set_ylabel("m·θ_i")
    ax1.set_title("原始 RoPE: m·θ_i")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # 绘制 NTK-Aware Scaled RoPE 的 m·θ_i
    ax2 = axes[0, 1]
    for j, m in enumerate(scaled_positions):
        if m == 0:
            continue  # 跳过 m=0 的情况
        m_theta_ntk = m * theta_ntk
        ax2.plot(
            i_values,
            m_theta_ntk,
            color=colors[j],
            linewidth=2,
            label=f"m={m}",
            alpha=0.8,
        )

    ax2.set_xlabel("维度索引 i")
    ax2.set_ylabel("m·θ_i")
    ax2.set_title("NTK-Aware Scaled RoPE: m·θ_i")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    # 重叠对比图
    ax3 = axes[0, 2]
    # 选择几个代表性的位置进行对比
    comparison_indices = [1, 2, 3, 4]  # 对应 m=512, 1024, 2048, 4096

    for j in comparison_indices:
        # 原始 RoPE
        m_original = original_positions[j]
        m_theta_original = m_original * theta_original
        ax3.plot(
            i_values,
            m_theta_original,
            color=colors[j],
            linewidth=2,
            linestyle="-",
            label=f"原始 m={m_original}",
            alpha=0.8,
        )

        # NTK-Aware RoPE
        m_scaled = scaled_positions[j]
        m_theta_ntk = m_scaled * theta_ntk
        ax3.plot(
            i_values,
            m_theta_ntk,
            color=colors[j],
            linewidth=2,
            linestyle="--",
            label=f"NTK m={m_scaled}",
            alpha=0.8,
        )

    ax3.set_xlabel("维度索引 i")
    ax3.set_ylabel("m·θ_i")
    ax3.set_title("重叠对比: 原始 vs NTK-Aware")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")

    # 不同维度的对比 (线性尺度)
    ax4 = axes[1, 0]
    selected_dims = [0, 16, 32, 48, 63]  # 选择几个代表性维度

    for dim_idx in selected_dims:
        if dim_idx >= len(i_values):
            continue

        # 原始 RoPE 在不同 m 下的值
        original_values = [
            m * theta_original[dim_idx] for m in original_positions[1:]
        ]  # 跳过 m=0
        # NTK-Aware 在不同 m 下的值
        ntk_values = [m * theta_ntk[dim_idx] for m in scaled_positions[1:]]  # 跳过 m=0

        positions = original_positions[1:]
        ax4.plot(
            positions,
            original_values,
            "o-",
            linewidth=2,
            label=f"原始 i={dim_idx}",
            alpha=0.8,
        )
        ax4.plot(
            positions,
            ntk_values,
            "s--",
            linewidth=2,
            label=f"NTK i={dim_idx}",
            alpha=0.8,
        )

    ax4.set_xlabel("原始位置 m")
    ax4.set_ylabel("m·θ_i")
    ax4.set_title("不同维度的缩放对比")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale("log")

    # 相对差异分析
    ax5 = axes[1, 1]
    for j in comparison_indices:
        m_original = original_positions[j]
        m_scaled = scaled_positions[j]

        m_theta_original = m_original * theta_original
        m_theta_ntk = m_scaled * theta_ntk

        # 计算相对差异
        relative_diff = np.abs(m_theta_ntk - m_theta_original) / m_theta_original
        ax5.plot(
            i_values,
            relative_diff,
            color=colors[j],
            linewidth=2,
            label=f"m={m_original} vs {m_scaled}",
            alpha=0.8,
        )

    ax5.set_xlabel("维度索引 i")
    ax5.set_ylabel("相对差异")
    ax5.set_title("相对差异分析")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale("log")

    # 数值统计
    ax6 = axes[1, 2]
    ax6.axis("off")

    # 打印一些统计信息
    stats_text = "数值统计:\n\n"
    stats_text += f"扩展倍数: {extension_factor}x\n"
    stats_text += f"维度数: {d}\n"
    stats_text += f"原始基地: {b}\n"
    adjusted_base = b * (extension_factor ** (d / (d - 2)))
    stats_text += f"调整后基地: {adjusted_base:.2f}\n\n"

    stats_text += "原始位置 → 扩展位置:\n"
    for orig, scaled in zip(original_positions, scaled_positions):
        stats_text += f"{orig} → {scaled}\n"

    stats_text += "\n高频维度 (i=0):\n"
    m_orig = original_positions[2]  # m=1024
    m_scaled = scaled_positions[2]  # m=40960
    val_orig = m_orig * theta_original[0]
    val_scaled = m_scaled * theta_ntk[0]
    stats_text += f"原始: {val_orig:.2f}\n"
    stats_text += f"NTK: {val_scaled:.2f}\n"
    stats_text += f"差异: {abs(val_scaled - val_orig):.2f}\n"

    stats_text += "\n低频维度 (i=63):\n"
    val_orig = m_orig * theta_original[63]
    val_scaled = m_scaled * theta_ntk[63]
    stats_text += f"原始: {val_orig:.6f}\n"
    stats_text += f"NTK: {val_scaled:.6f}\n"
    stats_text += f"差异: {abs(val_scaled - val_orig):.6f}\n"

    ax6.text(
        0.05,
        0.95,
        stats_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    plt.savefig("m_theta_scale_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 打印详细的数值对比
    print("\n=== m·θ_i 尺度对比详细分析 ===")
    print(f"扩展倍数: {extension_factor}")
    print(f"调整后基地: {adjusted_base:.2f}")
    print("\n位置对比:")
    print("原始位置 | 扩展位置 | 扩展倍数")
    print("-" * 30)
    for orig, scaled in zip(original_positions, scaled_positions):
        if orig == 0:
            print(f"{orig:8d} | {scaled:8d} | N/A")
        else:
            print(f"{orig:8d} | {scaled:8d} | {scaled/orig:.1f}x")

    print("\n不同维度的 m·θ_i 值对比 (m=1024 vs m=40960):")
    print("维度 i | 原始 m·θ_i | NTK m·θ_i | 绝对差异 | 相对差异")
    print("-" * 60)
    m_orig = 1024
    m_scaled = 40960
    for i in [0, 8, 16, 32, 48, 63]:
        val_orig = m_orig * theta_original[i]
        val_scaled = m_scaled * theta_ntk[i]
        abs_diff = abs(val_scaled - val_orig)
        rel_diff = abs_diff / val_orig
        print(
            f"{i:6d} | {val_orig:10.4f} | {val_scaled:10.4f} | {abs_diff:9.4f} | {rel_diff:8.4f}"
        )


def main():
    """主函数"""
    print("开始绘制 NTK-Aware Scaled RoPE 对比图...")

    # 打印统计信息
    print_comparison_statistics()

    # 绘制各种对比图
    print("\n1. 绘制 θ_i 对比图...")
    plot_theta_comparison()

    print("2. 绘制波长对比图...")
    plot_wavelength_comparison()

    # print("3. 绘制旋转周期对比图...")
    # plot_rotation_period_comparison()

    print("4. 绘制 m * θ_i 对比图...")
    plot_m_theta_comparison()

    print("5. 绘制扩展位置对比图...")
    plot_scaled_position_comparison()

    print("6. 绘制缩放效果分析图...")
    plot_scaling_effect_analysis()

    print("7. 绘制 m·θ_i 尺度对比图...")
    plot_m_theta_scale_comparison()

    print("\n所有 NTK-Aware Scaled RoPE 对比图已保存完成!")


if __name__ == "__main__":
    main()
