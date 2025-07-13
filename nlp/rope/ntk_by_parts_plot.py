import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def compute_ntk_by_parts_theta(
    d, base=10000, original_length=4096, extended_length=163840, alpha=1, beta=32
):
    """
    计算 NTK-by-parts 方法的 θ_i 值

    Args:
        d: 维度大小
        base: 基地，默认 10000
        original_length: 原始上下文长度
        extended_length: 扩展后的上下文长度
        alpha: γ 函数的参数 α
        beta: γ 函数的参数 β

    Returns:
        original_theta: 原始的 θ_i 值
        modified_theta: NTK-by-parts 修改后的 θ_i 值
        wavelengths: 波长 λ_i
        gamma_values: γ(r(i)) 值
    """

    # 扩展倍数
    k = extended_length / original_length

    # 计算维度索引
    i_values = np.arange(0, d // 2)

    # 计算原始 θ_i
    original_theta = base ** (-2 * i_values / d)

    # 计算波长 λ_i = 2π/θ_i
    wavelengths = 2 * np.pi / original_theta

    # 计算 r(i) = L/λ_i
    r_values = original_length / wavelengths

    # 计算 γ(r) 函数
    gamma_values = np.zeros_like(r_values)
    for idx, r in enumerate(r_values):
        if r > beta:
            gamma_values[idx] = 0
        elif r < alpha:
            gamma_values[idx] = 1
        else:
            gamma_values[idx] = (r - beta) / (alpha - beta)

    # 计算修改后的 θ_i
    # h(θ_i) = (1 - γ(r(i))) * θ_i/k + γ(r(i)) * θ_i
    modified_theta = (
        1 - gamma_values
    ) * original_theta / k + gamma_values * original_theta

    return original_theta, modified_theta, wavelengths, gamma_values, i_values


def plot_ntk_by_parts_curves(d=128):
    """
    绘制 NTK-by-parts 的曲线图

    Args:
        d: 维度大小
    """

    # 计算 NTK-by-parts 的各个值
    original_theta, modified_theta, wavelengths, gamma_values, i_values = (
        compute_ntk_by_parts_theta(d)
    )

    # 创建更大的图表
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("NTK-by-parts RoPE Analysis (d=128, L=4096→163840)", fontsize=16)

    # 子图1: θ_i 值对比
    ax1 = axes[0, 0]
    ax1.semilogy(i_values, original_theta, "b-", label="Original θ_i", linewidth=2)
    ax1.semilogy(i_values, modified_theta, "r-", label="NTK-by-parts θ_i", linewidth=2)
    ax1.set_xlabel("Dimension Index i")
    ax1.set_ylabel("θ_i (log scale)")
    ax1.set_title("θ_i Values Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 波长 λ_i
    ax2 = axes[0, 1]
    ax2.semilogy(i_values, wavelengths, "g-", linewidth=2)
    ax2.axhline(
        y=4096,
        color="r",
        linestyle="--",
        alpha=0.7,
        label="Original Context Length (4096)",
    )
    ax2.axhline(
        y=163840,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label="Extended Context Length (163840)",
    )
    ax2.set_xlabel("Dimension Index i")
    ax2.set_ylabel("Wavelength λ_i (log scale)")
    ax2.set_title("Wavelength λ_i = 2π/θ_i")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3: γ(r(i)) 函数
    ax3 = axes[0, 2]
    r_values = 4096 / wavelengths
    ax3.plot(i_values, gamma_values, "purple", linewidth=2)
    ax3.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
    ax3.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax3.set_xlabel("Dimension Index i")
    ax3.set_ylabel("γ(r(i))")
    ax3.set_title("Interpolation Factor γ(r(i))")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)

    # 子图4: m*θ_i 值对比 - 不同位置
    ax4 = axes[1, 0]
    positions = [1024, 4096, 8192, 16384]
    colors = ["blue", "red", "green", "orange"]

    for pos, color in zip(positions, colors):
        original_m_theta = pos * original_theta
        modified_m_theta = pos * modified_theta

        ax4.semilogy(
            i_values,
            original_m_theta,
            color=color,
            linestyle="-",
            label=f"Original m*θ_i (m={pos})",
            linewidth=2,
            alpha=0.7,
        )
        ax4.semilogy(
            i_values,
            modified_m_theta,
            color=color,
            linestyle="--",
            label=f"NTK-by-parts m*θ_i (m={pos})",
            linewidth=2,
        )

    ax4.set_xlabel("Dimension Index i")
    ax4.set_ylabel("m*θ_i (log scale)")
    ax4.set_title("Rotation Angles m*θ_i at Different Positions")
    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax4.grid(True, alpha=0.3)

    # 子图5: 角度范围对比
    ax5 = axes[1, 1]
    position_range = np.logspace(2, 5, 50)  # 从100到100000

    # 选择几个代表性的维度
    selected_dims = [0, 16, 32, 48, 63]
    colors_dims = ["red", "blue", "green", "orange", "purple"]

    for dim_idx, color in zip(selected_dims, colors_dims):
        if dim_idx < len(original_theta):
            original_angles = position_range * original_theta[dim_idx]
            modified_angles = position_range * modified_theta[dim_idx]

            ax5.loglog(
                position_range,
                original_angles,
                color=color,
                linestyle="-",
                label=f"Original i={dim_idx}",
                linewidth=2,
                alpha=0.7,
            )
            ax5.loglog(
                position_range,
                modified_angles,
                color=color,
                linestyle="--",
                label=f"NTK-by-parts i={dim_idx}",
                linewidth=2,
            )

    ax5.set_xlabel("Position m")
    ax5.set_ylabel("m*θ_i")
    ax5.set_title("m*θ_i vs Position for Selected Dimensions")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 子图6: 相对变化量
    ax6 = axes[1, 2]
    relative_change = (modified_theta - original_theta) / original_theta * 100

    ax6.plot(i_values, relative_change, "darkred", linewidth=2)
    ax6.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax6.set_xlabel("Dimension Index i")
    ax6.set_ylabel("Relative Change (%)")
    ax6.set_title("Relative Change in θ_i\n(NTK-by-parts vs Original)")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ntk_by_parts_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 输出统计信息
    print(f"维度: {d}")
    print(f"原始上下文长度: 4096")
    print(f"扩展后上下文长度: 163840")
    print(f"扩展倍数: {163840/4096:.1f}")
    print(f"\n维度分类统计:")
    print(f"高频维度 (γ=1, 不插值): {np.sum(gamma_values == 1)} 个")
    print(f"低频维度 (γ=0, 完全插值): {np.sum(gamma_values == 0)} 个")
    print(
        f"中频维度 (0<γ<1, 部分插值): {np.sum((gamma_values > 0) & (gamma_values < 1))} 个"
    )

    # 输出具体的θ_i值
    print(f"\n部分 θ_i 值:")
    print(
        "i   |  Original θ_i  |  NTK-by-parts θ_i  |  Wavelength  |  γ(r(i))  |  Relative Change"
    )
    print("-" * 85)
    for i in [0, 8, 16, 24, 32, 40, 48, 56, 63]:
        if i < len(original_theta):
            rel_change = (
                (modified_theta[i] - original_theta[i]) / original_theta[i] * 100
            )
            print(
                f"{i:2d} |  {original_theta[i]:.3e}     |  {modified_theta[i]:.3e}       |  {wavelengths[i]:8.1f}  |  {gamma_values[i]:.3f}    |  {rel_change:+6.1f}%"
            )


def plot_detailed_m_theta_comparison():
    """
    绘制详细的 m*θ_i 对比图
    """
    d = 128
    original_theta, modified_theta, wavelengths, gamma_values, i_values = (
        compute_ntk_by_parts_theta(d)
    )

    # 创建专门的 m*θ_i 对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Detailed m*θ_i Analysis: Original RoPE vs NTK-by-parts", fontsize=16)

    # 不同位置的对比
    positions = [1024, 4096, 16384, 65536]

    for idx, pos in enumerate(positions):
        ax = axes[idx // 2, idx % 2]

        original_m_theta = pos * original_theta
        modified_m_theta = pos * modified_theta

        ax.semilogy(
            i_values, original_m_theta, "b-", label="Original RoPE", linewidth=2
        )
        ax.semilogy(i_values, modified_m_theta, "r-", label="NTK-by-parts", linewidth=2)

        # 添加重要的角度标记
        ax.axhline(y=np.pi, color="gray", linestyle="--", alpha=0.5, label="π")
        ax.axhline(y=2 * np.pi, color="gray", linestyle=":", alpha=0.5, label="2π")

        ax.set_xlabel("Dimension Index i")
        ax.set_ylabel("m*θ_i (log scale)")
        ax.set_title(f"Position m = {pos}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("detailed_m_theta_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # 绘制主要的分析图
    plot_ntk_by_parts_curves(d=128)

    # 绘制详细的 m*θ_i 对比图
    plot_detailed_m_theta_comparison()
