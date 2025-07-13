import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 参数设置
d = 128  # 维度大小
b = 10000  # 基地
max_position = 4096  # 最大位置

# 计算维度索引
i_values = np.arange(0, d // 2)  # i = 0, 1, 2, ..., d/2-1


def plot_theta_i():
    """绘制 theta_i = b^(-2i/d) 曲线"""
    theta_i = b ** (-2 * i_values / d)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 非log维度
    ax1.plot(i_values, theta_i, "b-", linewidth=2, marker="o", markersize=4)
    ax1.set_xlabel("维度索引 i")
    ax1.set_ylabel("θ_i")
    ax1.set_title("θ_i = b^(-2i/d) vs 维度索引 i")
    ax1.grid(True, alpha=0.3)

    # log维度
    ax2.semilogy(i_values, theta_i, "r-", linewidth=2, marker="s", markersize=4)
    ax2.set_xlabel("维度索引 i")
    ax2.set_ylabel("θ_i (log scale)")
    ax2.set_title("θ_i = b^(-2i/d) vs 维度索引 i (对数尺度)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("theta_i_curves.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_wavelength():
    """绘制波长 λ_i = 2π/θ_i 曲线"""
    theta_i = b ** (-2 * i_values / d)
    lambda_i = 2 * np.pi / theta_i

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 线性尺度
    ax1.plot(i_values, lambda_i, "g-", linewidth=2, marker="o", markersize=4)
    ax1.set_xlabel("维度索引 i")
    ax1.set_ylabel("波长 λ_i")
    ax1.set_title("波长 λ_i = 2π/θ_i vs 维度索引 i")
    ax1.grid(True, alpha=0.3)

    # 对数尺度
    ax2.semilogy(i_values, lambda_i, "purple", linewidth=2, marker="s", markersize=4)
    ax2.set_xlabel("维度索引 i")
    ax2.set_ylabel("波长 λ_i (log scale)")
    ax2.set_title("波长 λ_i = 2π/θ_i vs 维度索引 i (对数尺度)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("wavelength_curves.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_m_theta_i():
    """绘制 m * θ_i 曲线"""
    theta_i = b ** (-2 * i_values / d)
    m_values = np.arange(0, max_position)

    # 创建网格
    M, I = np.meshgrid(m_values, i_values)
    THETA = b ** (-2 * I / d)
    Z = M * THETA

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 2D热力图
    im1 = ax1.imshow(
        Z, aspect="auto", cmap="viridis", extent=[0, max_position, 0, d // 2]
    )
    ax1.set_xlabel("位置 m")
    ax1.set_ylabel("维度索引 i")
    ax1.set_title("m * θ_i 热力图")
    plt.colorbar(im1, ax=ax1, label="m * θ_i")

    # 选择几个特定维度绘制曲线
    selected_dims = [0, 8, 16, 24, 31]
    colors = ["red", "blue", "green", "orange", "purple"]

    for idx, dim in enumerate(selected_dims):
        if dim < len(theta_i):
            ax2.plot(
                m_values,
                m_values * theta_i[dim],
                color=colors[idx],
                linewidth=2,
                label=f"i={dim}",
            )

    ax2.set_xlabel("位置 m")
    ax2.set_ylabel("m * θ_i")
    ax2.set_title("不同维度下的 m * θ_i")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 选择几个特定位置绘制曲线
    selected_positions = [0, 512, 1024, 2048, 4095]
    colors2 = ["red", "blue", "green", "orange", "purple"]

    for idx, pos in enumerate(selected_positions):
        if pos < max_position:
            ax3.plot(
                i_values,
                pos * theta_i,
                color=colors2[idx],
                linewidth=2,
                label=f"m={pos}",
            )

    ax3.set_xlabel("维度索引 i")
    ax3.set_ylabel("m * θ_i")
    ax3.set_title("不同位置下的 m * θ_i")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 3D表面图
    ax4.remove()
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")

    # 为了更好的可视化，降采样
    m_sample = m_values[::64]  # 每64个点取一个
    i_sample = i_values[::2]  # 每2个维度取一个

    M_sample, I_sample = np.meshgrid(m_sample, i_sample)
    THETA_sample = b ** (-2 * I_sample / d)
    Z_sample = M_sample * THETA_sample

    surf = ax4.plot_surface(M_sample, I_sample, Z_sample, cmap="viridis", alpha=0.8)
    ax4.set_xlabel("位置 m")
    ax4.set_ylabel("维度索引 i")
    ax4.set_zlabel("m * θ_i")
    ax4.set_title("m * θ_i 3D表面图")

    plt.tight_layout()
    plt.savefig("m_theta_i_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_sin_m_theta_i():
    """绘制 sin(m * θ_i) 曲线"""
    theta_i = b ** (-2 * i_values / d)
    m_values = np.arange(0, max_position)

    # 创建网格
    M, I = np.meshgrid(m_values, i_values)
    THETA = b ** (-2 * I / d)
    Z = np.sin(M * THETA)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 2D热力图
    im1 = ax1.imshow(
        Z,
        aspect="auto",
        cmap="RdBu",
        vmin=-1,
        vmax=1,
        extent=[0, max_position, 0, d // 2],
    )
    ax1.set_xlabel("位置 m")
    ax1.set_ylabel("维度索引 i")
    ax1.set_title("sin(m * θ_i) 热力图")
    plt.colorbar(im1, ax=ax1, label="sin(m * θ_i)")

    # 选择几个特定维度绘制曲线
    selected_dims = [0, 8, 16, 24, 31]
    colors = ["red", "blue", "green", "orange", "purple"]

    for idx, dim in enumerate(selected_dims):
        if dim < len(theta_i):
            ax2.plot(
                m_values,
                np.sin(m_values * theta_i[dim]),
                color=colors[idx],
                linewidth=1.5,
                label=f"i={dim}",
            )

    ax2.set_xlabel("位置 m")
    ax2.set_ylabel("sin(m * θ_i)")
    ax2.set_title("不同维度下的 sin(m * θ_i)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)

    # 选择几个特定位置绘制曲线
    selected_positions = [0, 512, 1024, 2048, 4095]
    colors2 = ["red", "blue", "green", "orange", "purple"]

    for idx, pos in enumerate(selected_positions):
        if pos < max_position:
            ax3.plot(
                i_values,
                np.sin(pos * theta_i),
                color=colors2[idx],
                linewidth=2,
                label=f"m={pos}",
            )

    ax3.set_xlabel("维度索引 i")
    ax3.set_ylabel("sin(m * θ_i)")
    ax3.set_title("不同位置下的 sin(m * θ_i)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.1, 1.1)

    # 3D表面图
    ax4.remove()
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")

    # 为了更好的可视化，降采样
    m_sample = m_values[::64]  # 每64个点取一个
    i_sample = i_values[::2]  # 每2个维度取一个

    M_sample, I_sample = np.meshgrid(m_sample, i_sample)
    THETA_sample = b ** (-2 * I_sample / d)
    Z_sample = np.sin(M_sample * THETA_sample)

    surf = ax4.plot_surface(
        M_sample, I_sample, Z_sample, cmap="RdBu", alpha=0.8, vmin=-1, vmax=1
    )
    ax4.set_xlabel("位置 m")
    ax4.set_ylabel("维度索引 i")
    ax4.set_zlabel("sin(m * θ_i)")
    ax4.set_title("sin(m * θ_i) 3D表面图")
    ax4.set_zlim(-1.1, 1.1)

    plt.tight_layout()
    plt.savefig("sin_m_theta_i_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_rotation_periods():
    """绘制旋转周期变化图，展示不同维度在不同序列长度下的旋转周期性"""
    theta_i = b ** (-2 * i_values / d)
    m_values = np.arange(0, max_position)

    # 选择几个具有代表性的维度
    selected_dims = [0, 8, 16, 32, 63]
    colors = ["red", "blue", "green", "orange", "purple"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 不同维度的 sin(m * θ_i) 周期变化
    for idx, dim in enumerate(selected_dims):
        if dim < len(theta_i):
            ax1.plot(
                m_values,
                np.sin(m_values * theta_i[dim]),
                color=colors[idx],
                linewidth=1.5,
                label=f"i={dim}",
                alpha=0.8,
            )

    ax1.set_xlabel("序列位置 m")
    ax1.set_ylabel("sin(m * θ_i)")
    ax1.set_title("不同维度的 sin(m * θ_i) 旋转周期")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)

    # 2. 不同维度的 cos(m * θ_i) 周期变化
    for idx, dim in enumerate(selected_dims):
        if dim < len(theta_i):
            ax2.plot(
                m_values,
                np.cos(m_values * theta_i[dim]),
                color=colors[idx],
                linewidth=1.5,
                label=f"i={dim}",
                alpha=0.8,
            )

    ax2.set_xlabel("序列位置 m")
    ax2.set_ylabel("cos(m * θ_i)")
    ax2.set_title("不同维度的 cos(m * θ_i) 旋转周期")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)

    # 3. 旋转向量的轨迹图 (复数平面上的旋转)
    # 选择几个特定的维度来展示旋转轨迹
    trajectory_dims = [0, 8, 16, 32]
    trajectory_colors = ["red", "blue", "green", "orange"]

    for idx, dim in enumerate(trajectory_dims):
        if dim < len(theta_i):
            # 只显示前256个位置的轨迹，避免过于密集
            m_subset = m_values[:256]
            x = np.cos(m_subset * theta_i[dim])
            y = np.sin(m_subset * theta_i[dim])
            ax3.plot(
                x,
                y,
                color=trajectory_colors[idx],
                linewidth=1.5,
                label=f"i={dim}",
                alpha=0.7,
            )

    ax3.set_xlabel("cos(m * θ_i)")
    ax3.set_ylabel("sin(m * θ_i)")
    ax3.set_title("旋转向量轨迹图 (复数平面)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-1.1, 1.1)
    ax3.set_ylim(-1.1, 1.1)
    ax3.set_aspect("equal")

    # 4. 周期长度分析
    # 计算每个维度的周期长度
    periods = []
    dim_indices = []

    for dim in range(len(theta_i)):
        if theta_i[dim] > 0:
            # 周期长度 = 2π / θ_i
            period = 2 * np.pi / theta_i[dim]
            periods.append(period)
            dim_indices.append(dim)

    ax4.semilogy(dim_indices, periods, "bo-", linewidth=2, markersize=6)
    ax4.set_xlabel("维度索引 i")
    ax4.set_ylabel("周期长度 (log scale)")
    ax4.set_title("不同维度的旋转周期长度")
    ax4.grid(True, alpha=0.3)

    # 添加一些参考线
    ax4.axhline(
        y=max_position,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"最大序列长度 = {max_position}",
    )
    ax4.axhline(
        y=max_position / 2,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"序列长度/2 = {max_position//2}",
    )
    ax4.legend()

    plt.tight_layout()
    plt.savefig("rotation_periods.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_detailed_rotation_analysis():
    """详细的旋转分析图，包括局部放大和频谱分析"""
    theta_i = b ** (-2 * i_values / d)

    # 选择三个具有代表性的维度进行详细分析
    analysis_dims = [0, 16, 63]  # 高频、中频、低频
    dim_labels = ["高频 (i=0)", "中频 (i=16)", "低频 (i=63)"]
    colors = ["red", "blue", "green"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    for row, (dim, label, color) in enumerate(zip(analysis_dims, dim_labels, colors)):
        if dim < len(theta_i):
            # 创建三个不同长度的序列来展示周期性
            m_short = np.arange(0, 512)  # 短序列
            m_medium = np.arange(0, 1024)  # 中等序列
            m_long = np.arange(0, 2048)  # 长序列

            sequences = [m_short, m_medium, m_long]
            seq_labels = ["短序列 (0-512)", "中等序列 (0-1024)", "长序列 (0-2048)"]

            for col, (m_seq, seq_label) in enumerate(zip(sequences, seq_labels)):
                ax = axes[row, col]

                # 绘制 sin 曲线
                sin_values = np.sin(m_seq * theta_i[dim])

                ax.plot(
                    m_seq,
                    sin_values,
                    color=color,
                    linewidth=2,
                    label=f"sin(m*θ_{dim})",
                    alpha=0.8,
                )

                ax.set_xlabel("序列位置 m")
                ax.set_ylabel("旋转值")
                ax.set_title(f"{label} - {seq_label}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-1.1, 1.1)

                # 添加周期标记
                if theta_i[dim] > 0:
                    period = 2 * np.pi / theta_i[dim]
                    if period < len(m_seq):
                        for p in np.arange(period, len(m_seq), period):
                            ax.axvline(x=p, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("detailed_rotation_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_theta_statistics():
    """打印 theta_i 的统计信息"""
    theta_i = b ** (-2 * i_values / d)
    lambda_i = 2 * np.pi / theta_i

    print("RoPE 参数统计信息:")
    print(f"维度 d = {d}")
    print(f"基地 b = {b}")
    print(f"最大位置 = {max_position}")
    print("\n维度索引 i\t\tθ_i\t\t\t波长 λ_i\t\t在{0}位置内的频率".format(max_position))
    print("-" * 80)

    for i in range(0, len(i_values), 4):  # 每4个维度打印一次
        theta = theta_i[i]
        wavelength = lambda_i[i]
        freq_in_context = max_position / wavelength
        print(f"{i}\t\t{theta:.2e}\t\t{wavelength:.1f}\t\t{freq_in_context:.2f}")


def main():
    """主函数"""
    print("开始绘制 RoPE 可视化图...")

    # 打印统计信息
    print_theta_statistics()

    # 绘制各种曲线
    print("\n1. 绘制 θ_i 曲线...")
    plot_theta_i()

    print("2. 绘制波长曲线...")
    plot_wavelength()

    print("3. 绘制 m * θ_i 可视化...")
    plot_m_theta_i()

    print("4. 绘制 sin(m * θ_i) 可视化...")
    plot_sin_m_theta_i()

    print("5. 绘制旋转周期变化图...")
    plot_rotation_periods()

    print("6. 绘制详细的旋转分析图...")
    plot_detailed_rotation_analysis()

    print("\n所有图像已保存完成!")


if __name__ == "__main__":
    main()
