import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

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

# 颜色配置
colors = {
    "original": "#1f77b4",
    "pi": "#ff7f0e",
    "ntk_aware": "#2ca02c",
    "ntk_by_parts": "#d62728",
}


def compute_original_rope(i_values, base, d):
    """计算原始RoPE的theta_i"""
    theta_i = base ** (-2 * i_values / d)
    return theta_i


def compute_wavelength(theta_i):
    """计算波长"""
    return 2 * np.pi / theta_i


def compute_frequency_ratio(L, wavelength):
    """计算频率比 r(i) = L / λ_i"""
    return L / wavelength


def compute_gamma(r, alpha, beta):
    """计算γ(r)函数"""
    gamma = np.zeros_like(r)
    gamma[r > beta] = 0
    gamma[r < alpha] = 1
    mask = (r >= alpha) & (r <= beta)
    gamma[mask] = (r[mask] - beta) / (alpha - beta)
    return gamma


def compute_ntk_aware_theta(i_values, base, d, k):
    """计算NTK-aware的theta_i"""
    new_base = base * (k ** (d / (d - 2)))
    theta_i = new_base ** (-2 * i_values / d)
    return theta_i


def compute_ntk_by_parts_theta(theta_i_orig, gamma, k):
    """计算NTK-by-parts的theta_i"""
    theta_i_new = (1 - gamma) * theta_i_orig / k + gamma * theta_i_orig
    return theta_i_new


def compute_position_interpolation_theta(theta_i_orig):
    """位置插值保持theta_i不变，只改变位置"""
    return theta_i_orig


def generate_theta_wavelength_comparison(i_values, theta_values, colors):
    """生成θ_i和波长对比图（包含log和非log版本）"""
    theta_orig, theta_pi, theta_ntk_aware, theta_ntk_by_parts = theta_values

    fig1 = plt.figure(figsize=(16, 12))
    gs1 = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    # 1.1 θ_i 对比 (log scale)
    ax1_1 = fig1.add_subplot(gs1[0, 0])
    ax1_1.semilogy(
        i_values,
        theta_orig,
        "o-",
        color=colors["original"],
        label="Original RoPE",
        markersize=3,
    )
    ax1_1.semilogy(
        i_values,
        theta_pi,
        "s-",
        color=colors["pi"],
        label="Position Interpolation",
        markersize=3,
    )
    ax1_1.semilogy(
        i_values,
        theta_ntk_aware,
        "^-",
        color=colors["ntk_aware"],
        label="NTK-aware",
        markersize=3,
    )
    ax1_1.semilogy(
        i_values,
        theta_ntk_by_parts,
        "v-",
        color=colors["ntk_by_parts"],
        label="NTK-by-parts",
        markersize=3,
    )
    ax1_1.set_xlabel("维度索引 i")
    ax1_1.set_ylabel("θ_i (log scale)")
    ax1_1.set_title("θ_i 对比 (对数尺度)")
    ax1_1.legend()
    ax1_1.grid(True, alpha=0.3)

    # 1.2 θ_i 对比 (linear scale)
    ax1_2 = fig1.add_subplot(gs1[0, 1])
    ax1_2.plot(
        i_values,
        theta_orig,
        "o-",
        color=colors["original"],
        label="Original RoPE",
        markersize=3,
    )
    ax1_2.plot(
        i_values,
        theta_pi,
        "s-",
        color=colors["pi"],
        label="Position Interpolation",
        markersize=3,
    )
    ax1_2.plot(
        i_values,
        theta_ntk_aware,
        "^-",
        color=colors["ntk_aware"],
        label="NTK-aware",
        markersize=3,
    )
    ax1_2.plot(
        i_values,
        theta_ntk_by_parts,
        "v-",
        color=colors["ntk_by_parts"],
        label="NTK-by-parts",
        markersize=3,
    )
    ax1_2.set_xlabel("维度索引 i")
    ax1_2.set_ylabel("θ_i (linear scale)")
    ax1_2.set_title("θ_i 对比 (线性尺度)")
    ax1_2.legend()
    ax1_2.grid(True, alpha=0.3)

    # 计算波长
    wavelength_orig = compute_wavelength(theta_orig)
    wavelength_pi = compute_wavelength(theta_pi)
    wavelength_ntk_aware = compute_wavelength(theta_ntk_aware)
    wavelength_ntk_by_parts = compute_wavelength(theta_ntk_by_parts)

    # 1.3 波长对比 (log scale)
    ax1_3 = fig1.add_subplot(gs1[1, 0])
    ax1_3.semilogy(
        i_values,
        wavelength_orig,
        "o-",
        color=colors["original"],
        label="Original RoPE",
        markersize=3,
    )
    ax1_3.semilogy(
        i_values,
        wavelength_pi,
        "s-",
        color=colors["pi"],
        label="Position Interpolation",
        markersize=3,
    )
    ax1_3.semilogy(
        i_values,
        wavelength_ntk_aware,
        "^-",
        color=colors["ntk_aware"],
        label="NTK-aware",
        markersize=3,
    )
    ax1_3.semilogy(
        i_values,
        wavelength_ntk_by_parts,
        "v-",
        color=colors["ntk_by_parts"],
        label="NTK-by-parts",
        markersize=3,
    )
    ax1_3.axhline(
        y=L_orig, color="red", linestyle="--", alpha=0.7, label=f"原始长度 {L_orig}"
    )
    ax1_3.axhline(
        y=L_ext, color="purple", linestyle="--", alpha=0.7, label=f"扩展长度 {L_ext}"
    )
    ax1_3.set_xlabel("维度索引 i")
    ax1_3.set_ylabel("波长 λ_i (log scale)")
    ax1_3.set_title("波长对比 (对数尺度)")
    ax1_3.legend()
    ax1_3.grid(True, alpha=0.3)

    # 1.4 波长对比 (linear scale)
    ax1_4 = fig1.add_subplot(gs1[1, 1])
    ax1_4.plot(
        i_values,
        wavelength_orig,
        "o-",
        color=colors["original"],
        label="Original RoPE",
        markersize=3,
    )
    ax1_4.plot(
        i_values,
        wavelength_pi,
        "s-",
        color=colors["pi"],
        label="Position Interpolation",
        markersize=3,
    )
    ax1_4.plot(
        i_values,
        wavelength_ntk_aware,
        "^-",
        color=colors["ntk_aware"],
        label="NTK-aware",
        markersize=3,
    )
    ax1_4.plot(
        i_values,
        wavelength_ntk_by_parts,
        "v-",
        color=colors["ntk_by_parts"],
        label="NTK-by-parts",
        markersize=3,
    )
    ax1_4.axhline(
        y=L_orig, color="red", linestyle="--", alpha=0.7, label=f"原始长度 {L_orig}"
    )
    ax1_4.axhline(
        y=L_ext, color="purple", linestyle="--", alpha=0.7, label=f"扩展长度 {L_ext}"
    )
    ax1_4.set_xlabel("维度索引 i")
    ax1_4.set_ylabel("波长 λ_i (linear scale)")
    ax1_4.set_title("波长对比 (线性尺度)")
    ax1_4.legend()
    ax1_4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("theta_wavelength_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def generate_frequency_ratio_gamma(i_values, r_orig, gamma_values, alpha, beta):
    """生成频率比r(i)和γ(r)以及NTK-by-parts维度分类图"""
    fig2 = plt.figure(figsize=(16, 12))
    gs2 = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    # 2.1 频率比 r(i) 和 γ(r) (对数尺度)
    ax2_1 = fig2.add_subplot(gs2[0, 0])
    ax2_1_twin = ax2_1.twinx()

    # 绘制r(i)
    line1 = ax2_1.plot(
        i_values, r_orig, "o-", color="blue", label="r(i) = L/λ_i", markersize=4
    )
    ax2_1.axhline(
        y=alpha, color="green", linestyle="--", alpha=0.7, label=f"α = {alpha}"
    )
    ax2_1.axhline(y=beta, color="red", linestyle="--", alpha=0.7, label=f"β = {beta}")
    ax2_1.set_xlabel("维度索引 i")
    ax2_1.set_ylabel("频率比 r(i)", color="blue")
    ax2_1.set_yscale("log")
    ax2_1.tick_params(axis="y", labelcolor="blue")

    # 绘制γ(r)
    line2 = ax2_1_twin.plot(
        i_values, gamma_values, "s-", color="red", label="γ(r)", markersize=4
    )
    ax2_1_twin.set_ylabel("γ(r)", color="red")
    ax2_1_twin.set_ylim(-0.1, 1.1)
    ax2_1_twin.tick_params(axis="y", labelcolor="red")

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2_1.legend(
        lines
        + [
            ax2_1.axhline(y=alpha, color="green", linestyle="--", alpha=0.7),
            ax2_1.axhline(y=beta, color="red", linestyle="--", alpha=0.7),
        ],
        labels + [f"α = {alpha}", f"β = {beta}"],
        loc="center right",
    )
    ax2_1.set_title("频率比 r(i) 和 γ(r) 函数 (对数尺度)", fontsize=12)
    ax2_1.grid(True, alpha=0.3)

    # 2.2 频率比 r(i) 和 γ(r) (线性尺度)
    ax2_2 = fig2.add_subplot(gs2[0, 1])
    ax2_2_twin = ax2_2.twinx()

    # 绘制r(i) - 线性尺度
    line1_linear = ax2_2.plot(
        i_values, r_orig, "o-", color="blue", label="r(i) = L/λ_i", markersize=4
    )
    ax2_2.axhline(
        y=alpha, color="green", linestyle="--", alpha=0.7, label=f"α = {alpha}"
    )
    ax2_2.axhline(y=beta, color="red", linestyle="--", alpha=0.7, label=f"β = {beta}")
    ax2_2.set_xlabel("维度索引 i")
    ax2_2.set_ylabel("频率比 r(i)", color="blue")
    ax2_2.tick_params(axis="y", labelcolor="blue")

    # 绘制γ(r) - 线性尺度
    line2_linear = ax2_2_twin.plot(
        i_values, gamma_values, "s-", color="red", label="γ(r)", markersize=4
    )
    ax2_2_twin.set_ylabel("γ(r)", color="red")
    ax2_2_twin.set_ylim(-0.1, 1.1)
    ax2_2_twin.tick_params(axis="y", labelcolor="red")

    # 合并图例 - 线性尺度
    lines_linear = line1_linear + line2_linear
    labels_linear = [l.get_label() for l in lines_linear]
    ax2_2.legend(
        lines_linear
        + [
            ax2_2.axhline(y=alpha, color="green", linestyle="--", alpha=0.7),
            ax2_2.axhline(y=beta, color="red", linestyle="--", alpha=0.7),
        ],
        labels_linear + [f"α = {alpha}", f"β = {beta}"],
        loc="center right",
    )
    ax2_2.set_title("频率比 r(i) 和 γ(r) 函数 (线性尺度)", fontsize=12)
    ax2_2.grid(True, alpha=0.3)

    # 2.3 NTK-by-parts 维度分类可视化
    ax2_3 = fig2.add_subplot(gs2[1, 0])
    dim_categories = np.zeros(len(i_values))
    dim_categories[r_orig > beta] = 0  # 不插值
    dim_categories[(r_orig >= alpha) & (r_orig <= beta)] = 1  # 部分插值
    dim_categories[r_orig < alpha] = 2  # 完全插值

    # 定义颜色和标签的映射关系
    category_mapping = {
        0: {"color": "#ff4444", "label": "不插值 (r>β)"},
        1: {"color": "#ffaa44", "label": "部分插值 (α≤r≤β)"},
        2: {"color": "#44ff44", "label": "完全插值 (r<α)"},
    }

    # 为每个维度分配颜色
    bar_colors = [category_mapping[int(cat)]["color"] for cat in dim_categories]

    bars = ax2_3.bar(i_values, np.ones(len(i_values)), color=bar_colors)
    ax2_3.set_xlabel("维度索引 i")
    ax2_3.set_ylabel("维度分类")
    ax2_3.set_title("NTK-by-parts 维度分类")

    # 创建图例，确保颜色和标签一致
    legend_elements = []
    for cat_id in sorted(category_mapping.keys()):
        legend_elements.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=category_mapping[cat_id]["color"],
                label=category_mapping[cat_id]["label"],
            )
        )
    ax2_3.legend(handles=legend_elements)

    # 2.4 维度分类统计饼图
    ax2_4 = fig2.add_subplot(gs2[1, 1])
    no_interp = np.sum(r_orig > beta)
    partial_interp = np.sum((r_orig >= alpha) & (r_orig <= beta))
    full_interp = np.sum(r_orig < alpha)

    sizes = [no_interp, partial_interp, full_interp]
    labels_pie = [
        f"不插值\n{no_interp}个({no_interp/len(i_values)*100:.1f}%)",
        f"部分插值\n{partial_interp}个({partial_interp/len(i_values)*100:.1f}%)",
        f"完全插值\n{full_interp}个({full_interp/len(i_values)*100:.1f}%)",
    ]

    # 使用相同的颜色映射
    pie_colors = [
        category_mapping[0]["color"],
        category_mapping[1]["color"],
        category_mapping[2]["color"],
    ]
    ax2_4.pie(sizes, labels=labels_pie, colors=pie_colors, autopct="", startangle=90)
    ax2_4.set_title("维度分类统计")

    plt.tight_layout()
    plt.savefig("frequency_ratio_gamma.png", dpi=300, bbox_inches="tight")
    plt.show()


def generate_other_analyses(i_values, theta_values, colors):
    """生成其他分析图表（旋转角度对比）"""
    theta_orig, theta_pi, theta_ntk_aware, theta_ntk_by_parts = theta_values

    fig3 = plt.figure(figsize=(12, 8))

    # 3.1 不同位置的旋转角度 m·θ_i (选择几个典型位置)
    ax3_1 = fig3.add_subplot(111)
    positions = [1024, 2048, 4096, 8192, 16384]  # 去掉位置0，避免重叠
    line_styles = ["-", "--", ":", "-.", "--"]  # 不同位置使用不同线型
    alphas = [0.7, 0.8, 0.9, 1.0, 0.8]  # 不同位置使用不同透明度

    # 先绘制所有位置的线条
    for i, pos in enumerate(positions):
        if pos <= L_ext:
            # 原始RoPE
            angles_orig = pos * theta_orig
            # 位置插值
            pos_pi = pos * L_orig / L_ext  # 插值后的位置
            angles_pi = pos_pi * theta_pi
            # NTK-aware
            angles_ntk_aware = pos * theta_ntk_aware
            # NTK-by-parts
            angles_ntk_by_parts = pos * theta_ntk_by_parts

            # 使用不同的线型和透明度来区分位置
            ax3_1.plot(
                i_values,
                angles_orig,
                line_styles[i],
                color=colors["original"],
                alpha=alphas[i],
                linewidth=2,
            )
            ax3_1.plot(
                i_values,
                angles_pi,
                line_styles[i],
                color=colors["pi"],
                alpha=alphas[i],
                linewidth=2,
            )
            ax3_1.plot(
                i_values,
                angles_ntk_aware,
                line_styles[i],
                color=colors["ntk_aware"],
                alpha=alphas[i],
                linewidth=2,
            )
            ax3_1.plot(
                i_values,
                angles_ntk_by_parts,
                line_styles[i],
                color=colors["ntk_by_parts"],
                alpha=alphas[i],
                linewidth=2,
            )

    # 创建图例 - 分别为方法和位置创建图例
    # 方法图例
    method_lines = []
    method_lines.append(
        plt.Line2D(
            [0], [0], color=colors["original"], linewidth=2, label="Original RoPE"
        )
    )
    method_lines.append(
        plt.Line2D(
            [0], [0], color=colors["pi"], linewidth=2, label="Position Interpolation"
        )
    )
    method_lines.append(
        plt.Line2D([0], [0], color=colors["ntk_aware"], linewidth=2, label="NTK-aware")
    )
    method_lines.append(
        plt.Line2D(
            [0], [0], color=colors["ntk_by_parts"], linewidth=2, label="NTK-by-parts"
        )
    )

    # 位置图例
    position_lines = []
    for i, pos in enumerate(positions):
        if pos <= L_ext:
            position_lines.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="gray",
                    linestyle=line_styles[i],
                    alpha=alphas[i],
                    linewidth=2,
                    label=f"pos={pos}",
                )
            )

    # 创建两个图例
    legend1 = ax3_1.legend(handles=method_lines, loc="upper left", title="方法")
    legend2 = ax3_1.legend(handles=position_lines, loc="upper right", title="位置")
    ax3_1.add_artist(legend1)  # 添加第一个图例

    # 添加位置标注在右侧
    for i, pos in enumerate(positions):
        if pos <= L_ext and i < 4:  # 只标注前4个位置，避免过于拥挤
            y_pos = pos * theta_orig[-8]  # 使用倒数第8个维度的角度
            ax3_1.annotate(
                f"{pos}",
                xy=(len(i_values) - 6, y_pos),
                xytext=(len(i_values) - 3, y_pos),
                fontsize=9,
                alpha=0.8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

    ax3_1.set_xlabel("维度索引 i")
    ax3_1.set_ylabel("旋转角度 m·θ_i")
    ax3_1.set_title("不同位置的旋转角度对比")
    ax3_1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("other_analyses.png", dpi=300, bbox_inches="tight")
    plt.show()


def compute_angle_matrix(positions, theta_values):
    """计算角度矩阵 [positions, dimensions]"""
    angle_matrix = np.zeros((len(positions), len(theta_values)))
    for i, pos in enumerate(positions):
        angle_matrix[i, :] = pos * theta_values
    return angle_matrix


def compute_period_matrix(positions, theta_values):
    """计算周期矩阵 [positions, dimensions] - 角度除以2π得到周期数"""
    angle_matrix = compute_angle_matrix(positions, theta_values)
    period_matrix = angle_matrix / (2 * np.pi)
    return period_matrix


def generate_3d_heatmap_analysis(i_values, theta_values, colors):
    """生成m·θ_i的三维图和热力图对比"""
    theta_orig, theta_pi, theta_ntk_aware, theta_ntk_by_parts = theta_values

    fig4 = plt.figure(figsize=(20, 12))
    gs4 = gridspec.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

    # 创建位置和维度的网格
    positions_3d = np.arange(0, L_ext, 1024)  # 每1024个位置取样
    positions_3d = positions_3d[positions_3d <= L_ext][:20]  # 限制在20个位置以内
    i_mesh, m_mesh = np.meshgrid(i_values, positions_3d)

    # 计算各方法的角度矩阵
    angles_orig_matrix = compute_angle_matrix(positions_3d, theta_orig)
    angles_pi_matrix = compute_angle_matrix(positions_3d, theta_pi)
    angles_ntk_aware_matrix = compute_angle_matrix(positions_3d, theta_ntk_aware)
    angles_ntk_by_parts_matrix = compute_angle_matrix(positions_3d, theta_ntk_by_parts)

    # 4.1 Original RoPE 三维图
    ax4_1 = fig4.add_subplot(gs4[0, 0], projection="3d")
    surf1 = ax4_1.plot_surface(
        i_mesh,
        m_mesh,
        angles_orig_matrix,
        cmap="viridis",
        alpha=0.7,
        linewidth=0,
        antialiased=True,
    )
    ax4_1.set_xlabel("维度索引 i")
    ax4_1.set_ylabel("位置 m")
    ax4_1.set_zlabel("旋转角度 m·θ_i")
    ax4_1.set_title("Original RoPE", fontsize=12)
    ax4_1.view_init(elev=30, azim=45)

    # 4.2 Position Interpolation 三维图
    ax4_2 = fig4.add_subplot(gs4[0, 1], projection="3d")
    # 计算位置插值的实际位置
    positions_pi_3d = positions_3d * L_orig / L_ext
    angles_pi_matrix_actual = compute_angle_matrix(positions_pi_3d, theta_pi)
    surf2 = ax4_2.plot_surface(
        i_mesh,
        m_mesh,
        angles_pi_matrix_actual,
        cmap="plasma",
        alpha=0.7,
        linewidth=0,
        antialiased=True,
    )
    ax4_2.set_xlabel("维度索引 i")
    ax4_2.set_ylabel("位置 m")
    ax4_2.set_zlabel("旋转角度 m·θ_i")
    ax4_2.set_title("Position Interpolation", fontsize=12)
    ax4_2.view_init(elev=30, azim=45)

    # 4.3 NTK-aware 三维图
    ax4_3 = fig4.add_subplot(gs4[0, 2], projection="3d")
    surf3 = ax4_3.plot_surface(
        i_mesh,
        m_mesh,
        angles_ntk_aware_matrix,
        cmap="coolwarm",
        alpha=0.7,
        linewidth=0,
        antialiased=True,
    )
    ax4_3.set_xlabel("维度索引 i")
    ax4_3.set_ylabel("位置 m")
    ax4_3.set_zlabel("旋转角度 m·θ_i")
    ax4_3.set_title("NTK-aware", fontsize=12)
    ax4_3.view_init(elev=30, azim=45)

    # 4.4 NTK-by-parts 三维图
    ax4_4 = fig4.add_subplot(gs4[0, 3], projection="3d")
    surf4 = ax4_4.plot_surface(
        i_mesh,
        m_mesh,
        angles_ntk_by_parts_matrix,
        cmap="inferno",
        alpha=0.7,
        linewidth=0,
        antialiased=True,
    )
    ax4_4.set_xlabel("维度索引 i")
    ax4_4.set_ylabel("位置 m")
    ax4_4.set_zlabel("旋转角度 m·θ_i")
    ax4_4.set_title("NTK-by-parts", fontsize=12)
    ax4_4.view_init(elev=30, azim=45)

    # 4.5 Original RoPE 热力图
    ax4_5 = fig4.add_subplot(gs4[1, 0])
    im1 = ax4_5.imshow(
        angles_orig_matrix,
        cmap="viridis",
        aspect="auto",
        extent=[i_values[0], i_values[-1], positions_3d[0], positions_3d[-1]],
    )
    ax4_5.set_xlabel("维度索引 i")
    ax4_5.set_ylabel("位置 m")
    ax4_5.set_title("Original RoPE (热力图)", fontsize=12)
    plt.colorbar(im1, ax=ax4_5, shrink=0.6)

    # 4.6 Position Interpolation 热力图
    ax4_6 = fig4.add_subplot(gs4[1, 1])
    im2 = ax4_6.imshow(
        angles_pi_matrix_actual,
        cmap="plasma",
        aspect="auto",
        extent=[i_values[0], i_values[-1], positions_3d[0], positions_3d[-1]],
    )
    ax4_6.set_xlabel("维度索引 i")
    ax4_6.set_ylabel("位置 m")
    ax4_6.set_title("Position Interpolation (热力图)", fontsize=12)
    plt.colorbar(im2, ax=ax4_6, shrink=0.6)

    # 4.7 NTK-aware 热力图
    ax4_7 = fig4.add_subplot(gs4[1, 2])
    im3 = ax4_7.imshow(
        angles_ntk_aware_matrix,
        cmap="coolwarm",
        aspect="auto",
        extent=[i_values[0], i_values[-1], positions_3d[0], positions_3d[-1]],
    )
    ax4_7.set_xlabel("维度索引 i")
    ax4_7.set_ylabel("位置 m")
    ax4_7.set_title("NTK-aware (热力图)", fontsize=12)
    plt.colorbar(im3, ax=ax4_7, shrink=0.6)

    # 4.8 NTK-by-parts 热力图
    ax4_8 = fig4.add_subplot(gs4[1, 3])
    im4 = ax4_8.imshow(
        angles_ntk_by_parts_matrix,
        cmap="inferno",
        aspect="auto",
        extent=[i_values[0], i_values[-1], positions_3d[0], positions_3d[-1]],
    )
    ax4_8.set_xlabel("维度索引 i")
    ax4_8.set_ylabel("位置 m")
    ax4_8.set_title("NTK-by-parts (热力图)", fontsize=12)
    plt.colorbar(im4, ax=ax4_8, shrink=0.6)

    plt.tight_layout()
    plt.savefig("rotation_angle_3d_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()


def generate_period_comparison_3d(i_values, theta_values, colors):
    """生成m·θ_i周期对比的3D图"""
    theta_orig, theta_pi, theta_ntk_aware, theta_ntk_by_parts = theta_values

    fig5 = plt.figure(figsize=(20, 12))
    gs5 = gridspec.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

    # 创建位置和维度的网格
    positions_3d = np.arange(0, L_ext, 1024)  # 每1024个位置取样
    positions_3d = positions_3d[positions_3d <= L_ext][:20]  # 限制在20个位置以内
    i_mesh, m_mesh = np.meshgrid(i_values, positions_3d)

    # 计算各方法的周期矩阵
    periods_orig_matrix = compute_period_matrix(positions_3d, theta_orig)
    periods_pi_matrix = compute_period_matrix(positions_3d, theta_pi)
    periods_ntk_aware_matrix = compute_period_matrix(positions_3d, theta_ntk_aware)
    periods_ntk_by_parts_matrix = compute_period_matrix(
        positions_3d, theta_ntk_by_parts
    )

    # 5.1 Original RoPE 周期三维图
    ax5_1 = fig5.add_subplot(gs5[0, 0], projection="3d")
    surf1 = ax5_1.plot_surface(
        i_mesh,
        m_mesh,
        periods_orig_matrix,
        cmap="viridis",
        alpha=0.7,
        linewidth=0,
        antialiased=True,
    )
    ax5_1.set_xlabel("维度索引 i")
    ax5_1.set_ylabel("位置 m")
    ax5_1.set_zlabel("周期数 m·θ_i/(2π)")
    ax5_1.set_title("Original RoPE (周期)", fontsize=12)
    ax5_1.view_init(elev=30, azim=45)

    # 5.2 Position Interpolation 周期三维图
    ax5_2 = fig5.add_subplot(gs5[0, 1], projection="3d")
    # 计算位置插值的实际位置周期
    positions_pi_3d = positions_3d * L_orig / L_ext
    periods_pi_matrix_actual = compute_period_matrix(positions_pi_3d, theta_pi)
    surf2 = ax5_2.plot_surface(
        i_mesh,
        m_mesh,
        periods_pi_matrix_actual,
        cmap="plasma",
        alpha=0.7,
        linewidth=0,
        antialiased=True,
    )
    ax5_2.set_xlabel("维度索引 i")
    ax5_2.set_ylabel("位置 m")
    ax5_2.set_zlabel("周期数 m·θ_i/(2π)")
    ax5_2.set_title("Position Interpolation (周期)", fontsize=12)
    ax5_2.view_init(elev=30, azim=45)

    # 5.3 NTK-aware 周期三维图
    ax5_3 = fig5.add_subplot(gs5[0, 2], projection="3d")
    surf3 = ax5_3.plot_surface(
        i_mesh,
        m_mesh,
        periods_ntk_aware_matrix,
        cmap="coolwarm",
        alpha=0.7,
        linewidth=0,
        antialiased=True,
    )
    ax5_3.set_xlabel("维度索引 i")
    ax5_3.set_ylabel("位置 m")
    ax5_3.set_zlabel("周期数 m·θ_i/(2π)")
    ax5_3.set_title("NTK-aware (周期)", fontsize=12)
    ax5_3.view_init(elev=30, azim=45)

    # 5.4 NTK-by-parts 周期三维图
    ax5_4 = fig5.add_subplot(gs5[0, 3], projection="3d")
    surf4 = ax5_4.plot_surface(
        i_mesh,
        m_mesh,
        periods_ntk_by_parts_matrix,
        cmap="inferno",
        alpha=0.7,
        linewidth=0,
        antialiased=True,
    )
    ax5_4.set_xlabel("维度索引 i")
    ax5_4.set_ylabel("位置 m")
    ax5_4.set_zlabel("周期数 m·θ_i/(2π)")
    ax5_4.set_title("NTK-by-parts (周期)", fontsize=12)
    ax5_4.view_init(elev=30, azim=45)

    # 5.5 Original RoPE 周期热力图
    ax5_5 = fig5.add_subplot(gs5[1, 0])
    im1 = ax5_5.imshow(
        periods_orig_matrix,
        cmap="viridis",
        aspect="auto",
        extent=[i_values[0], i_values[-1], positions_3d[0], positions_3d[-1]],
    )
    ax5_5.set_xlabel("维度索引 i")
    ax5_5.set_ylabel("位置 m")
    ax5_5.set_title("Original RoPE (周期热力图)", fontsize=12)
    plt.colorbar(im1, ax=ax5_5, shrink=0.6, label="周期数")

    # 5.6 Position Interpolation 周期热力图
    ax5_6 = fig5.add_subplot(gs5[1, 1])
    im2 = ax5_6.imshow(
        periods_pi_matrix_actual,
        cmap="plasma",
        aspect="auto",
        extent=[i_values[0], i_values[-1], positions_3d[0], positions_3d[-1]],
    )
    ax5_6.set_xlabel("维度索引 i")
    ax5_6.set_ylabel("位置 m")
    ax5_6.set_title("Position Interpolation (周期热力图)", fontsize=12)
    plt.colorbar(im2, ax=ax5_6, shrink=0.6, label="周期数")

    # 5.7 NTK-aware 周期热力图
    ax5_7 = fig5.add_subplot(gs5[1, 2])
    im3 = ax5_7.imshow(
        periods_ntk_aware_matrix,
        cmap="coolwarm",
        aspect="auto",
        extent=[i_values[0], i_values[-1], positions_3d[0], positions_3d[-1]],
    )
    ax5_7.set_xlabel("维度索引 i")
    ax5_7.set_ylabel("位置 m")
    ax5_7.set_title("NTK-aware (周期热力图)", fontsize=12)
    plt.colorbar(im3, ax=ax5_7, shrink=0.6, label="周期数")

    # 5.8 NTK-by-parts 周期热力图
    ax5_8 = fig5.add_subplot(gs5[1, 3])
    im4 = ax5_8.imshow(
        periods_ntk_by_parts_matrix,
        cmap="inferno",
        aspect="auto",
        extent=[i_values[0], i_values[-1], positions_3d[0], positions_3d[-1]],
    )
    ax5_8.set_xlabel("维度索引 i")
    ax5_8.set_ylabel("位置 m")
    ax5_8.set_title("NTK-by-parts (周期热力图)", fontsize=12)
    plt.colorbar(im4, ax=ax5_8, shrink=0.6, label="周期数")

    plt.tight_layout()
    plt.savefig("rotation_period_3d_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_analysis_results(i_values, r_orig, gamma_values, theta_orig, wavelength_orig):
    """打印分析结果统计信息"""
    print("=== NTK-by-parts 分析结果 ===")
    print(f"总维度数: {d}")
    print(f"扩展倍数: {k}")
    print(f"原始上下文长度: {L_orig}")
    print(f"扩展后上下文长度: {L_ext}")
    print(f"α = {alpha}, β = {beta}")
    print()

    # 统计各类别的维度数量
    no_interp = np.sum(r_orig > beta)
    partial_interp = np.sum((r_orig >= alpha) & (r_orig <= beta))
    full_interp = np.sum(r_orig < alpha)

    print(f"维度分类统计:")
    print(f"  不插值 (r > β): {no_interp} 个维度 ({no_interp/len(i_values)*100:.1f}%)")
    print(
        f"  部分插值 (α ≤ r ≤ β): {partial_interp} 个维度 ({partial_interp/len(i_values)*100:.1f}%)"
    )
    print(
        f"  完全插值 (r < α): {full_interp} 个维度 ({full_interp/len(i_values)*100:.1f}%)"
    )
    print()

    # 输出一些关键维度的信息
    print("关键维度信息:")
    key_dims = [0, 8, 16, 24, 32, 40, 48, 56, 63]
    for i in key_dims:
        if i < len(i_values):
            print(
                f"  维度 {i}: θ_i={theta_orig[i]:.2e}, 波长={wavelength_orig[i]:.1f}, r={r_orig[i]:.2f}, γ={gamma_values[i]:.2f}"
            )


def main():
    """主函数：计算参数并生成所有图像"""
    # 计算各种方法的theta_i
    theta_orig = compute_original_rope(i_values, base, d)
    theta_ntk_aware = compute_ntk_aware_theta(i_values, base, d, k)
    theta_pi = compute_position_interpolation_theta(theta_orig)

    # 计算波长和频率比
    wavelength_orig = compute_wavelength(theta_orig)
    r_orig = compute_frequency_ratio(L_orig, wavelength_orig)
    gamma_values = compute_gamma(r_orig, alpha, beta)

    # 计算NTK-by-parts的theta_i
    theta_ntk_by_parts = compute_ntk_by_parts_theta(theta_orig, gamma_values, k)

    # 组织theta值
    theta_values = (theta_orig, theta_pi, theta_ntk_aware, theta_ntk_by_parts)

    # 生成所有图像
    generate_theta_wavelength_comparison(i_values, theta_values, colors)
    generate_frequency_ratio_gamma(i_values, r_orig, gamma_values, alpha, beta)
    generate_other_analyses(i_values, theta_values, colors)
    generate_3d_heatmap_analysis(i_values, theta_values, colors)
    generate_period_comparison_3d(i_values, theta_values, colors)

    # 打印分析结果
    print_analysis_results(i_values, r_orig, gamma_values, theta_orig, wavelength_orig)


if __name__ == "__main__":
    main()
