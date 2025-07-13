import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_rope_angles():
    """
    绘制 RoPE 旋转角度变化曲线，重点展示 sin(m*θ_i) 的周期性
    公式: m * θ_i = m * 10000^(-2i/d)
    其中 d=128, m从1到1024, i从0到d/2-1
    """
    # 参数设置
    d = 128  # 维度
    m_max = 1024  # 最大位置
    base = 10000  # 基数

    # 位置 m 的范围
    m_values = np.arange(1, m_max + 1)

    # 维度索引 i 的范围 (0 到 d/2-1)
    i_values = np.arange(0, d // 2)

    # 计算 θ_i = 10000^(-2i/d)
    theta_i = base ** (-2 * i_values / d)

    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 绘制 θ_i 随维度索引的变化（对数刻度）
    ax1.plot(i_values, theta_i, "b-", linewidth=2, marker="o", markersize=3)
    ax1.set_xlabel("维度索引 i")
    ax1.set_ylabel("θ_i = 10000^(-2i/d)")
    ax1.set_title("RoPE 角度基数 θ_i 随维度索引的变化")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # 添加周期信息
    for i in [0, 16, 32, 48, 63]:
        period = 2 * np.pi / theta_i[i]
        ax1.annotate(
            f"i={i}\n周期={period:.1f}",
            xy=(i, theta_i[i]),
            xytext=(i + 5, theta_i[i] * 2),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
            fontsize=8,
            ha="left",
        )

    # 2. 绘制 m*θ_i 随 m 的变化，同时显示 sin(m*θ_i) 的周期
    selected_i = [0, 8, 16, 32, 48, 63]  # 选择几个代表性的 i 值
    colors = cm.viridis(np.linspace(0, 1, len(selected_i)))

    for idx, i in enumerate(selected_i):
        angles = m_values * theta_i[i]
        ax2.plot(
            m_values,
            angles,
            color=colors[idx],
            linewidth=2,
            label=f"i={i}, θ_{i}={theta_i[i]:.2e}",
        )

        # 标记完整周期的位置
        period = 2 * np.pi / theta_i[i]
        if period <= m_max:
            # 标记第一个完整周期
            ax2.axvline(x=period, color=colors[idx], linestyle="--", alpha=0.5)
            ax2.axhline(y=2 * np.pi, color="gray", linestyle=":", alpha=0.3)

    ax2.set_xlabel("位置 m")
    ax2.set_ylabel("旋转角度 m*θ_i")
    ax2.set_title("旋转角度 m*θ_i 随位置的变化（虚线表示一个周期）")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 绘制 sin(m*θ_i) 的周期性变化
    selected_i_sin = [0, 16, 32, 48]  # 选择几个有代表性的 i 值
    colors_sin = cm.plasma(np.linspace(0, 1, len(selected_i_sin)))

    for idx, i in enumerate(selected_i_sin):
        angles = m_values * theta_i[i]
        sin_angles = np.sin(angles)
        ax3.plot(
            m_values,
            sin_angles,
            color=colors_sin[idx],
            linewidth=2,
            label=f"i={i}, 周期={2*np.pi/theta_i[i]:.1f}",
        )

        # 标记周期位置
        period = 2 * np.pi / theta_i[i]
        if period <= m_max:
            for n in range(1, int(m_max / period) + 1):
                ax3.axvline(
                    x=n * period, color=colors_sin[idx], linestyle="--", alpha=0.3
                )

    ax3.set_xlabel("位置 m")
    ax3.set_ylabel("sin(m*θ_i)")
    ax3.set_title("sin(m*θ_i) 的周期性变化")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.1, 1.1)

    # 4. 绘制不同 i 值的周期对比
    periods = 2 * np.pi / theta_i
    ax4.plot(i_values, periods, "ro-", linewidth=2, markersize=4)
    ax4.set_xlabel("维度索引 i")
    ax4.set_ylabel("sin(m*θ_i) 的周期长度")
    ax4.set_title("不同维度索引对应的周期长度")
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale("log")

    # 标记一些关键点
    for i in [0, 16, 32, 48, 63]:
        ax4.annotate(
            f"i={i}\n周期={periods[i]:.1f}",
            xy=(i, periods[i]),
            xytext=(i + 3, periods[i] * 1.5),
            arrowprops=dict(arrowstyle="->", color="blue", alpha=0.7),
            fontsize=8,
            ha="left",
        )

    plt.tight_layout()
    plt.show()

    # 打印详细的周期分析
    print("RoPE 角度和周期分析:")
    print(f"维度 d = {d}")
    print(f"位置范围: 1 到 {m_max}")
    print(f"θ_i 范围: {theta_i.min():.2e} 到 {theta_i.max():.2e}")
    print(f"周期范围: {periods.min():.1f} 到 {periods.max():.1f}")

    print("\n不同维度索引的详细信息:")
    print("i\tθ_i\t\t周期\t\t在1024位置内的周期数")
    print("-" * 60)
    for i in [0, 8, 16, 24, 32, 40, 48, 56, 63]:
        period = periods[i]
        cycles_in_range = m_max / period
        print(f"{i}\t{theta_i[i]:.2e}\t{period:.1f}\t\t{cycles_in_range:.2f}")

    # 分析频率特性
    print(f"\n频率分析:")
    print(f"最高频率分量 (i=0): 周期 = {periods[0]:.1f} 位置")
    print(f"最低频率分量 (i={d//2-1}): 周期 = {periods[-1]:.1f} 位置")
    print(f"频率比 (最低/最高): {periods[-1]/periods[0]:.1f}")


def plot_detailed_sin_cycles():
    """
    详细展示几个具体 i 值的 sin(m*θ_i) 周期性变化
    """
    d = 128
    m_max = 512  # 减小范围以便更清楚地看到周期
    base = 10000

    m_values = np.arange(1, m_max + 1)
    i_values = np.arange(0, d // 2)
    theta_i = base ** (-2 * i_values / d)

    # 选择几个有代表性的 i 值
    selected_i = [0, 16, 32, 48]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    colors = ["red", "blue", "green", "purple"]

    for idx, i in enumerate(selected_i):
        ax = axes[idx]

        # 计算角度和正弦值
        angles = m_values * theta_i[i]
        sin_angles = np.sin(angles)

        # 绘制 sin(m*θ_i)
        ax.plot(
            m_values, sin_angles, color=colors[idx], linewidth=2, label=f"sin(m*θ_{i})"
        )

        # 计算并标记周期
        period = 2 * np.pi / theta_i[i]

        # 标记周期边界
        for n in range(1, int(m_max / period) + 1):
            ax.axvline(x=n * period, color=colors[idx], linestyle="--", alpha=0.5)
            ax.text(n * period, 0.8, f"{n}T", rotation=90, ha="right", va="bottom")

        ax.set_xlabel("位置 m")
        ax.set_ylabel("sin(m*θ_i)")
        ax.set_title(f"i={i}: θ_{i}={theta_i[i]:.2e}, 周期={period:.1f}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_rope_angles()
    print("\n" + "=" * 50)
    print("详细的正弦函数周期展示:")
    plot_detailed_sin_cycles()
