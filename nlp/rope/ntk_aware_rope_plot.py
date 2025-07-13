import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

def plot_ntk_aware_rope():
    """
    绘制 NTK-Aware Scaled RoPE 的对比分析
    对比原始 base 和新 base 下的 θ_i 以及 m*θ_i 变化
    """
    # 参数设置
    d = 128  # 维度
    max_position_embeddings = 16384  # 新的最大位置
    a = 8  # Alpha value (扩展倍数)
    old_base = 10000  # 原始基数

    # 计算新的基数
    new_base = old_base * (a ** (d / (d - 2)))

    print(f"NTK-Aware Scaled RoPE 参数:")
    print(f"维度 d = {d}")
    print(f"扩展倍数 a = {a}")
    print(f"原始 base = {old_base}")
    print(f"新 base = {new_base:.2f}")
    print(f"base 扩展倍数 = {new_base/old_base:.2f}")

    # 位置范围
    m_values = np.arange(1, max_position_embeddings + 1)
    m_values_short = np.arange(1, 2048 + 1)  # 原始训练长度

    # 维度索引
    i_values = np.arange(0, d // 2)

    # 计算 θ_i
    theta_i_old = old_base ** (-2 * i_values / d)
    theta_i_new = new_base ** (-2 * i_values / d)

    # 创建图形
    fig = plt.figure(figsize=(20, 16))

    # 1. θ_i 对比
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(
        i_values,
        theta_i_old,
        "b-",
        linewidth=2,
        label="原始 base=10000",
        marker="o",
        markersize=3,
    )
    ax1.plot(
        i_values,
        theta_i_new,
        "r-",
        linewidth=2,
        label=f"新 base={new_base:.0f}",
        marker="s",
        markersize=3,
    )
    ax1.set_xlabel("维度索引 i")
    ax1.set_ylabel("θ_i")
    ax1.set_title("θ_i 对比 (对数刻度)")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. θ_i 比值
    ax2 = plt.subplot(3, 3, 2)
    theta_ratio = theta_i_new / theta_i_old
    ax2.plot(i_values, theta_ratio, "g-", linewidth=2, marker="o", markersize=3)
    ax2.set_xlabel("维度索引 i")
    ax2.set_ylabel("θ_i_new / θ_i_old")
    ax2.set_title("θ_i 缩放比例")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color="k", linestyle="--", alpha=0.5)

    # 3. 周期对比
    ax3 = plt.subplot(3, 3, 3)
    periods_old = 2 * np.pi / theta_i_old
    periods_new = 2 * np.pi / theta_i_new

    ax3.plot(
        i_values,
        periods_old,
        "b-",
        linewidth=2,
        label="原始周期",
        marker="o",
        markersize=3,
    )
    ax3.plot(
        i_values,
        periods_new,
        "r-",
        linewidth=2,
        label="新周期",
        marker="s",
        markersize=3,
    )
    ax3.set_xlabel("维度索引 i")
    ax3.set_ylabel("周期长度")
    ax3.set_title("sin(m*θ_i) 周期对比")
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 选择几个代表性的 i 值绘制 m*θ_i 变化
    selected_i = [0, 16, 32, 48, 63]
    colors = cm.viridis(np.linspace(0, 1, len(selected_i)))

    ax4 = plt.subplot(3, 3, 4)
    for idx, i in enumerate(selected_i):
        angles_old = m_values_short * theta_i_old[i]
        ax4.plot(
            m_values_short,
            angles_old,
            color=colors[idx],
            linewidth=2,
            linestyle="-",
            label=f"原始 i={i}",
        )

    ax4.set_xlabel("位置 m")
    ax4.set_ylabel("m*θ_i (原始)")
    ax4.set_title("原始 base 下 m*θ_i 变化 (2048 位置)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(3, 3, 5)
    for idx, i in enumerate(selected_i):
        angles_new = m_values_short * theta_i_new[i]
        ax5.plot(
            m_values_short,
            angles_new,
            color=colors[idx],
            linewidth=2,
            linestyle="--",
            label=f"新 i={i}",
        )

    ax5.set_xlabel("位置 m")
    ax5.set_ylabel("m*θ_i (新)")
    ax5.set_title("新 base 下 m*θ_i 变化 (2048 位置)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 在扩展位置范围内的对比
    ax6 = plt.subplot(3, 3, 6)
    m_extended = np.arange(1, 16384 + 1, 16)  # 降采样以便可视化

    # 选择一个中等频率的 i 值进行详细对比
    i_demo = 32
    angles_old_ext = m_extended * theta_i_old[i_demo]
    angles_new_ext = m_extended * theta_i_new[i_demo]

    ax6.plot(m_extended, angles_old_ext, "b-", linewidth=2, label=f"原始 i={i_demo}")
    ax6.plot(m_extended, angles_new_ext, "r-", linewidth=2, label=f"新 i={i_demo}")
    ax6.set_xlabel("位置 m")
    ax6.set_ylabel("m*θ_i")
    ax6.set_title(f"扩展位置范围内 m*θ_i 对比 (i={i_demo})")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. sin(m*θ_i) 周期性对比 - 原始
    ax7 = plt.subplot(3, 3, 7)
    selected_i_sin = [0, 16, 32, 48]
    colors_sin = cm.plasma(np.linspace(0, 1, len(selected_i_sin)))

    for idx, i in enumerate(selected_i_sin):
        angles = m_values_short * theta_i_old[i]
        sin_angles = np.sin(angles)
        ax7.plot(
            m_values_short,
            sin_angles,
            color=colors_sin[idx],
            linewidth=2,
            label=f"i={i}, T={2*np.pi/theta_i_old[i]:.1f}",
        )

    ax7.set_xlabel("位置 m")
    ax7.set_ylabel("sin(m*θ_i)")
    ax7.set_title("原始 base: sin(m*θ_i) 周期性")
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(-1.1, 1.1)

    # 8. sin(m*θ_i) 周期性对比 - 新
    ax8 = plt.subplot(3, 3, 8)

    for idx, i in enumerate(selected_i_sin):
        angles = m_values_short * theta_i_new[i]
        sin_angles = np.sin(angles)
        ax8.plot(
            m_values_short,
            sin_angles,
            color=colors_sin[idx],
            linewidth=2,
            label=f"i={i}, T={2*np.pi/theta_i_new[i]:.1f}",
        )

    ax8.set_xlabel("位置 m")
    ax8.set_ylabel("sin(m*θ_i)")
    ax8.set_title("新 base: sin(m*θ_i) 周期性")
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(-1.1, 1.1)

    # 9. 周期比率分析
    ax9 = plt.subplot(3, 3, 9)
    period_ratio = periods_new / periods_old
    ax9.plot(i_values, period_ratio, "purple", linewidth=2, marker="o", markersize=4)
    ax9.set_xlabel("维度索引 i")
    ax9.set_ylabel("周期比率 (新/原始)")
    ax9.set_title("周期扩展比率")
    ax9.grid(True, alpha=0.3)
    ax9.axhline(y=a, color="r", linestyle="--", alpha=0.7, label=f"目标比率 a={a}")
    ax9.legend()

    plt.tight_layout()
    plt.show()

    # 详细分析报告
    print(f"\n{'='*60}")
    print("详细分析报告:")
    print(f"{'='*60}")

    print(f"\n1. 基数变化:")
    print(f"   原始 base: {old_base}")
    print(f"   新 base: {new_base:.2f}")
    print(f"   变化倍数: {new_base/old_base:.2f}")

    print(f"\n2. 不同维度索引的变化:")
    print(
        f"{'i':<3} {'θ_i_old':<12} {'θ_i_new':<12} {'比率':<8} {'周期_old':<12} {'周期_new':<12} {'周期比率':<8}"
    )
    print("-" * 80)

    for i in [0, 8, 16, 24, 32, 40, 48, 56, 63]:
        theta_old = theta_i_old[i]
        theta_new = theta_i_new[i]
        ratio = theta_new / theta_old
        period_old = 2 * np.pi / theta_old
        period_new = 2 * np.pi / theta_new
        period_ratio = period_new / period_old

        print(
            f"{i:<3} {theta_old:<12.2e} {theta_new:<12.2e} {ratio:<8.2f} "
            f"{period_old:<12.1f} {period_new:<12.1f} {period_ratio:<8.2f}"
        )

    print(f"\n3. 频率分量分析:")
    print(f"   最高频率 (i=0):")
    print(f"     原始周期: {periods_old[0]:.1f} 位置")
    print(f"     新周期: {periods_new[0]:.1f} 位置")
    print(f"     扩展比率: {periods_new[0]/periods_old[0]:.2f}")

    print(f"   最低频率 (i={d//2-1}):")
    print(f"     原始周期: {periods_old[-1]:.1f} 位置")
    print(f"     新周期: {periods_new[-1]:.1f} 位置")
    print(f"     扩展比率: {periods_new[-1]/periods_old[-1]:.2f}")

    print(f"\n4. 位置扩展效果:")
    print(f"   原始最大位置: 2048")
    print(f"   新最大位置: {max_position_embeddings}")
    print(f"   位置扩展倍数: {max_position_embeddings/2048:.1f}")

    # 验证在新位置范围内的角度范围
    max_angle_old = max_position_embeddings * theta_i_old.max()
    max_angle_new = max_position_embeddings * theta_i_new.max()

    print(f"\n5. 角度范围验证:")
    print(f"   在位置 {max_position_embeddings} 处:")
    print(
        f"     原始最大角度: {max_angle_old:.2f} 弧度 ({max_angle_old/(2*np.pi):.1f} 个周期)"
    )
    print(
        f"     新最大角度: {max_angle_new:.2f} 弧度 ({max_angle_new/(2*np.pi):.1f} 个周期)"
    )


def plot_interpolation_comparison():
    """
    对比 NTK-Aware 和位置插值两种方法
    """
    print(f"\n{'='*60}")
    print("NTK-Aware vs 位置插值对比:")
    print(f"{'='*60}")

    d = 128
    old_base = 10000
    a = 8
    original_length = 2048
    extended_length = 16384

    # NTK-Aware 方法
    new_base = old_base * (a ** (d / (d - 2)))

    # 位置插值方法的缩放因子
    scale_factor = original_length / extended_length

    i_values = np.arange(0, d // 2)

    # 计算不同方法的 θ_i
    theta_original = old_base ** (-2 * i_values / d)
    theta_ntk = new_base ** (-2 * i_values / d)

    # 位置插值等效于在计算角度时乘以缩放因子
    # 即 m * scale_factor * theta_original

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 选择一个中等频率的维度索引
    i_demo = 32
    m_values = np.arange(1, extended_length + 1, 16)

    # 1. 角度对比
    angles_original = m_values * theta_original[i_demo]
    angles_ntk = m_values * theta_ntk[i_demo]
    angles_interpolation = m_values * scale_factor * theta_original[i_demo]

    ax1.plot(m_values, angles_original, "b-", linewidth=2, label="原始 (无扩展)")
    ax1.plot(m_values, angles_ntk, "r-", linewidth=2, label="NTK-Aware")
    ax1.plot(m_values, angles_interpolation, "g-", linewidth=2, label="位置插值")
    ax1.set_xlabel("位置 m")
    ax1.set_ylabel("m*θ_i")
    ax1.set_title(f"角度对比 (i={i_demo})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 正弦函数对比
    ax2.plot(m_values, np.sin(angles_original), "b-", linewidth=2, label="原始")
    ax2.plot(m_values, np.sin(angles_ntk), "r-", linewidth=2, label="NTK-Aware")
    ax2.plot(
        m_values, np.sin(angles_interpolation), "g-", linewidth=2, label="位置插值"
    )
    ax2.set_xlabel("位置 m")
    ax2.set_ylabel("sin(m*θ_i)")
    ax2.set_title(f"正弦函数对比 (i={i_demo})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)

    # 3. 周期对比
    periods_original = 2 * np.pi / theta_original
    periods_ntk = 2 * np.pi / theta_ntk
    periods_interpolation = 2 * np.pi / (scale_factor * theta_original)

    ax3.plot(i_values, periods_original, "b-", linewidth=2, label="原始")
    ax3.plot(i_values, periods_ntk, "r-", linewidth=2, label="NTK-Aware")
    ax3.plot(i_values, periods_interpolation, "g-", linewidth=2, label="位置插值")
    ax3.set_xlabel("维度索引 i")
    ax3.set_ylabel("周期长度")
    ax3.set_title("周期对比")
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 周期扩展比率
    ratio_ntk = periods_ntk / periods_original
    ratio_interpolation = periods_interpolation / periods_original

    ax4.plot(
        i_values,
        ratio_ntk,
        "r-",
        linewidth=2,
        marker="o",
        markersize=3,
        label="NTK-Aware",
    )
    ax4.plot(
        i_values,
        ratio_interpolation,
        "g-",
        linewidth=2,
        marker="s",
        markersize=3,
        label="位置插值",
    )
    ax4.axhline(y=a, color="k", linestyle="--", alpha=0.7, label=f"目标比率 {a}")
    ax4.set_xlabel("维度索引 i")
    ax4.set_ylabel("周期扩展比率")
    ax4.set_title("周期扩展比率对比")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n方法对比总结:")
    print(f"NTK-Aware 方法:")
    print(f"  - 新 base = {new_base:.2f}")
    print(f"  - 所有频率分量的周期都按不同比例扩展")
    print(f"  - 低频分量扩展更多，高频分量扩展较少")

    print(f"\n位置插值方法:")
    print(f"  - 缩放因子 = {scale_factor:.4f}")
    print(f"  - 所有频率分量的周期都按相同比例 {1/scale_factor:.1f} 扩展")
    print(f"  - 等效于将位置坐标压缩到原始范围")


if __name__ == "__main__":
    plot_ntk_aware_rope()
    plot_interpolation_comparison()
