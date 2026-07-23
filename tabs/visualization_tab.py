import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.ndimage import gaussian_filter

from core.common import downsample_for_plot

def render_visualization_tab(df, group_col, group_list, g_limit, grid_val):
    for item in group_list:
        plot_data = df[df[group_col] == item]
        if plot_data.empty: continue

        with st.expander(f"📍 {item} 리포트 ({len(plot_data):,} 샘플)", expanded=True):
            c1, c2, c3 = st.columns(3)

            # 속도 히스토그램
            with c1:
                fig, ax = plt.subplots(figsize=(4, 4))
                st.markdown("<h3 style='text-align: center; margin-bottom: 0px;'>🎢 주행 속도 분포</h3>", unsafe_allow_html=True)
                # KDE 시각화만 축소하고 평균/최빈값 등 통계는 전체 데이터로 계산
                speed_plot_data = downsample_for_plot(plot_data[['speed']], max_points=15000)
                sns.histplot(speed_plot_data['speed'], bins=20, kde=True, color='skyblue', ax=ax)
                avg_speed = plot_data['speed'].mean()

                if not plot_data['speed'].dropna().empty:
                    max_dist_speed = plot_data['speed'].round(1).mode().iloc[0]
                else:
                    max_dist_speed = 0

                ax.axvline(avg_speed, color='red', linestyle='--', linewidth=1.5)
                ax.text(avg_speed + 2, ax.get_ylim()[1] * 0.9, f'평균: {avg_speed:.1f}',
                        color='red', fontweight='bold', fontsize=9)

                ax.axvline(max_dist_speed, color='purple', linestyle='-', linewidth=2)
                ax.text(max_dist_speed + 2, ax.get_ylim()[1] * 0.8, f'최대 분포: {max_dist_speed:.1f}',
                        color='purple', fontweight='bold', fontsize=9)

                ax.set_xlabel("Speed (km/h)", fontsize=10)
                ax.set_ylabel("Count", fontsize=10)
                st.pyplot(fig, width="stretch")
                plt.close(fig)

            # G-G Map
            with c2:
                boundaries = np.linspace(-g_limit, g_limit, 6)
                colors = ["#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF0000"]
                cm = LinearSegmentedColormap.from_list("custom_gg", colors, N=256)
                fig_g, ax_g = plt.subplots(figsize=(4, 4))
                st.markdown("<h3 style='text-align: center; margin-bottom: 0px;'>🎯 주행 패턴 맵</h3>", unsafe_allow_html=True)
                counts, xedges, yedges = np.histogram2d(
                    plot_data['accYG'], plot_data['accXG'],
                    bins=100, range=[[-g_limit, g_limit], [-g_limit, g_limit]]
                )
                counts_smoothed = gaussian_filter(counts, sigma=1.2)
                counts_smoothed[counts_smoothed < 0.1] = np.nan
                ax_g.imshow(
                    counts_smoothed.T, origin='lower',
                    extent=[-g_limit, g_limit, -g_limit, g_limit],
                    cmap=cm, aspect='equal', interpolation='bilinear'
                )

                ax_g.set_xticks(boundaries)
                ax_g.set_yticks(boundaries)

                tick_labels = [f"{b:.1f}" for b in boundaries]
                ax_g.set_xticklabels(tick_labels, fontsize=9)
                ax_g.set_yticklabels(tick_labels, fontsize=9)

                ax_g.set_facecolor('#000033')
                ax_g.axhline(0, color='white', alpha=0.5, linewidth=0.8)
                ax_g.axvline(0, color='white', alpha=0.5, linewidth=0.8)

                for b in boundaries:
                    ax_g.axhline(b, color='white', alpha=0.1, linewidth=0.5)
                    ax_g.axvline(b, color='white', alpha=0.1, linewidth=0.5)

                ax_g.set_xlabel("Lateral Acceleration (accYG)", fontsize=10)
                ax_g.set_ylabel("Longitudinal Acceleration (accXG)", fontsize=10)
                st.pyplot(fig_g, width="stretch")
                plt.close(fig_g)

            with c3:
                st.markdown("<h3 style='text-align: center; margin-bottom: 0px;'>🔳 주행 패턴 분포</h3>", unsafe_allow_html=True)
                boundaries = np.linspace(-g_limit, g_limit, grid_val + 1)

                # [최적화 2] 느린 pd.crosstab을 np.histogram2d 연산으로 교체 (수십 배 연산 속도 향상)
                hist_counts, _, _ = np.histogram2d(
                    plot_data['accYG'], plot_data['accXG'],
                    bins=[boundaries, boundaries]
                )
                # 화면 표출(imshow)을 위해 배열 상하 반전 및 전치
                matrix_vals = hist_counts.T[::-1, :]
                total_count = matrix_vals.sum()

                fig, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(matrix_vals, cmap='RdYlGn_r',
                            extent=[-g_limit, g_limit, -g_limit, g_limit],
                            norm=LogNorm(vmin=1, vmax=matrix_vals.max() if matrix_vals.max() > 0 else 1))

                ax.set_xticks(boundaries)
                ax.set_yticks(boundaries)

                tick_labels = [f"{b:.1f}" for b in boundaries]
                ax.set_xticklabels(tick_labels, fontsize=9)
                ax.set_yticklabels(tick_labels, fontsize=9)

                bin_centers = (boundaries[:-1] + boundaries[1:]) / 2
                y_centers = bin_centers[::-1]
                offset = (g_limit / grid_val) * 0.7
                base_font = 10 if grid_val <= 5 else 8

                for i in range(len(y_centers)):
                    for j in range(len(bin_centers)):
                        val = int(matrix_vals[i, j])
                        if val > 0:
                            percentage = (val / total_count) * 100
                            center_idx = grid_val // 2
                            txt_color = "white" if (i == center_idx and j == center_idx) else "black"

                            ax.text(bin_centers[j], y_centers[i] + (offset * 0.2), f"{val:,}",
                                    ha="center", va="center", color=txt_color, fontsize=base_font)

                            ax.text(bin_centers[j], y_centers[i] - (offset * 0.35), f"({percentage:.1f}%)",
                                    ha="center", va="center", color=txt_color, fontsize=base_font-2, alpha=0.8, fontweight='bold')

                ax.set_aspect('equal')
                ax.grid(True, which='major', color='white', linestyle='-', linewidth=1.5 if grid_val <= 5 else 0.8, alpha=0.4)
                ax.set_xlabel("Lateral Acceleration (accYG)", fontsize=10)
                ax.set_ylabel("Longitudinal Acceleration (accXG)", fontsize=10)

                st.pyplot(fig, width="stretch")
                plt.close(fig)

            del plot_data
            gc.collect()

