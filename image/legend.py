import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 颜色和样式设置（与主图一致）
colors = {
    "OD-CASE": "#ff7f0e",
    "OD-DART": "#03721f", 
    "OD-TF": "#0024f1",
    "FusionQuery": "#636601",
    "IncreQueryFusion": "#d62728"
}

line_styles = {
    "OD-CASE": "--",
    "OD-DART": "-.",
    "OD-TF": ":",
    "FusionQuery": "--",
    "IncreQueryFusion": "-"
}

# 创建单独的图例图形
fig, ax = plt.subplots(1, 1, figsize=(20, 1))  # 宽度是原图的两倍，高度较小

# 创建图例句柄
legend_handles = []
methods = ["OD-CASE", "OD-DART", "OD-TF", "FusionQuery", "IncreQueryFusion"]

for method in methods:
    handle = mlines.Line2D([], [], 
                          color=colors[method], 
                          linestyle=line_styles[method],
                          linewidth=3,
                          label=method)
    legend_handles.append(handle)

# 创建图例并放置在顶部中央
legend = ax.legend(handles=legend_handles, 
                  loc='upper center', 
                  bbox_to_anchor=(0.5, 0.5),
                  ncol=5,  # 水平排列所有图例项
                  fontsize=28,  # 稍微增大字体
                  frameon=True,  # 显示边框
                  fancybox=True,  # 圆角边框
                  shadow=True,   # 阴影效果
                  framealpha=0.9)  # 背景透明度

# 隐藏坐标轴
ax.axis('off')

# 调整布局
plt.tight_layout()

# 保存图例图片
plt.savefig("./legend_only.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

print("图例已保存为 'legend_only.pdf'")