from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1. 读图
img = Image.open("img.png").convert("RGBA")
arr = np.array(img)

R = arr[:, :, 0].astype(int)
G = arr[:, :, 1].astype(int)
B = arr[:, :, 2].astype(int)

# 2. 取“绿色”掩码（G 高且比 R、B 都大一些）
green_mask = (G > 150) & (R < 200) & (B < 150) & (G > R + 10) & (G > B + 10)

# 3. 对每一列 x，找到该列绿色像素的 y 位置（用中位数/平均都可以）
xs = np.where(green_mask.any(axis=0))[0]        # 有绿点的所有 x
ys_pix = []

for x in xs:
    ys = np.where(green_mask[:, x])[0]
    if len(ys) == 0:
        continue
    ys_pix.append(np.median(ys))  # 该列绿线的 y 像素坐标（0 在上方）

xs = np.array(xs, dtype=float)
ys_pix = np.array(ys_pix, dtype=float)

# 4. 像素坐标 → 真实坐标：这里需要你手动测两个 tick 的像素位置
#    可以用截图工具量一量，比如：
#    假设：x_pix_min 对应 step_min = 0
#          x_pix_max 对应 step_max = 2800
#    同理 y 轴：y_pix_top 对应 norm_max，y_pix_bottom 对应 norm_min
#    注意：图像坐标 y 往下增，而数值坐标是向上增，所以要反向映射

# ==== 这几个数字需要你根据图像手动改 ====
x_pix_min, step_min = xs.min(), 0.0
x_pix_max, step_max = xs.max(), 2800.0

y_pix_top, norm_max = 120.0, 250.0   # 顶部某个 y 像素位置对应的 Frobenius Norm
y_pix_bot, norm_min = 820.0, 30.0    # 底部某个 y 像素位置对应的 Frobenius Norm
# =====================================

# 线性映射
steps = step_min + (xs - x_pix_min) / (x_pix_max - x_pix_min) * (step_max - step_min)

# 图像 y 越大在越下方，所以 Norm 要反过来
norms = norm_max + (ys_pix - y_pix_top) / (y_pix_bot - y_pix_top) * (norm_min - norm_max)

# 5. 取倒数（避免除 0）
eps = 1e-8
inv_norms = 1.0 / (norms + eps)

# 6. 画图
plt.figure(figsize=(6,4))
plt.plot(steps, inv_norms)
plt.xlabel("Step")
plt.ylabel("1 / Frobenius Norm")
plt.title("Reciprocal of Green Curve")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
