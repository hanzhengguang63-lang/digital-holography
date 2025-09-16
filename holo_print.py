import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 读取灰度图
img_path = r"readtest100_print.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"无法读取图像: {img_path}")

height, width = img.shape
print(f"图像分辨率: {width} x {height}")

# 分块大小保持100×100（根据之前修改）
blocks_x, blocks_y = 100, 100
block_w_f = width / blocks_x
block_h_f = height / blocks_y

# 转为彩色图用于标注
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 用于存储每个色块等级
levels_array = np.zeros((blocks_y, blocks_x), dtype=int)

# 图像灰度直方图分析
all_pixels = img.flatten()

# 使用K-means聚类确定三个灰度中心
pixel_values = all_pixels.reshape(-1, 1)
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(pixel_values)
centers = np.sort(kmeans.cluster_centers_.flatten())

# 基于聚类中心确定阈值
t_low = (centers[0] + centers[1]) / 2
t_high = (centers[1] + centers[2]) / 2
print(f"聚类灰度中心: {centers[0]:.1f} (黑), {centers[1]:.1f} (灰), {centers[2]:.1f} (白)")
print(f"聚类阈值: t_low={t_low:.1f}, t_high={t_high:.1f}")

# 颜色定义
colors = [
    (0, 0, 255),  # 红 - 白 (最亮)
    (0, 165, 255),  # 橙 - 灰
    (0, 255, 0)  # 绿 - 黑 (最暗)
]

# 处理每个区块 - 专注混合区域的占比最多色阶标注
for by in range(blocks_y):
    for bx in range(blocks_x):
        # 计算区块边界
        x_start = int(round(bx * block_w_f))
        x_end = int(round((bx + 1) * block_w_f))
        y_start = int(round(by * block_h_f))
        y_end = int(round((by + 1) * block_h_f))
        x_end = min(x_end, width)
        y_end = min(y_end, height)

        # 获取当前区块
        block = img[y_start:y_end, x_start:x_end]

        # 统计三种色阶的像素数量
        black_pixels = np.sum(block < t_low)
        gray_pixels = np.sum((block >= t_low) & (block < t_high))
        white_pixels = np.sum(block >= t_high)

        # 选择占比最多的色阶作为整个区块的等级
        if white_pixels >= max(gray_pixels, black_pixels):
            level = 0  # 白色占比最多
        elif gray_pixels >= max(white_pixels, black_pixels):
            level = 1  # 灰色占比最多
        else:
            level = 2  # 黑色占比最多

        levels_array[by, bx] = level

        # 计算中心点
        cx = (x_start + x_end) // 2
        cy = (y_start + y_end) // 2

        # 修改此处：将字体大小从0.5调小为0.3（可根据需要进一步调整）
        text_size = 0.9
        # 可选：根据区块大小动态调整字体位置偏移（避免文字超出区块）
        offset = int(max(block_w_f, block_h_f) * 0.1)  # 动态偏移量（约区块边长的10%）
        cv2.putText(img_color, str(level), (cx - offset, cy + offset),
                    cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), 1, cv2.LINE_AA)

        # 绘制色块边界（边界线颜色与标注等级相同）
        cv2.rectangle(img_color, (x_start, y_start), (x_end, y_end), colors[level], 1)

# 保存结果
output_path = r"D:\PythonProject2\holo_new2_3level_100-100\output\readtest_print_annotated_dominant.png"
cv2.imwrite(output_path, img_color)
print(f"带标注图像已保存: {output_path}")

# 保存等级数组
excel_path = r"D:\PythonProject2\holo_new2_3level_100-100\output\\print_gray_levels_dominant.xlsx"
df = pd.DataFrame(levels_array)
df.to_excel(excel_path, index=False, header=False)
print(f"灰度等级数组已保存为 Excel 文件: {excel_path}")

# 显示结果
cv2.imshow("Dominant Gray Level Analysis", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 生成分类报告
white_blocks = np.sum(levels_array == 0)
gray_blocks = np.sum(levels_array == 1)
black_blocks = np.sum(levels_array == 2)
total_blocks = blocks_x * blocks_y

print("\n===== 区域分类报告 =====")
print(f"白色区块: {white_blocks} ({white_blocks / total_blocks:.1%})")
print(f"灰色区块: {gray_blocks} ({gray_blocks / total_blocks:.1%})")
print(f"黑色区块: {black_blocks} ({black_blocks / total_blocks:.1%})")
print("=========================")