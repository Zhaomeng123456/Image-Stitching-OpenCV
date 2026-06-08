"""
基于特征的自动图像拼接算法。

使用 SIFT 提取特征点，KNN 进行特征匹配，RANSAC 估计单应性矩阵，
再通过透视变换与线性渐变融合生成无缝全景图。
"""

import cv2
import numpy as np
import sys
import time


class Image_Stitching():
    """图像拼接类，实现两张重叠图像的配准与融合。"""

    def __init__(self):
        # Lowe 比率测试阈值：最近邻距离 / 次近邻距离 小于该值才视为有效匹配
        self.ratio = 0.85
        # 计算单应性矩阵所需的最少匹配点对数量
        self.min_match = 10
        # SIFT 特征检测器（需 opencv-contrib-python）
        self.sift = cv2.xfeatures2d.SIFT_create()
        # 重叠区域线性融合窗口最大宽度（像素），实际宽度不超过真实重叠区
        self.smoothing_window_size = 800

    def registration(self, img1, img2):
        """
        图像配准：提取特征、匹配特征点，并估计从 img2 到 img1 的单应性矩阵 H。

        参数:
            img1: 左图（参考图），BGR 格式
            img2: 右图（待变换图），BGR 格式

        返回:
            H: 3x3 单应性矩阵，将 img2 映射到 img1 坐标系；匹配不足时可能未定义
        """
        # 分别在两张图像上检测 SIFT 关键点并计算描述子
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        # 暴力匹配器，对每个描述子找 k=2 个最近邻
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)

        good_points = []   # 有效匹配点对索引：(img2 中的 idx, img1 中的 idx)
        good_matches = []  # 用于可视化绘制的匹配结果

        # Lowe 比率测试：保留距离明显小于次近邻的匹配，剔除歧义匹配
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])

        # 绘制特征匹配可视化图（GUI 可通过 match_vis 属性获取）
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        self.match_vis = img3
        self.num_good_matches = len(good_points)

        # 匹配点足够多时，用 RANSAC 鲁棒估计单应性矩阵
        if len(good_points) > self.min_match:
            # 提取 img1 中匹配点的像素坐标
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            # 提取 img2 中匹配点的像素坐标
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            # findHomography：将 img2 的点映射到 img1，RANSAC 阈值 5.0 像素
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
        else:
            H = None
        return H

    def _build_canvas_masks(self, img1, img2, H, width_panorama, height_panorama):
        """
        在全景画布上构建左图/右图有效区域掩膜，并求真实重叠区列范围。

        返回:
            mask1, mask2: float32 掩膜，取值 0 或 1
            overlap: bool 重叠区掩膜
            x_blend_start, x_blend_end: 重叠区水平融合起止列
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        mask1 = np.zeros((height_panorama, width_panorama), dtype=np.float32)
        mask1[:h1, :w1] = 1.0

        mask2_src = np.ones((h2, w2), dtype=np.uint8)
        mask2 = cv2.warpPerspective(mask2_src, H, (width_panorama, height_panorama))
        mask2 = (mask2 > 0).astype(np.float32)

        overlap = (mask1 > 0) & (mask2 > 0)
        overlap_cols = np.where(overlap.any(axis=0))[0]
        if len(overlap_cols) == 0:
            raise RuntimeError("未检测到有效重叠区域，无法拼接")

        x_overlap_start = int(overlap_cols[0])
        x_overlap_end = int(overlap_cols[-1])
        overlap_width = x_overlap_end - x_overlap_start + 1

        # 融合带限制在真实重叠区内，宽度不超过 smoothing_window_size
        blend_width = min(self.smoothing_window_size, overlap_width)
        x_blend_start = x_overlap_start + (overlap_width - blend_width) // 2
        x_blend_end = x_blend_start + blend_width - 1

        return mask1, mask2, overlap, x_blend_start, x_blend_end

    def compensate_exposure(self, img1_canvas, img2_warped, overlap):
        """
        在重叠区估计每通道增益与偏置，将右图亮度对齐到左图（参考图）。

        模型: img1 ≈ gain * img2 + bias（分 BGR 三通道独立估计）

        返回:
            曝光校正后的 img2_warped（float32）
        """
        img2_corrected = img2_warped.copy()
        self.exposure_gains = []
        self.exposure_biases = []

        for c in range(3):
            src = img2_warped[:, :, c][overlap]
            ref = img1_canvas[:, :, c][overlap]
            if src.size < 50:
                self.exposure_gains.append(1.0)
                self.exposure_biases.append(0.0)
                continue

            src_mean = src.mean()
            ref_mean = ref.mean()
            src_std = src.std()
            ref_std = ref.std()

            # 增益：匹配均值；偏置：匹配标准差差异带来的整体偏移
            gain = ref_mean / src_mean if src_mean > 1.0 else 1.0
            bias = ref_mean - gain * src_mean

            # 若对比度差异大，仅用增益避免过校正
            if src_std > 1.0 and ref_std > 1.0:
                std_ratio = ref_std / src_std
                if abs(std_ratio - 1.0) > 0.3:
                    bias = 0.0

            gain = float(np.clip(gain, 0.5, 2.0))
            bias = float(np.clip(bias, -40.0, 40.0))

            img2_corrected[:, :, c] = img2_corrected[:, :, c] * gain + bias
            self.exposure_gains.append(gain)
            self.exposure_biases.append(bias)

        return np.clip(img2_corrected, 0, 255)

    def create_blend_weights(self, mask1, mask2, overlap, x_blend_start, x_blend_end,
                             height_panorama, width_panorama):
        """
        基于真实重叠区创建线性渐变融合权重。

        非重叠区：左图区域 weight_left=1，右图区域 weight_right=1
        重叠区内：沿 x 方向线性过渡，left 1→0，right 0→1

        返回:
            weight_left, weight_right: float32 权重图，重叠区两者之和为 1
        """
        weight_left = np.zeros((height_panorama, width_panorama), dtype=np.float32)
        weight_right = np.zeros((height_panorama, width_panorama), dtype=np.float32)

        only_left = (mask1 > 0) & (mask2 == 0)
        only_right = (mask2 > 0) & (mask1 == 0)

        weight_left[only_left] = 1.0
        weight_right[only_right] = 1.0

        blend_cols = x_blend_end - x_blend_start + 1
        if blend_cols > 1:
            ramp = np.linspace(1.0, 0.0, blend_cols, dtype=np.float32)
            overlap_blend = overlap[:, x_blend_start:x_blend_end + 1]
            weight_left[:, x_blend_start:x_blend_end + 1] = np.where(
                overlap_blend, ramp, weight_left[:, x_blend_start:x_blend_end + 1])
            weight_right[:, x_blend_start:x_blend_end + 1] = np.where(
                overlap_blend, 1.0 - ramp, weight_right[:, x_blend_start:x_blend_end + 1])

        # 重叠区但不在渐变带内的像素：按最近侧归属
        overlap_only = overlap & (weight_left == 0) & (weight_right == 0)
        weight_left[overlap_only & (np.arange(width_panorama) < (x_blend_start + x_blend_end) // 2)] = 1.0
        weight_right[overlap_only & (np.arange(width_panorama) >= (x_blend_start + x_blend_end) // 2)] = 1.0

        return weight_left, weight_right

    def _merge_weight(self, weight):
        """单通道权重扩展为三通道，便于与 BGR 图像相乘。"""
        return cv2.merge([weight, weight, weight])

    # ── 标定参数持久化 ──────────────────────────────────────────────
    @staticmethod
    def save_calibration(filepath, H, img1_shape, img2_shape):
        """
        保存标定参数到 .npz 文件。

        H: 3x3 单应性矩阵（img2 → img1）
        img1_shape: 左图 (height, width) 或 (height, width, channels)
        img2_shape: 右图 (height, width) 或 (height, width, channels)

        后续切片时图像分辨率必须与标定一致，否则 H 不适用。
        """
        np.savez(filepath,
                 H=H,
                 img1_h=int(img1_shape[0]), img1_w=int(img1_shape[1]),
                 img2_h=int(img2_shape[0]), img2_w=int(img2_shape[1]))
        return True

    @staticmethod
    def load_calibration(filepath):
        """
        从 .npz 文件加载标定参数。

        返回:
            dict: {'H': np.ndarray (3x3),
                   'img1_shape': (h, w),
                   'img2_shape': (h, w)}
            文件不存在或格式错误时返回 None。
        """
        try:
            data = np.load(filepath)
            H = data['H']
            img1_shape = (int(data['img1_h']), int(data['img1_w']))
            img2_shape = (int(data['img2_h']), int(data['img2_w']))
            data.close()
            return {'H': H, 'img1_shape': img1_shape, 'img2_shape': img2_shape}
        except (IOError, KeyError, ValueError):
            return None

    # ── 标定模式：只计算 H，不做融合 ─────────────────────────────────
    def calibrate(self, img1, img2):
        """
        标定模式：对一对样本图执行配准，返回单应性矩阵 H。
        不进行融合与曝光补偿，比完整拼接更快。
        """
        return self.registration(img1, img2)

    def blending(self, img1, img2, H=None):
        """
        拼接融合：配准 img2 到 img1 坐标系，加权叠加并裁剪黑边。

        参数:
            img1: 左图（参考图），BGR 格式
            img2: 右图（经单应性变换后叠加），BGR 格式
            H:    可选的 3x3 单应性矩阵。传入时跳过 SIFT 配准直接使用；
                  不传则自动计算。

        返回:
            final_result: 裁剪后的全景图
        """
        t_total = time.perf_counter()

        # 获取 img2 → img1 的单应性矩阵
        t_reg = time.perf_counter()
        if H is not None:
            H = np.array(H, dtype=np.float64)
            # 检验分辨率一致性
            calib_h, calib_w = H.shape if H.shape == (3, 3) else (None, None)
            self.match_vis = None
            self.num_good_matches = 0
            self._using_cached_H = True
        else:
            H = self.registration(img1, img2)
            self._using_cached_H = False
            if H is None:
                raise RuntimeError("特征匹配不足，无法估计单应性矩阵")
        t_registration = time.perf_counter() - t_reg

        height_panorama = img1.shape[0]
        width_panorama = img1.shape[1] + img2.shape[1]
        h1, w1 = img1.shape[:2]

        # 左图放入画布
        img1_canvas = np.zeros((height_panorama, width_panorama, 3), dtype=np.float32)
        img1_canvas[:h1, :w1] = img1.astype(np.float32)

        # 右图透视变换到画布
        img2_warped = cv2.warpPerspective(
            img2.astype(np.float32), H, (width_panorama, height_panorama))

        # 基于单应矩阵计算真实重叠区与融合带
        mask1, mask2, overlap, x_blend_start, x_blend_end = self._build_canvas_masks(
            img1, img2, H, width_panorama, height_panorama)
        self.blend_range = (x_blend_start, x_blend_end)

        # 重叠区曝光补偿：右图对齐左图
        img2_warped = self.compensate_exposure(img1_canvas, img2_warped, overlap)

        # 真实重叠区线性渐变权重
        weight_left, weight_right = self.create_blend_weights(
            mask1, mask2, overlap, x_blend_start, x_blend_end,
            height_panorama, width_panorama)

        panorama1 = img1_canvas * self._merge_weight(weight_left)
        panorama2 = img2_warped * self._merge_weight(weight_right)

        # 两路加权图像相加，重叠区为线性混合
        result = panorama1 + panorama2

        # 裁剪掉透视变换产生的黑色无效区域
        valid = (result.sum(axis=2) > 0)
        rows, cols = np.where(valid)
        if len(rows) == 0:
            raise RuntimeError("拼接结果为空")
        min_row, max_row = int(rows.min()), int(rows.max()) + 1
        min_col, max_col = int(cols.min()), int(cols.max()) + 1
        final_result = np.clip(result[min_row:max_row, min_col:max_col], 0, 255).astype(np.uint8)

        self.elapsed = {
            'registration': t_registration,
            'blending': time.perf_counter() - t_total - t_registration,
            'total': time.perf_counter() - t_total,
        }
        return final_result


# ── 实时拼接运行时（固定 H，预计算所有中间数据）───────────────────
class StitchingRuntime:
    """
    实时拼接运行时。

    给定固定的单应性矩阵 H（通常来自标定），预计算 remap 映射表、
    融合权重图和裁剪区域。每帧拼接仅需 cv2.remap + 加权混合 + 裁剪，
    避免重复的 warpPerspective 和权重计算，实现实时性能。
    """

    def __init__(self, H, img1_shape, img2_shape, blend_window=800):
        """
        参数:
            H: 3x3 单应性矩阵，img2 → img1
            img1_shape: (h, w) 左图分辨率
            img2_shape: (h, w) 右图分辨率
            blend_window: 融合带宽度（像素），不超过真实重叠区
        """
        self.h1, self.w1 = int(img1_shape[0]), int(img1_shape[1])
        self.h2, self.w2 = int(img2_shape[0]), int(img2_shape[1])

        # 全景画布（假定右图不向上/下超出左图高度范围，且拼接以左图为准）
        self.pano_h = self.h1
        self.pano_w = self.w1 + self.w2

        H = np.array(H, dtype=np.float64)

        # 预计算阶段
        self._build_remap_maps(H)
        self._build_blend_and_crop(H, blend_window)
        self._build_gain_bias_map(H)

    # ── remap 映射表 ─────────────────────────────────────────────
    def _build_remap_maps(self, H):
        """构建 cv2.remap 所用的 map_x, map_y，将 img2 扭曲到全景画布上。"""
        H_inv = np.linalg.inv(H)
        grid_x, grid_y = np.meshgrid(np.arange(self.pano_w), np.arange(self.pano_h))
        denom = (H_inv[2, 0] * grid_x + H_inv[2, 1] * grid_y + H_inv[2, 2])
        self.map_x = ((H_inv[0, 0] * grid_x + H_inv[0, 1] * grid_y + H_inv[0, 2]) / denom).astype(np.float32)
        self.map_y = ((H_inv[1, 0] * grid_x + H_inv[1, 1] * grid_y + H_inv[1, 2]) / denom).astype(np.float32)

    # ── 融合权重与裁剪 ───────────────────────────────────────────
    def _build_blend_and_crop(self, H, blend_window):
        """预计算融合权重图与裁剪矩形（均在 __init__ 中一次性完成）。"""
        # 左图有效区域
        mask1 = np.zeros((self.pano_h, self.pano_w), dtype=np.float32)
        mask1[:self.h1, :self.w1] = 1.0

        # 右图 warp 后的有效区域
        mask2_src = np.ones((self.h2, self.w2), dtype=np.uint8)
        mask2 = cv2.warpPerspective(mask2_src, H, (self.pano_w, self.pano_h))
        mask2 = (mask2 > 0).astype(np.float32)

        overlap = (mask1 > 0) & (mask2 > 0)
        only_left = (mask1 > 0) & (mask2 == 0)
        only_right = (mask2 > 0) & (mask1 == 0)

        # 真实重叠区范围
        overlap_cols = np.where(overlap.any(axis=0))[0]
        if len(overlap_cols) == 0:
            raise RuntimeError("未检测到有效重叠区域，无法构建融合带")
        x_start = int(overlap_cols[0])
        x_end = int(overlap_cols[-1])
        ov_width = x_end - x_start + 1

        blend_width = min(blend_window, ov_width)
        x_blend_start = x_start + (ov_width - blend_width) // 2
        x_blend_end = x_blend_start + blend_width - 1
        self.blend_range = (x_blend_start, x_blend_end)

        # 权重图
        w_left = np.zeros((self.pano_h, self.pano_w), dtype=np.float32)
        w_right = np.zeros((self.pano_h, self.pano_w), dtype=np.float32)

        w_left[only_left] = 1.0
        w_right[only_right] = 1.0

        if blend_width > 1:
            ramp = np.linspace(1.0, 0.0, blend_width, dtype=np.float32)
            overlap_blend = overlap[:, x_blend_start:x_blend_end + 1]
            w_left[:, x_blend_start:x_blend_end + 1] = np.where(
                overlap_blend, ramp, w_left[:, x_blend_start:x_blend_end + 1])
            w_right[:, x_blend_start:x_blend_end + 1] = np.where(
                overlap_blend, 1.0 - ramp, w_right[:, x_blend_start:x_blend_end + 1])

        self.weight_left = cv2.merge([w_left, w_left, w_left])
        self.weight_right = cv2.merge([w_right, w_right, w_right])

        # 裁剪区域
        all_valid = mask1 + mask2 > 0
        rows, cols = np.where(all_valid)
        self.crop_y1, self.crop_y2 = int(rows.min()), int(rows.max()) + 1
        self.crop_x1, self.crop_x2 = int(cols.min()), int(cols.max()) + 1

        # 存储重叠掩膜，供曝光补偿用
        self.overlap_mask = overlap

    # ── 曝光补偿增益/偏置映射（可选，运行时动态应用） ──────────────
    def _build_gain_bias_map(self, H):
        """为曝光补偿预留占位，实际值由 set_exposure_params 设定。"""
        self.exp_gain = np.ones(3, dtype=np.float32)
        self.exp_bias = np.zeros(3, dtype=np.float32)
        self._exposure_set = False

    def set_exposure_params(self, gains, biases):
        """设置曝光补偿参数（BGR 三通道 gain 和 bias）。"""
        self.exp_gain = np.array(gains, dtype=np.float32)
        self.exp_bias = np.array(biases, dtype=np.float32)
        self._exposure_set = True

    # ── 单帧拼接 ─────────────────────────────────────────────────
    def stitch(self, img1, img2):
        """
        单帧实时拼接。

        参数:
            img1: 左相机帧 BGR uint8 (h1, w1, 3)
            img2: 右相机帧 BGR uint8 (h2, w2, 3)

        返回:
            全景图 BGR uint8
        """
        # 左图加权放入画布
        canvas = np.zeros((self.pano_h, self.pano_w, 3), dtype=np.float32)
        canvas[:self.h1, :self.w1] = img1.astype(np.float32) * self.weight_left[:self.h1, :self.w1]

        # 右图透视变换（remap 代替 warpPerspective，快 3-5 倍）
        img2_w = cv2.remap(img2, self.map_x, self.map_y,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 可选曝光补偿
        if self._exposure_set:
            img2_f = np.clip(img2_w.astype(np.float32) * self.exp_gain + self.exp_bias, 0, 255)
        else:
            img2_f = img2_w.astype(np.float32)

        canvas += img2_f * self.weight_right

        # 裁剪
        result = np.clip(canvas, 0, 255).astype(np.uint8)
        return result[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2]

    @property
    def result_shape(self):
        """全景裁剪后尺寸 (height, width)。"""
        return (self.crop_y2 - self.crop_y1, self.crop_x2 - self.crop_x1)

    @staticmethod
    def from_calibration(calib_dict, blend_window=800):
        """从 load_calibration 返回的 dict 创建运行时。"""
        H = calib_dict['H']
        s1 = calib_dict['img1_shape']
        s2 = calib_dict['img2_shape']
        return StitchingRuntime(H, s1, s2, blend_window)


def main(img1_path, img2_path):
    """
    主流程：读取两张输入图像，执行拼接并保存结果。

    参数:
        img1_path: 第一张图像路径（左图）
        img2_path: 第二张图像路径（右图）
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None:
        raise FileNotFoundError(f"无法读取左图: {img1_path}")
    if img2 is None:
        raise FileNotFoundError(f"无法读取右图: {img2_path}")

    t_start = time.perf_counter()
    stitcher = Image_Stitching()
    final = stitcher.blending(img1, img2)
    t_stitch_total = time.perf_counter() - t_start

    t_save = time.perf_counter()
    cv2.imwrite('result.jpg', final)
    t_save_cost = time.perf_counter() - t_save

    print(f"拼接完成，结果已保存: result.jpg")
    if hasattr(stitcher, 'blend_range'):
        print(f"融合带列范围: {stitcher.blend_range[0]} ~ {stitcher.blend_range[1]}")
    if hasattr(stitcher, 'exposure_gains'):
        print(f"曝光补偿 BGR 增益: {[f'{g:.3f}' for g in stitcher.exposure_gains]}")
        print(f"曝光补偿 BGR 偏置: {[f'{b:.1f}' for b in stitcher.exposure_biases]}")
    print(f"耗时统计:")
    print(f"  特征配准 (SIFT+匹配+RANSAC): {stitcher.elapsed['registration']:.3f} s")
    print(f"  图像融合 (变换+掩膜+裁剪):   {stitcher.elapsed['blending']:.3f} s")
    print(f"  拼接核心耗时:               {stitcher.elapsed['total']:.3f} s")
    print(f"  保存结果:                   {t_save_cost:.3f} s")
    print(f"  总计:                       {t_stitch_total + t_save_cost:.3f} s")


if __name__ == '__main__':
    # 未传命令行参数时使用默认测试图；传参则使用指定路径
    default_left = "images/pair_001_left.png"
    default_right = "images/pair_001_right.png"

    if len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 1:
        print(f"未指定输入图像，使用默认: {default_left}, {default_right}")
        main(default_left, default_right)
    else:
        print("用法: python Image_Stitching.py [左图路径] [右图路径]")
        print(f"示例: python Image_Stitching.py {default_left} {default_right}")
