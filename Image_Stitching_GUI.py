"""
图像拼接 GUI 工具。

支持两种模式:
  - 本地文件：浏览加载图像 → 标定 → 拼接
  - 相机采集：双 Basler USB 相机实时预览 → 标定 → 实时拼接
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__ if '__file__' in dir() else 'Image_Stitching_GUI.py')))
from Image_Stitching import Image_Stitching, StitchingRuntime
from camera_utils import BaslerDualCamera


# ── 辅助 ───────────────────────────────────────────────────────────
def bgr_to_pil(img_bgr, max_w=None, max_h=None):
    """OpenCV BGR → PIL Image，可选等比例缩放。"""
    if img_bgr is None:
        return None
    img = img_bgr
    if max_w and max_h:
        h, w = img.shape[:2]
        s = min(max_w / w, max_h / h, 1.0)
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)))
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ── 可缩放/拖拽画布 ────────────────────────────────────────────────
class ZoomPanCanvas(tk.Canvas):
    """支持鼠标滚轮缩放 + 左键拖拽平移的图像画布。"""

    def __init__(self, parent, **kw):
        bg = kw.pop('bg', '#1e1e1e')
        super().__init__(parent, bg=bg, highlightthickness=0, **kw)
        self._img_bgr = None       # 原始 BGR 图像
        self._pil_full = None      # 原始分辨率 PIL 图
        self._tk_img = None
        self._img_id = None
        self._scale = 1.0          # 当前缩放比例
        self._offset_x = 0         # 视图偏移
        self._offset_y = 0

        # 事件绑定
        self.bind('<MouseWheel>', self._on_mousewheel)      # Windows
        self.bind('<Button-4>', self._on_mousewheel_up)     # Linux 滚轮 ↑
        self.bind('<Button-5>', self._on_mousewheel_down)   # Linux 滚轮 ↓
        self.bind('<ButtonPress-1>', self._on_drag_start)
        self.bind('<B1-Motion>', self._on_drag_move)
        self.bind('<Configure>', self._on_resize)

    # ── public API ────────────────────────────────────────────────
    def set_image(self, img_bgr):
        """设置要显示的图像（BGR uint8），自动 fit 到画布。"""
        if img_bgr is None:
            self._img_bgr = None
            self._pil_full = None
            self.delete('all')
            return
        self._img_bgr = img_bgr
        self._pil_full = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        self._fit_to_canvas()

    def reset_view(self):
        """回到 fit 状态。"""
        if self._img_bgr is None:
            return
        self._fit_to_canvas()

    # ── 内部 ──────────────────────────────────────────────────────
    def _on_resize(self, _event=None):
        """画布大小改变时重新 fit。"""
        if self._img_bgr is not None and self._scale == self._auto_fit_scale():
            self._fit_to_canvas()

    def _auto_fit_scale(self):
        """计算让图像完全适配画布的缩放比例。"""
        if self._img_bgr is None:
            return 1.0
        ih, iw = self._img_bgr.shape[:2]
        cw, ch = self.winfo_width() or 400, self.winfo_height() or 300
        return min(cw / iw, ch / ih, 1.0)

    def _fit_to_canvas(self):
        """缩放图像使其恰好填满画布（第一次显示或 reset 时调用）。"""
        s = self._auto_fit_scale()
        self._scale = s
        self._offset_x = 0
        self._offset_y = 0
        self._redraw()

    def _redraw(self):
        """根据当前 scale/offset 重绘图像。"""
        self.delete('all')
        if self._img_bgr is None:
            return
        ih, iw = self._img_bgr.shape[:2]
        new_w = max(1, int(iw * self._scale))
        new_h = max(1, int(ih * self._scale))
        pil_img = self._pil_full.resize((new_w, new_h), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(pil_img)
        canvas_w = self.winfo_width() or new_w
        canvas_h = self.winfo_height() or new_h
        cx = canvas_w // 2 + self._offset_x
        cy = canvas_h // 2 + self._offset_y
        self._img_id = self.create_image(cx, cy, anchor='center', image=self._tk_img)

    # ── 缩放 ──────────────────────────────────────────────────────
    def _on_mousewheel(self, event):
        self._zoom(event.delta / 120.0, event.x, event.y)

    def _on_mousewheel_up(self, event):
        self._zoom(1.0, event.x, event.y)

    def _on_mousewheel_down(self, event):
        self._zoom(-1.0, event.x, event.y)

    def _zoom(self, direction, cx, cy):
        """在鼠标位置缩放。direction > 0 放大，< 0 缩小。"""
        if self._img_bgr is None:
            return
        factor = 1.1 if direction > 0 else 0.9
        new_scale = self._scale * factor
        if not 0.05 <= new_scale <= 10.0:
            return

        # 以鼠标位置为锚点调整偏移
        self._offset_x = int(cx - (cx - self._offset_x) * new_scale / self._scale)
        self._offset_y = int(cy - (cy - self._offset_y) * new_scale / self._scale)
        self._scale = new_scale
        self._redraw()

    # ── 拖拽平移 ──────────────────────────────────────────────────
    def _on_drag_start(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag_move(self, event):
        dx = event.x - self._drag_x
        dy = event.y - self._drag_y
        self._drag_x = event.x
        self._drag_y = event.y
        self._offset_x += dx
        self._offset_y += dy
        # 延迟重绘以流畅跟手
        if not getattr(self, '_redraw_scheduled', False):
            self._redraw_scheduled = True
            self.after_idle(self._deferred_redraw)

    def _deferred_redraw(self):
        self._redraw_scheduled = False
        self._redraw()


# ── 容器 ────────────────────────────────────────────────────────────
class PanZoomImageFrame(ttk.Frame):
    """包裹 ZoomPanCanvas 的 Frame。"""

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.canvas = ZoomPanCanvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def set_image(self, img_bgr, max_w=None, max_h=None):
        self.canvas.set_image(img_bgr)


# ── 参数滑块行 ─────────────────────────────────────────────────────
class ParamRow(ttk.Frame):
    def __init__(self, parent, label, from_, to, initial, resolution=0.01, **kw):
        super().__init__(parent, **kw)
        self.var = tk.DoubleVar(value=initial)
        ttk.Label(self, text=label, width=14, anchor='e').pack(side=tk.LEFT, padx=(0, 4))
        self.scale = ttk.Scale(self, from_=from_, to=to, variable=self.var,
                               orient=tk.HORIZONTAL, length=200)
        self.scale.pack(side=tk.LEFT, padx=4)
        self.val_label = ttk.Label(self, text=f'{initial:.2f}', width=6)
        self.val_label.pack(side=tk.LEFT)
        self.resolution = resolution
        self.var.trace_add('write', self._on_change)

    def _on_change(self, *_):
        v = round(self.var.get() / self.resolution) * self.resolution
        digits = max(0, len(str(self.resolution).split('.')[-1].rstrip('0')))
        self.val_label.config(text=f'{v:.{digits}f}')


# ── 主窗口 ─────────────────────────────────────────────────────────
class StitchingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Image Stitching Tool')
        self.geometry('1200x780')
        self.minsize(1000, 650)
        self.configure(bg='#2d2d2d')
        self.attributes('-topmost', True)  # 强制窗口置顶
        self.after(500, lambda: self.attributes('-topmost', False))  # 半秒后取消置顶

        # 样式
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('.', background='#2d2d2d', foreground='#e0e0e0', fieldbackground='#3c3c3c')
        style.configure('TButton', padding=6, font=('Microsoft YaHei UI', 9))
        style.configure('TLabel', background='#2d2d2d', foreground='#e0e0e0')
        style.configure('TLabelframe', background='#2d2d2d', foreground='#cccccc')
        style.configure('TLabelframe.Label', background='#2d2d2d', foreground='#cccccc')
        style.configure('TNotebook', background='#2d2d2d', borderwidth=0)
        style.configure('TNotebook.Tab', padding=[12, 4], font=('Microsoft YaHei UI', 9))
        style.configure('TScale', background='#2d2d2d')
        style.configure('TEntry', fieldbackground='#3c3c3c', foreground='#e0e0e0')
        # Mode toggle accent style
        style.configure('Mode.TButton', font=('Microsoft YaHei UI', 9, 'bold'), padding=8)
        style.map('Mode.TButton',
                  background=[('active', '#3c3c3c'), ('!active', '#2d2d2d')])

        # 共享状态
        self.img1 = None
        self.img2 = None
        self.result = None
        self.cached_H = None
        self.calib_info = None
        self.runtime = None          # StitchingRuntime 实例
        self.camera = None           # BaslerDualCamera 实例
        self._live_stitch_active = False
        self._mode = 'file'          # 'file' | 'camera'
        self._cam_sn_left = None     # 当前左相机 SN
        self._cam_sn_right = None    # 当前右相机 SN

        self._build_header()
        self._build_file_mode()
        self._build_camera_mode()
        self._build_status_bar()

        # 初始显示文件模式
        self.file_mode_frame.pack(fill=tk.BOTH, expand=True)

        # 确保窗口显示在最前方
        self.lift()
        self.focus_force()

        # 注册窗口关闭回调
        self.protocol('WM_DELETE_WINDOW', self.on_close)

    # ── 顶部标题 + 模式切换 ───────────────────────────────────────
    def _build_header(self):
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=10, pady=(8, 4))
        ttk.Label(header, text='Image Stitching Tool',
                  font=('Microsoft YaHei UI', 14, 'bold')).pack(side=tk.LEFT)

        self.mode_file_btn = ttk.Button(header, text='本地文件', style='Mode.TButton',
                                        command=lambda: self._switch_mode('file'))
        self.mode_file_btn.pack(side=tk.RIGHT, padx=(2, 0))
        self.mode_camera_btn = ttk.Button(header, text='相机采集', command=lambda: self._switch_mode('camera'))
        self.mode_camera_btn.pack(side=tk.RIGHT, padx=2)

        ttk.Label(header, text='SIFT + 曝光补偿 + 自适应融合 | 双图拼接',
                  foreground='#888').pack(side=tk.LEFT, padx=12)

    def _switch_mode(self, mode):
        """在文件模式与相机模式之间切换。"""
        if self._live_stitch_active:
            self._stop_live_stitch()

        if mode == 'camera':
            self.file_mode_frame.pack_forget()
            self.camera_mode_frame.pack(fill=tk.BOTH, expand=True)
            self.mode_file_btn.configure(style='TButton')
            self.mode_camera_btn.configure(style='Mode.TButton')
            self._mode = 'camera'
            self._open_camera()
        else:
            self._close_camera()
            self.camera_mode_frame.pack_forget()
            self.file_mode_frame.pack(fill=tk.BOTH, expand=True)
            self.mode_camera_btn.configure(style='TButton')
            self.mode_file_btn.configure(style='Mode.TButton')
            self._mode = 'file'

    # ═════════════════════════════════════════════════════════════════
    # 文件模式
    # ═════════════════════════════════════════════════════════════════

    def _build_file_mode(self):
        frame = ttk.Frame(self)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)
        self.file_mode_frame = frame

        self._build_file_control_panel(frame)
        self._build_file_preview_panel(frame)

    def _build_file_control_panel(self, parent):
        panel = ttk.Frame(parent, width=360)
        panel.grid(row=0, column=0, sticky='nsw', padx=(0, 8))
        panel.grid_propagate(False)

        # ── 文件选择 ──
        ff = ttk.Labelframe(panel, text='输入图像', padding=8)
        ff.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(ff, text='左图 (参考图)').pack(anchor=tk.W)
        r = ttk.Frame(ff)
        r.pack(fill=tk.X, pady=(2, 6))
        self.left_path_var = tk.StringVar()
        ttk.Entry(r, textvariable=self.left_path_var, font=('Consolas', 8)).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(r, text='浏览', command=lambda: self._browse_image('left')).pack(side=tk.RIGHT, padx=(4, 0))

        ttk.Label(ff, text='右图 (待变换)').pack(anchor=tk.W)
        r2 = ttk.Frame(ff)
        r2.pack(fill=tk.X, pady=(2, 6))
        self.right_path_var = tk.StringVar()
        ttk.Entry(r2, textvariable=self.right_path_var, font=('Consolas', 8)).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(r2, text='浏览', command=lambda: self._browse_image('right')).pack(side=tk.RIGHT, padx=(4, 0))

        ttk.Button(ff, text='加载图像', command=self._load_images).pack(anchor=tk.W, pady=(0, 4))

        tf = ttk.Frame(ff)
        tf.pack(fill=tk.X)
        self.thumb_left = ttk.Label(tf, text='(左图预览)', anchor=tk.CENTER, background='#3c3c3c', width=24)
        self.thumb_left.pack(side=tk.LEFT, padx=(0, 2))
        self.thumb_right = ttk.Label(tf, text='(右图预览)', anchor=tk.CENTER, background='#3c3c3c', width=24)
        self.thumb_right.pack(side=tk.LEFT, padx=(2, 0))

        # ── 参数 ──
        pf = ttk.Labelframe(panel, text='拼接参数', padding=8)
        pf.pack(fill=tk.X, pady=(0, 8))
        self.param_ratio = ParamRow(pf, '匹配比率', 0.5, 0.99, 0.85, 0.01)
        self.param_ratio.pack(fill=tk.X, pady=2)
        self.param_min = ParamRow(pf, '最少匹配点数', 4, 50, 10, 1)
        self.param_min.pack(fill=tk.X, pady=2)
        self.param_blend = ParamRow(pf, '融合带宽度', 100, 2000, 800, 10)
        self.param_blend.pack(fill=tk.X, pady=2)

        # ── 标定 ──
        cf = ttk.Labelframe(panel, text='标定参数 (固定相机可复用)', padding=8)
        cf.pack(fill=tk.X, pady=(0, 8))
        br = ttk.Frame(cf)
        br.pack(fill=tk.X)
        self.calib_save_btn = ttk.Button(br, text='标定并保存', command=self._save_calibration, width=12)
        self.calib_save_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.calib_load_btn = ttk.Button(br, text='加载标定', command=self._load_calibration, width=12)
        self.calib_load_btn.pack(side=tk.LEFT, padx=4)
        self.calib_clear_btn = ttk.Button(br, text='清除标定', command=self._clear_calibration,
                                          width=12, state=tk.DISABLED)
        self.calib_clear_btn.pack(side=tk.RIGHT)
        self.calib_status_var = tk.StringVar(value='未加载标定 - 每次拼接将重新匹配')
        ttk.Label(cf, textvariable=self.calib_status_var, foreground='#888',
                  font=('Microsoft YaHei UI', 8)).pack(anchor=tk.W, pady=(4, 0))

        # ── 操作 ──
        bf = ttk.Frame(panel)
        bf.pack(fill=tk.X, pady=(0, 8))
        self.stitch_btn = ttk.Button(bf, text='执行拼接', command=self._run_stitching, width=14)
        self.stitch_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.save_btn = ttk.Button(bf, text='保存结果', command=self._save_result,
                                   width=14, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=4)

        # ── 信息 ──
        inf = ttk.Labelframe(panel, text='拼接信息', padding=8)
        inf.pack(fill=tk.BOTH, expand=True)
        self.info_text = tk.Text(inf, height=12, bg='#1e1e1e', fg='#c0c0c0',
                                 insertbackground='white', font=('Consolas', 9),
                                 relief=tk.FLAT, borderwidth=4, wrap=tk.WORD, state=tk.DISABLED)
        self.info_text.pack(fill=tk.BOTH, expand=True)

    def _build_file_preview_panel(self, parent):
        nb = ttk.Notebook(parent)
        nb.grid(row=0, column=1, sticky='nsew')
        self.match_frame = PanZoomImageFrame(nb)
        nb.add(self.match_frame, text=' 特征匹配 ')
        self.result_frame = PanZoomImageFrame(nb)
        nb.add(self.result_frame, text=' 拼接结果 ')

    # ═════════════════════════════════════════════════════════════════
    # 相机模式
    # ═════════════════════════════════════════════════════════════════

    def _build_camera_mode(self):
        frame = ttk.Frame(self)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)
        self.camera_mode_frame = frame

        # 左面板
        left = ttk.Frame(frame, width=380)
        left.grid(row=0, column=0, sticky='nsw', padx=(0, 8))
        left.grid_propagate(False)

        # 双路预览
        prev_f = ttk.Labelframe(left, text='相机预览', padding=4)
        prev_f.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        prev_f.rowconfigure(0, weight=1)
        prev_f.rowconfigure(1, weight=1)
        prev_f.columnconfigure(0, weight=1)

        self.cam_left_label = ttk.Label(prev_f, text='左相机 (未连接)', anchor=tk.CENTER,
                                         background='#1e1e1e')
        self.cam_left_label.grid(row=0, column=0, sticky='nsew', pady=(0, 2))
        self.cam_right_label = ttk.Label(prev_f, text='右相机 (未连接)', anchor=tk.CENTER,
                                          background='#1e1e1e')
        self.cam_right_label.grid(row=1, column=0, sticky='nsew', pady=(2, 0))

        # 相机参数
        cp = ttk.Labelframe(left, text='拼接参数', padding=4)
        cp.pack(fill=tk.X, pady=(0, 6))
        self.cam_blend = ParamRow(cp, '融合带宽度', 100, 2000, 800, 10)
        self.cam_blend.pack(fill=tk.X, pady=2)

        # 相机按钮
        cb = ttk.Frame(left)
        cb.pack(fill=tk.X, pady=(0, 6))

        self.cam_calib_btn = ttk.Button(cb, text='抓取并标定', command=self._cam_calibrate, width=13)
        self.cam_calib_btn.pack(side=tk.LEFT, padx=(0, 3))
        self.cam_live_btn = ttk.Button(cb, text='开始实时拼接', command=self._toggle_live_stitch, width=13)
        self.cam_live_btn.pack(side=tk.LEFT, padx=3)

        self.cam_load_btn = ttk.Button(cb, text='加载标定', command=self._cam_load_calib, width=10)
        self.cam_load_btn.pack(side=tk.RIGHT, padx=(3, 0))
        self.cam_save_btn = ttk.Button(cb, text='保存标定', command=self._cam_save_calib, width=10)
        self.cam_save_btn.pack(side=tk.RIGHT, padx=3)

        # 相机分配
        af = ttk.Labelframe(left, text='相机分配 (SN)', padding=4)
        af.pack(fill=tk.X, pady=(0, 6))
        ar = ttk.Frame(af)
        ar.pack(fill=tk.X)
        ttk.Label(ar, text='左', width=2).pack(side=tk.LEFT)
        self.cam_sn_left_label = ttk.Label(ar, text='--', foreground='#aaa', font=('Consolas', 9))
        self.cam_sn_left_label.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(ar, text='右', width=2).pack(side=tk.LEFT)
        self.cam_sn_right_label = ttk.Label(ar, text='--', foreground='#aaa', font=('Consolas', 9))
        self.cam_sn_right_label.pack(side=tk.LEFT, padx=(0, 8))
        self.cam_swap_btn = ttk.Button(ar, text='交换', command=self._cam_swap, width=6, state=tk.DISABLED)
        self.cam_swap_btn.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(ar, text='重连', command=self._cam_reconnect, width=5).pack(side=tk.RIGHT)

        # 相机信息
        cf = ttk.Labelframe(left, text='标定状态', padding=4)
        cf.pack(fill=tk.X)
        self.cam_calib_status_var = tk.StringVar(value='相机未标定 - 请点击"抓取并标定"')
        ttk.Label(cf, textvariable=self.cam_calib_status_var, foreground='#888',
                  font=('Microsoft YaHei UI', 8)).pack(anchor=tk.W)

        # 右侧结果区
        right = ttk.Frame(frame)
        right.grid(row=0, column=1, sticky='nsew')
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self.cam_result_frame = PanZoomImageFrame(right)
        self.cam_result_frame.grid(row=0, column=0, sticky='nsew')

        # FPS 标签
        self.cam_fps_var = tk.StringVar(value='')
        ttk.Label(right, textvariable=self.cam_fps_var, foreground='#888',
                  font=('Consolas', 9)).grid(row=1, column=0, sticky='e')

    # ── 相机生命周期 ──────────────────────────────────────────────
    def _open_camera(self):
        try:
            devs = BaslerDualCamera.list_devices()
            if len(devs) < 2:
                messagebox.showwarning('相机不足', f'检测到 {len(devs)} 台相机，需要 2 台。')
                return

            # 记录当前分配
            self._cam_sn_left, self._cam_sn_right = devs[0][0], devs[1][0]

            self._start_camera()
        except Exception as e:
            messagebox.showerror('相机错误', str(e))
            self._set_status(f'相机连接失败: {e}')

    def _start_camera(self):
        """根据 _cam_sn_left / _cam_sn_right 启动相机。"""
        if self._cam_sn_left is None or self._cam_sn_right is None:
            return
        # 先保存 SN，因为 _close_camera 会清空它们
        sn_l, sn_r = self._cam_sn_left, self._cam_sn_right
        self._close_camera()
        self._cam_sn_left, self._cam_sn_right = sn_l, sn_r
        self.camera = BaslerDualCamera(self._cam_sn_left, self._cam_sn_right)
        self.camera.open()
        self.camera.start()

        # 更新 UI 标签
        devs = BaslerDualCamera.list_devices()
        name_left = next((n for sn, n in devs if sn == self._cam_sn_left), self._cam_sn_left)
        name_right = next((n for sn, n in devs if sn == self._cam_sn_right), self._cam_sn_right)
        self.cam_left_label.configure(text=f'{name_left} (SN:{self._cam_sn_left})')
        self.cam_right_label.configure(text=f'{name_right} (SN:{self._cam_sn_right})')
        self.cam_sn_left_label.configure(text=self._cam_sn_left, foreground='#8f8')
        self.cam_sn_right_label.configure(text=self._cam_sn_right, foreground='#8f8')
        self.cam_swap_btn.config(state=tk.NORMAL)
        self._set_status(f'相机已连接: {name_left} / {name_right}')

        # 清除旧标定/运行时（相机分配改变后旧的 H 无效）
        self.runtime = None
        self.cached_H = None
        self.calib_info = None
        self.cam_calib_status_var.set('相机未标定 - 请点击"抓取并标定"')

        # 启动预览循环
        self._cam_preview_loop()

    def _cam_swap(self):
        """交换左右相机分配。"""
        if self._cam_sn_left is None or self._cam_sn_right is None:
            return
        self._stop_live_stitch()
        self._cam_sn_left, self._cam_sn_right = self._cam_sn_right, self._cam_sn_left
        self._start_camera()
        self._set_status(f'相机已交换: 左={self._cam_sn_right} → {self._cam_sn_left}')

    def _cam_reconnect(self):
        """重新连接相机（适用于更换 USB 后重新检测）。"""
        self._stop_live_stitch()
        devs = BaslerDualCamera.list_devices()
        if len(devs) < 2:
            messagebox.showwarning('相机不足', f'检测到 {len(devs)} 台相机，需要 2 台。')
            return
        self._cam_sn_left, self._cam_sn_right = devs[0][0], devs[1][0]
        self._start_camera()

    def _close_camera(self):
        self._stop_live_stitch()
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
            self.camera = None
        self.cam_left_label.configure(image='', text='左相机 (未连接)')
        self.cam_left_label.image = None
        self.cam_right_label.configure(image='', text='右相机 (未连接)')
        self.cam_right_label.image = None
        self.cam_sn_left_label.configure(text='--', foreground='#aaa')
        self.cam_sn_right_label.configure(text='--', foreground='#aaa')
        self.cam_swap_btn.config(state=tk.DISABLED)
        self._cam_sn_left = None
        self._cam_sn_right = None

    # ── 相机预览刷新 ──────────────────────────────────────────────
    def _cam_preview_loop(self):
        if self._mode != 'camera' or self.camera is None:
            return
        l, r = self.camera.get_frames()

        # 相机预览缩略图（始终快速更新）
        if l is not None:
            thumb = ImageTk.PhotoImage(bgr_to_pil(l, max_w=350, max_h=200))
            self.cam_left_label.configure(image=thumb, text='')
            self.cam_left_label.image = thumb
        if r is not None:
            thumb = ImageTk.PhotoImage(bgr_to_pil(r, max_w=350, max_h=200))
            self.cam_right_label.configure(image=thumb, text='')
            self.cam_right_label.image = thumb

        # 实时拼接：仅在上一帧完成后才触发新帧，避免帧堆积
        if (self._live_stitch_active and self.runtime is not None
                and l is not None and r is not None
                and not getattr(self, '_stitch_busy', False)):
            self._stitch_busy = True
            self.after(1, self._do_live_stitch, l.copy(), r.copy())

        self.after(30, self._cam_preview_loop)

    def _do_live_stitch(self, left, right):
        """单帧实时拼接，完成后释放 busy 标记。"""
        if not self._live_stitch_active:
            self._stitch_busy = False
            return
        try:
            t0 = time.perf_counter()
            result = self.runtime.stitch(left, right)
            t = time.perf_counter() - t0

            # 缩小显示以减轻 UI 负载（全分辨率显示太重）
            h, w = result.shape[:2]
            scale = min(650 / w, 420 / h, 1.0)
            if scale < 1.0:
                disp = cv2.resize(result, (int(w * scale), int(h * scale)))
            else:
                disp = result

            self.cam_result_frame.set_image(disp)
            fps = 1.0 / t if t > 0 else 0
            self.cam_fps_var.set(f'stitch: {t*1000:.0f} ms ({fps:.0f} FPS)')
        except Exception:
            pass
        self._stitch_busy = False

    # ── 相机标定 ──────────────────────────────────────────────────
    def _cam_calibrate(self):
        if self.camera is None:
            return
        self.cam_calib_btn.config(state=tk.DISABLED, text='标定中...')
        self._set_status('正在抓取并标定...')
        threading.Thread(target=self._cam_calib_thread, daemon=True).start()

    def _cam_calib_thread(self):
        try:
            snap_l, snap_r = self.camera.get_snapshot()
            stitcher = Image_Stitching()
            H = stitcher.calibrate(snap_l, snap_r)
            if H is None:
                self.after(0, lambda: messagebox.showerror('标定失败', '特征匹配不足'))
                self.after(0, lambda: self.cam_calib_btn.config(state=tk.NORMAL, text='抓取并标定'))
                return

            self.cached_H = H
            self.calib_info = {
                'img1_shape': snap_l.shape[:2],
                'img2_shape': snap_r.shape[:2],
            }
            blend = int(self.cam_blend.var.get())
            self.runtime = StitchingRuntime(H, snap_l.shape, snap_r.shape, blend)
            self.after(0, self._on_cam_calib_done, stitcher.num_good_matches)
        except Exception as e:
            self.after(0, lambda: self.cam_calib_btn.config(state=tk.NORMAL, text='抓取并标定'))
            self.after(0, lambda: messagebox.showerror('标定失败', str(e)))

    def _on_cam_calib_done(self, num_matches):
        self.cam_calib_btn.config(state=tk.NORMAL, text='抓取并标定')
        rs = self.runtime.result_shape
        self.cam_calib_status_var.set(
            f'标定完成! 匹配点: {num_matches}  结果: {rs[1]}x{rs[0]}  融合带: {self.runtime.blend_range}')
        self._set_status(f'标定完成 - {num_matches} 匹配点，可开始实时拼接')

    def _cam_load_calib(self):
        path = filedialog.askopenfilename(
            title='加载标定参数', filetypes=[('标定文件', '*.npz')])
        if not path:
            return
        calib = Image_Stitching.load_calibration(path)
        if calib is None:
            messagebox.showerror('加载失败', '无法读取标定文件')
            return
        self.cached_H = calib['H']
        self.calib_info = {
            'img1_shape': calib['img1_shape'],
            'img2_shape': calib['img2_shape'],
        }
        blend = int(self.cam_blend.var.get())
        self.runtime = StitchingRuntime(calib['H'], calib['img1_shape'], calib['img2_shape'], blend)
        rs = self.runtime.result_shape
        self.cam_calib_status_var.set(
            f'标定已加载: {os.path.basename(path)}  结果: {rs[1]}x{rs[0]}')
        self._set_status(f'标定已加载: {os.path.basename(path)}')

    def _cam_save_calib(self):
        if self.cached_H is None or self.calib_info is None:
            messagebox.showwarning('未标定', '请先完成标定')
            return
        path = filedialog.asksaveasfilename(
            title='保存标定参数', defaultextension='.npz',
            filetypes=[('标定文件', '*.npz')])
        if not path:
            return
        Image_Stitching.save_calibration(
            path, self.cached_H,
            (self.calib_info['img1_shape'][0], self.calib_info['img1_shape'][1], 3),
            (self.calib_info['img2_shape'][0], self.calib_info['img2_shape'][1], 3))
        self._set_status(f'标定已保存: {os.path.basename(path)}')

    # ── 实时拼接 ──────────────────────────────────────────────────
    def _toggle_live_stitch(self):
        if self._live_stitch_active:
            self._stop_live_stitch()
        else:
            if self.runtime is None:
                messagebox.showwarning('未标定', '请先完成标定再开始实时拼接')
                return
            self._live_stitch_active = True
            self.cam_live_btn.config(text='停止实时拼接')
            self._set_status('实时拼接运行中...')

    def _stop_live_stitch(self):
        self._live_stitch_active = False
        self._stitch_busy = False
        self.cam_live_btn.config(text='开始实时拼接')
        if hasattr(self, 'cam_fps_var'):
            self.cam_fps_var.set('')

    # ═════════════════════════════════════════════════════════════════
    # 文件模式 — 图像加载
    # ═════════════════════════════════════════════════════════════════
    def _browse_image(self, side):
        path = filedialog.askopenfilename(
            title=f'选择{"左" if side == "left" else "右"}图',
            filetypes=[('图像文件', '*.png *.jpg *.jpeg *.bmp *.tiff'), ('所有文件', '*.*')])
        if path:
            if side == 'left':
                self.left_path_var.set(path)
            else:
                self.right_path_var.set(path)

    def _load_images(self):
        left_path = self.left_path_var.get()
        right_path = self.right_path_var.get()
        if not left_path or not right_path:
            messagebox.showwarning('缺少路径', '请先选择左右两张图像。')
            return
        self.img1 = cv2.imread(left_path)
        self.img2 = cv2.imread(right_path)
        if self.img1 is None:
            messagebox.showerror('读取失败', f'无法加载左图:\n{left_path}')
            return
        if self.img2 is None:
            messagebox.showerror('读取失败', f'无法加载右图:\n{right_path}')
            return
        self._show_thumb(self.thumb_left, self.img1)
        self._show_thumb(self.thumb_right, self.img2)
        s1 = f'{self.img1.shape[1]}x{self.img1.shape[0]}'
        s2 = f'{self.img2.shape[1]}x{self.img2.shape[0]}'
        self._set_status(f'已加载 - 左图 {s1} | 右图 {s2}')
        self._info_clear()
        self._info_append(f'左图: {left_path} ({s1})')
        self._info_append(f'右图: {right_path} ({s2})')
        self.result = None
        self.save_btn.config(state=tk.DISABLED)
        self.match_frame.set_image(None)
        self.result_frame.set_image(None)

    def _show_thumb(self, label, img):
        thumb = ImageTk.PhotoImage(bgr_to_pil(img, max_w=140, max_h=90))
        if thumb:
            label.configure(image=thumb, text='')
            label.image = thumb
        else:
            label.configure(image='', text='(加载失败)')

    # ═════════════════════════════════════════════════════════════════
    # 文件模式 — 拼接
    # ═════════════════════════════════════════════════════════════════
    def _run_stitching(self):
        if self.img1 is None or self.img2 is None:
            messagebox.showwarning('未加载图像', '请先加载左右两张图像。')
            return
        self.stitch_btn.config(state=tk.DISABLED, text='拼接中...')
        self._set_status('正在执行拼接...')
        threading.Thread(target=self._stitch_thread, daemon=True).start()

    def _stitch_thread(self):
        try:
            ratio = self.param_ratio.var.get()
            min_match = int(self.param_min.var.get())
            blend = int(self.param_blend.var.get())
            stitcher = Image_Stitching()
            stitcher.ratio = ratio
            stitcher.min_match = min_match
            stitcher.smoothing_window_size = blend

            using_calib = self.cached_H is not None
            if using_calib and self.calib_info is not None:
                cur_h1, cur_w1 = self.img1.shape[:2]
                cur_h2, cur_w2 = self.img2.shape[:2]
                cal_h1, cal_w1 = self.calib_info['img1_shape']
                cal_h2, cal_w2 = self.calib_info['img2_shape']
                if (cur_h1, cur_w1, cur_h2, cur_w2) != (cal_h1, cal_w1, cal_h2, cal_w2):
                    self.after(0, lambda: messagebox.showwarning(
                        '分辨率不匹配',
                        f'当前图像分辨率与标定不一致!\n\n'
                        f'标定: 左 {cal_w1}x{cal_h1}  右 {cal_w2}x{cal_h2}\n'
                        f'当前: 左 {cur_w1}x{cur_h1}  右 {cur_w2}x{cur_h2}\n\n'
                        f'将自动回退到完整 SIFT 匹配。'))
                    self.cached_H = None
                    using_calib = False

            t0 = time.perf_counter()
            if using_calib:
                result = stitcher.blending(self.img1, self.img2, H=self.cached_H)
            else:
                result = stitcher.blending(self.img1, self.img2)
            total_time = time.perf_counter() - t0
            self.after(0, self._on_stitch_done, stitcher, result, total_time)
        except Exception as e:
            self.after(0, self._on_stitch_error, str(e))

    def _on_stitch_done(self, stitcher, result, total_time):
        self.result = result
        self.stitch_btn.config(state=tk.NORMAL, text='执行拼接')
        self.save_btn.config(state=tk.NORMAL)
        if hasattr(stitcher, 'match_vis') and stitcher.match_vis is not None:
            self.match_frame.set_image(stitcher.match_vis)
        self.result_frame.set_image(result)
        self._info_clear()
        using_calib = getattr(stitcher, '_using_cached_H', False)
        tag = '(使用标定 H, 跳过 SIFT) ' if using_calib else ''
        self._info_append(f'拼接成功! {tag}总耗时: {total_time:.3f} s')
        elapsed = stitcher.elapsed
        if stitcher.num_good_matches > 0:
            self._info_append(f'  匹配点数: {stitcher.num_good_matches}')
        self._info_append(f'  配准耗时: {elapsed["registration"]:.3f} s')
        self._info_append(f'  融合耗时: {elapsed["blending"]:.3f} s')
        if hasattr(stitcher, 'blend_range'):
            self._info_append(f'  融合带: {stitcher.blend_range[0]} ~ {stitcher.blend_range[1]}')
        if hasattr(stitcher, 'exposure_gains'):
            gains = ', '.join(f'{g:.3f}' for g in stitcher.exposure_gains)
            biases = ', '.join(f'{b:.1f}' for b in stitcher.exposure_biases)
            self._info_append(f'  曝光补偿增益 (BGR): {gains}')
            self._info_append(f'  曝光补偿偏置 (BGR): {biases}')
        rh, rw = result.shape[:2]
        self._info_append(f'  结果尺寸: {rw}x{rh}')
        self._set_status(f'拼接完成 - 总耗时 {total_time:.3f} s, 结果 {rw}x{rh}')

    def _on_stitch_error(self, msg):
        self.stitch_btn.config(state=tk.NORMAL, text='执行拼接')
        messagebox.showerror('拼接失败', msg)
        self._set_status(f'拼接失败 - {msg}')

    # ═════════════════════════════════════════════════════════════════
    # 文件模式 — 标定
    # ═════════════════════════════════════════════════════════════════
    def _save_calibration(self):
        if self.img1 is None or self.img2 is None:
            messagebox.showwarning('未加载图像', '请先加载用于标定的左右两张图像。')
            return
        path = filedialog.asksaveasfilename(
            title='保存标定参数', defaultextension='.npz',
            filetypes=[('标定文件', '*.npz'), ('所有文件', '*.*')])
        if not path:
            return
        self.calib_save_btn.config(state=tk.DISABLED, text='标定中...')
        self._set_status('正在标定...')
        threading.Thread(target=self._calib_save_thread, args=(path,), daemon=True).start()

    def _calib_save_thread(self, path):
        try:
            stitcher = Image_Stitching()
            stitcher.ratio = self.param_ratio.var.get()
            stitcher.min_match = int(self.param_min.var.get())
            H = stitcher.calibrate(self.img1, self.img2)
            if H is None:
                self.after(0, self._on_calib_error, '特征匹配不足，无法完成标定')
                return
            Image_Stitching.save_calibration(path, H, self.img1.shape, self.img2.shape)
            self.cached_H = H
            self.calib_info = {
                'path': path,
                'img1_shape': self.img1.shape[:2],
                'img2_shape': self.img2.shape[:2],
            }
            self.after(0, self._on_calib_saved, path, stitcher.num_good_matches)
        except Exception as e:
            self.after(0, self._on_calib_error, str(e))

    def _on_calib_saved(self, path, num_matches):
        self.calib_save_btn.config(state=tk.NORMAL, text='标定并保存')
        self.calib_clear_btn.config(state=tk.NORMAL)
        s1 = f'{self.calib_info["img1_shape"][1]}x{self.calib_info["img1_shape"][0]}'
        s2 = f'{self.calib_info["img2_shape"][1]}x{self.calib_info["img2_shape"][0]}'
        self.calib_status_var.set(f'标定已加载 ({num_matches} 匹配点) - 左 {s1} 右 {s2}')
        self._info_clear()
        self._info_append(f'标定完成! 匹配点数: {num_matches}')
        self._info_append(f'标定文件: {path}')
        self._info_append(f'标定分辨率: 左 {s1}, 右 {s2}')
        self._info_append('后续拼接将跳过 SIFT 匹配，直接使用保存的 H')
        self._set_status(f'标定已保存至 {path}，后续拼接将跳过 SIFT')

    def _load_calibration(self):
        path = filedialog.askopenfilename(
            title='加载标定参数', filetypes=[('标定文件', '*.npz'), ('所有文件', '*.*')])
        if not path:
            return
        calib = Image_Stitching.load_calibration(path)
        if calib is None:
            messagebox.showerror('加载失败', f'无法读取标定文件:\n{path}')
            return
        self.cached_H = calib['H']
        self.calib_info = {
            'path': path,
            'img1_shape': calib['img1_shape'],
            'img2_shape': calib['img2_shape'],
        }
        self.calib_clear_btn.config(state=tk.NORMAL)
        s1 = f'{calib["img1_shape"][1]}x{calib["img1_shape"][0]}'
        s2 = f'{calib["img2_shape"][1]}x{calib["img2_shape"][0]}'
        self.calib_status_var.set(f'标定已加载 - 左 {s1} 右 {s2}')
        self._info_clear()
        self._info_append(f'已加载标定: {path}')
        self._info_append(f'标定分辨率: 左 {s1}, 右 {s2}')
        self._info_append('后续拼接将跳过 SIFT 匹配，直接使用保存的 H')
        self._set_status(f'标定已加载: {os.path.basename(path)}')

    def _clear_calibration(self):
        self.cached_H = None
        self.calib_info = None
        self.calib_clear_btn.config(state=tk.DISABLED)
        self.calib_status_var.set('未加载标定 - 每次拼接将重新匹配')
        self._set_status('标定已清除，将恢复完整 SIFT 匹配流程')

    def _on_calib_error(self, msg):
        self.calib_save_btn.config(state=tk.NORMAL, text='标定并保存')
        messagebox.showerror('标定失败', msg)
        self._set_status(f'标定失败 - {msg}')

    # ═════════════════════════════════════════════════════════════════
    # 共享 — 保存结果 / 状态栏 / 信息
    # ═════════════════════════════════════════════════════════════════
    def _save_result(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            title='保存拼接结果', defaultextension='.jpg',
            filetypes=[('JPEG', '*.jpg'), ('PNG', '*.png'), ('BMP', '*.bmp')])
        if path:
            cv2.imwrite(path, self.result)
            self._info_append(f'已保存: {path}')
            self._set_status(f'结果已保存至 {path}')

    def _build_status_bar(self):
        self.status_var = tk.StringVar(value='就绪 - 当前: 本地文件模式')
        ttk.Label(self, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor=tk.W, padding=(8, 2)).pack(fill=tk.X, side=tk.BOTTOM)

    def _set_status(self, text):
        self.status_var.set(text)

    def _info_clear(self):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        self.info_text.config(state=tk.DISABLED)

    def _info_append(self, text):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, text + '\n')
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)

    # ── 退出清理 ──────────────────────────────────────────────────
    def on_close(self):
        """窗口关闭时的清理回调。"""
        self._close_camera()
        tk.Tk.destroy(self)


if __name__ == '__main__':
    app = StitchingApp()
    app.mainloop()
