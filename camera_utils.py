"""
Basler 双相机封装。

按序列号打开两台 Basler USB 相机，提供连续采流与同步快照功能。
"""

import threading
import cv2
import numpy as np
from pypylon import pylon


class BaslerDualCamera:
    """Basler USB 双相机管理类。

    使用示例:
        cam = BaslerDualCamera(left_sn="40223748", right_sn="40226323")
        cam.open()
        cam.start()
        left, right = cam.get_frames()
        cam.stop()
    """

    def __init__(self, left_sn="40223748", right_sn="40226323"):
        """
        参数:
            left_sn: 左相机序列号（字符串）
            right_sn: 右相机序列号（字符串）
        """
        self.left_sn = left_sn
        self.right_sn = right_sn
        self.cam_left = None
        self.cam_right = None
        self._running = False
        self._latest_left = None
        self._latest_right = None
        self._lock = threading.Lock()
        self._thread = None
        self._last_frame_time = 0

    # ── 打开/配置 ──────────────────────────────────────────────────
    def open(self):
        """按序列号查找并打开双相机。"""
        tl = pylon.TlFactory.GetInstance()
        devs = tl.EnumerateDevices()
        if len(devs) < 2:
            raise RuntimeError(f"未检测到足够相机：仅发现 {len(devs)} 台，需要 2 台")

        for d in devs:
            sn = d.GetSerialNumber()
            if sn == self.left_sn:
                self.cam_left = pylon.InstantCamera(tl.CreateDevice(d))
            elif sn == self.right_sn:
                self.cam_right = pylon.InstantCamera(tl.CreateDevice(d))

        if self.cam_left is None:
            raise RuntimeError(f"未找到左相机 SN={self.left_sn}")
        if self.cam_right is None:
            raise RuntimeError(f"未找到右相机 SN={self.right_sn}")

        self.cam_left.Open()
        self.cam_right.Open()
        self._configure(self.cam_left)
        self._configure(self.cam_right)

    def _configure(self, cam):
        """配置相机参数。"""
        # 连续采集模式
        cam.AcquisitionMode.Value = "Continuous"

    # ── 连续采流 ──────────────────────────────────────────────────
    def start(self):
        """启动后台连续采流。"""
        if self._running:
            return
        self.cam_left.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.cam_right.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _grab_loop(self):
        """后台线程：持续取帧。"""
        while self._running:
            g1 = self.cam_left.RetrieveResult(100, pylon.TimeoutHandling_Return)
            g2 = self.cam_right.RetrieveResult(100, pylon.TimeoutHandling_Return)
            if (g1 is not None and g1.GrabSucceeded() and
                    g2 is not None and g2.GrabSucceeded()):
                with self._lock:
                    self._latest_left = self._to_bgr(g1.Array)
                    self._latest_right = self._to_bgr(g2.Array)
                import time
                self._last_frame_time = time.perf_counter()
            if g1 is not None:
                g1.Release()
            if g2 is not None:
                g2.Release()

    # ── 取帧 ──────────────────────────────────────────────────────
    def get_frames(self):
        """返回最新一帧 (left, right)，均为 BGR uint8 格式。无帧时返回 (None, None)。"""
        with self._lock:
            if self._latest_left is None:
                return None, None
            return self._latest_left.copy(), self._latest_right.copy()

    def has_frame(self):
        """是否有可用帧。"""
        with self._lock:
            return self._latest_left is not None

    # ── 同步快照（停止连续采流、单次抓取、恢复连续） ──────────────
    def get_snapshot(self):
        """
        抓取一对一帧对齐的同步快照。
        内部会暂时暂停连续采流，快照完成后恢复。
        """
        was_running = self._running
        self._stop_grabbing()

        # 切换到单帧模式
        self.cam_left.StartGrabbing(pylon.GrabStrategy_OneByOne)
        self.cam_right.StartGrabbing(pylon.GrabStrategy_OneByOne)

        # 丢弃前几帧以清空缓冲
        for _ in range(3):
            g = self.cam_left.RetrieveResult(2000, pylon.TimeoutHandling_Return)
            if g is not None:
                g.Release()
            g = self.cam_right.RetrieveResult(2000, pylon.TimeoutHandling_Return)
            if g is not None:
                g.Release()

        g1 = self.cam_left.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        g2 = self.cam_right.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        left = self._to_bgr(g1.Array)
        right = self._to_bgr(g2.Array)

        g1.Release()
        g2.Release()
        self.cam_left.StopGrabbing()
        self.cam_right.StopGrabbing()

        # 恢复连续采流
        if was_running:
            self.start()

        return left, right

    # ── 停止 ──────────────────────────────────────────────────────
    def _stop_grabbing(self):
        """内部：停止采流和线程。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        if self.cam_left is not None and self.cam_left.IsGrabbing():
            self.cam_left.StopGrabbing()
        if self.cam_right is not None and self.cam_right.IsGrabbing():
            self.cam_right.StopGrabbing()

    def stop(self):
        """停止采流并关闭相机。"""
        self._stop_grabbing()
        if self.cam_left is not None:
            self.cam_left.Close()
        if self.cam_right is not None:
            self.cam_right.Close()
        self.cam_left = None
        self.cam_right = None

    # ── 辅助 ──────────────────────────────────────────────────────
    @staticmethod
    def _to_bgr(arr):
        """将相机原始数组转为 BGR uint8 格式。"""
        if arr.ndim == 2:
            # 尝试 Bayer → BGR（彩色相机）
            try:
                return cv2.cvtColor(arr, cv2.COLOR_BayerRG2BGR)
            except cv2.error:
                # 单色相机：复制为三通道
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3:
            if arr.shape[2] == 3:
                # 假设已是 BGR
                return arr
            elif arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        return arr

    @staticmethod
    def list_devices():
        """列出所有已连接的 Basler 相机。返回 [(serial, friendly_name), ...]"""
        tl = pylon.TlFactory.GetInstance()
        return [(d.GetSerialNumber(), d.GetFriendlyName()) for d in tl.EnumerateDevices()]
