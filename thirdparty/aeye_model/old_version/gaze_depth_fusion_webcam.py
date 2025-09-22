#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gaze_depth_fusion_webcam.py  (PyAV 우선 저지연 캡처 + OpenCV 폴백)

Eye(Webcam/RTSP) + World(Webcam/RTSP) 입력으로
- MediaPipe FaceMesh 기반 양안 시선 추정
- Video-Depth-Anything 스트리밍(metric)으로 월드 깊이맵 생성/갱신
- 깊이와 레이-월드 교차로 응시점 월드 프레임 좌표 오버레이

입력:
  - "video=장치명" / "rtsp://..." / "http..." → PyAV(FFmpeg) 경로
  - 숫자 인덱스(예: 2, 1) → OpenCV 폴백(저지연 그랩버)

실행 예 (Windows, dshow, PyAV):
  python gaze_depth_fusion_webcam.py ^
    --eye_cam "video=Iriun Webcam" --world_cam "video=USB2.0 Camera" ^
    --av_backend_eye dshow --pixel_format_eye mjpeg --width_eye 640 --height_eye 480 --fps_eye 30 ^
    --av_backend_world dshow --pixel_format_world mjpeg --width_world 512 --height_world 384 --fps_world 30 ^
    --vda_dir .\Video-Depth-Anything --encoder vits --metric --input_size 392 --max_res 960 ^
    --segment_secs 1 --target_fps 20 --show_fps ^
    --fx_w (Get-Content .\data\intrinsics_world_512x384.json | ConvertFrom-Json).fx ^
    --fy_w (Get-Content .\data\intrinsics_world_512x384.json | ConvertFrom-Json).fy ^
    --cx_w (Get-Content .\data\intrinsics_world_512x384.json | ConvertFrom-Json).cx ^
    --cy_w (Get-Content .\data\intrinsics_world_512x384.json | ConvertFrom-Json).cy ^
    --extrinsic_json .\data\extrinsic_eye_to_world.json
"""

import os
import sys
import json
import time
import math
import glob
import argparse
import threading
import subprocess
from typing import Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp

# 프로젝트 구조 기준: 루트에 이 파일, scripts/ 아래에 유틸 3개
from scripts.io_open import BACKENDS, open_source                  # OpenCV 폴백용
from scripts.utils_framegrabber import LatestFrameGrabber          # OpenCV 폴백용
try:
    from scripts.pyav_grabber import PyAvLatestGrabber             # PyAV 저지연 그랩버
    _HAS_PYAV = True
except Exception:
    _HAS_PYAV = False

# =========================
# MediaPipe FaceMesh 설정
# =========================
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_IDX  = [33, 133, 160, 159, 158, 157, 173, 246, 161, 163, 144, 145, 153, 154, 155, 33]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 466, 388, 390, 373, 374, 380, 381, 382, 362]
LEFT_IRIS_IDX  = [468, 469, 470, 471]
RIGHT_IRIS_IDX = [473, 474, 475, 476]

# =========================
# 유틸/구조체
# =========================
class EMA:
    """단위벡터 EMA 평활화"""
    def __init__(self, alpha: float=0.25):
        self.alpha = float(alpha)
        self.v = None
    def update(self, new_v: np.ndarray):
        new_v = new_v.astype(np.float32)
        n = np.linalg.norm(new_v)
        if n > 1e-8: new_v /= n
        if self.v is None:
            self.v = new_v
        else:
            self.v = self.alpha * new_v + (1.0 - self.alpha) * self.v
            n2 = np.linalg.norm(self.v)
            if n2 > 1e-8: self.v /= n2
        return self.v

def apply_clahe_bgr(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_Lab2BGR)

def lm3d_px(lm, W, H):
    # MediaPipe 정규화 → 픽셀 스케일 3D. Z는 W 스케일(상대값).
    return np.array([lm.x * W, lm.y * H, lm.z * W], dtype=np.float32)

def lmidx_to_xyz(landmarks, idx_list, W, H):
    return np.array([lm3d_px(landmarks[i], W, H) for i in idx_list], dtype=np.float32)

def fit_plane_svd(pts3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c = pts3d.mean(axis=0)
    X = pts3d - c
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1, :]
    n = n / (np.linalg.norm(n) + 1e-8)
    return c, n

def orient_normal_to_camera(n: np.ndarray):
    # 카메라를 향해 -Z가 전방이 되도록 법선 방향 정렬
    cam_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    if np.dot(n, cam_dir) < 0: n = -n
    return n

def min_enclosing_circle_2d(pts2d):
    (cx, cy), r = cv2.minEnclosingCircle(pts2d.astype(np.float32))
    return float(cx), float(cy), float(r)

def project_point_K(P: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> Optional[Tuple[int,int]]:
    """카메라좌표계 3D점 → 픽셀 좌표(정수)"""
    Z = P[2]
    if Z <= 1e-6: return None
    u = int(round(fx * (P[0]/Z) + cx))
    v = int(round(fy * (P[1]/Z) + cy))
    return (u, v)

def depth_at(depth_m: np.ndarray, u: int, v: int, patch: int=5) -> float:
    """깊이맵 중앙값 샘플링(작은 패치, 잡음완화)"""
    H, W = depth_m.shape
    r0 = max(0, v - patch//2); r1 = min(H, v + patch//2 + 1)
    c0 = max(0, u - patch//2); c1 = min(W, u + patch//2 + 1)
    roi = depth_m[r0:r1, c0:c1]
    roi = roi[np.isfinite(roi)]
    if roi.size == 0: return float('nan')
    return float(np.median(roi))

def ray_depth_intersection(
    O: np.ndarray, d: np.ndarray,
    depth_m: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    t_min: float=0.1, t_max: float=8.0, step: float=0.01,
    patch: int=5, z_tol: float=0.05
):
    """
    샘플링으로 ray와 깊이맵 교차점을 탐색.
    반환: (P_hit, u, v, Z_depth, t_hit, residual)
    실패시: (None, None, None, nan, nan, inf)
    """
    H, W = depth_m.shape[:2]
    best = (None, None, None, float('nan'), float('nan'), float('inf'))
    prev_res = None

    t = t_min
    while t <= t_max:
        P = O + t * d
        uv = project_point_K(P, fx, fy, cx, cy)
        if uv is not None:
            u, v = uv
            if 0 <= u < W and 0 <= v < H:
                Z_d = depth_at(depth_m, u, v, patch=patch)
                if math.isfinite(Z_d):
                    res = abs(P[2] - Z_d)
                    if res < best[5]:
                        best = (P.copy(), u, v, Z_d, t, res)
                    if res < z_tol:
                        return (P, u, v, Z_d, t, res)
                    if prev_res is not None and res > prev_res and best[0] is not None:
                        return best
                    prev_res = res
        t += step

    return best

def load_extrinsic(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    R = np.array(meta["R"], dtype=np.float32).reshape(3,3)
    t = np.array(meta["t"], dtype=np.float32).reshape(3,1)
    return R, t

# ===================================================
# VDA 스트리밍(세그먼트 → run_streaming.py)
# ===================================================
class VDASegmentStreamer:
    """
    월드 프레임을 일정 길이 세그먼트(mp4/avi)로 저장하고,
    VDA run_streaming.py를 비동기 호출하여 출력 디렉터리에 결과(depth npz/exr 등)를 만든다.
    - npz 저장은 VDA 스크립트에 한 줄 추가하는 패치를 권장.
    """
    def __init__(self, vda_dir: str, encoder: str="vits", metric: bool=True,
                 input_size: int=392, max_res: int=960, fp32: bool=False,
                 segment_secs: int=1, target_fps: float=20.0,
                 tmp_dir: str="./_segments", out_root: str="./_vda_out"):
        self.vda_dir = vda_dir
        self.encoder = encoder
        self.metric = metric
        self.input_size = input_size
        self.max_res = max_res
        self.fp32 = fp32
        self.segment_secs = int(segment_secs)
        self.target_fps = float(target_fps)
        self.tmp_dir = tmp_dir
        self.out_root = out_root

        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.out_root, exist_ok=True)

        self.writer = None
        self.seg_path = None
        self.out_dir = None
        self.fps = None
        self.written = 0
        self.seg_frame_limit = None
        self._last_depth = None
        self._last_depth_time = 0.0
        self._scan_cooldown = 0.2

    def _build_writer(self, path, w, h, fps):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if not vw.isOpened():
            base, _ = os.path.splitext(path)
            path = base + ".avi"
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
        if not vw.isOpened():
            raise RuntimeError("VideoWriter open failed. Install codecs or try different params.")
        return vw, path

    def push(self, frame_bgr):
        """월드 프레임을 세그먼트 파일에 순차 저장"""
        H, W = frame_bgr.shape[:2]
        if self.writer is None:
            self.fps = 30.0 if self.target_fps <= 0 else float(self.target_fps)
            self.seg_frame_limit = int(round(self.fps * self.segment_secs))
            ts = time.strftime("%Y%m%d_%H%M%S")
            base = f"seg_{ts}"
            self.seg_path = os.path.join(self.tmp_dir, base + ".mp4")
            self.writer, self.seg_path = self._build_writer(self.seg_path, W, H, self.fps)
            self.out_dir = os.path.join(self.out_root, base)
            os.makedirs(self.out_dir, exist_ok=True)
            self.written = 0

        self.writer.write(frame_bgr)
        self.written += 1
        if self.written >= self.seg_frame_limit:
            self.writer.release()
            self.writer = None
            self._launch_vda(self.seg_path, self.out_dir)

    def _launch_vda(self, seg_path, out_dir):
        """run_streaming.py 비동기 호출"""
        py = sys.executable if sys.executable else "python"
        cmd = [
            py, os.path.join(self.vda_dir, "run_streaming.py"),
            "--input_video", seg_path,
            "--output_dir", out_dir,
            "--encoder", self.encoder
        ]
        if self.metric:
            cmd.append("--metric")
        if self.input_size:
            cmd += ["--input_size", str(self.input_size)]
        if self.max_res:
            cmd += ["--max_res", str(self.max_res)]
        if self.fp32:
            cmd.append("--fp32")

        threading.Thread(
            target=lambda: subprocess.run(cmd, cwd=self.vda_dir),
            daemon=True
        ).start()

    def _scan_latest_depth(self):
        """
        최근 세그먼트 출력에서 depth(npz 우선, 없으면 exr)를 로드시도.
        """
        now = time.time()
        if now - self._last_depth_time < self._scan_cooldown:
            return self._last_depth

        self._last_depth_time = now
        seg_outs = sorted(glob.glob(os.path.join(self.out_root, "seg_*")))
        if not seg_outs:
            return self._last_depth
        latest = seg_outs[-1]

        # npz 우선
        npzs = sorted(glob.glob(os.path.join(latest, "*.npz")))
        exrs = [] if npzs else sorted(glob.glob(os.path.join(latest, "*.exr")))
        if npzs:
            path = npzs[-1]
            try:
                z = np.load(path)
                for k in ("depth","arr_0"):
                    if k in z:
                        D = z[k].astype(np.float32)
                        self._last_depth = D
                        return D
            except Exception:
                pass
        elif exrs:
            path = exrs[-1]
            try:
                os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.ndim == 3: img = img[...,0]
                    self._last_depth = img.astype(np.float32)
                    return self._last_depth
            except Exception:
                pass
        return self._last_depth

    def get_latest_depth(self):
        return self._scan_latest_depth()

# =========================
# 시선 추정 (양눈) from Eye
# =========================
def draw_eye_polyline(frame, lms, idxs, color=(0,255,0)):
    H, W = frame.shape[:2]
    pts = [(int(lms[i].x*W), int(lms[i].y*H)) for i in idxs]
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)

def single_cam_dual_iris_step(frame_bgr, face_mesh, ema_L, ema_R,
                              iris_to_eyeball_ratio=2.1, flip=False, clahe=False):
    """
    입력 프레임에서 양쪽 홍채/시선 추정 (eye 카메라 좌표계의 '상대' 단위).
    반환 dict: oL,dL,cL2,rL2,oR,dR,cR2,rR2,pmid,denom
    """
    out = {k: None for k in ["oL","dL","cL2","rL2","oR","dR","cR2","rR2","pmid","denom"]}
    if flip: frame_bgr = cv2.flip(frame_bgr, 1)
    if clahe: frame_bgr = apply_clahe_bgr(frame_bgr)

    H, W = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return frame_bgr, out

    lms = res.multi_face_landmarks[0].landmark
    draw_eye_polyline(frame_bgr, lms, LEFT_EYE_IDX,  (0,255,0))
    draw_eye_polyline(frame_bgr, lms, RIGHT_EYE_IDX, (0,255,0))

    # 왼쪽
    irisL_3d = lmidx_to_xyz(lms, LEFT_IRIS_IDX, W, H)
    cL3, nL = fit_plane_svd(irisL_3d)
    nL = orient_normal_to_camera(nL)
    irisL_2d = np.array([[lms[i].x*W, lms[i].y*H] for i in LEFT_IRIS_IDX], dtype=np.float32)
    cxL, cyL, rL = min_enclosing_circle_2d(irisL_2d)
    R_e_L = iris_to_eyeball_ratio * rL
    oL = cL3 - nL * R_e_L
    dL = ema_L.update(nL)
    out.update(oL=oL, dL=dL, cL2=(int(cxL),int(cyL)), rL2=rL)
    cv2.circle(frame_bgr, out["cL2"], int(rL), (255,0,255), 2, lineType=cv2.LINE_AA)
    cv2.circle(frame_bgr, out["cL2"], 2, (255,255,255), -1, lineType=cv2.LINE_AA)
    p2 = (int(cxL + dL[0]*200), int(cyL + dL[1]*200))
    cv2.arrowedLine(frame_bgr, out["cL2"], p2, (0,255,255), 2, tipLength=0.18)

    # 오른쪽
    irisR_3d = lmidx_to_xyz(lms, RIGHT_IRIS_IDX, W, H)
    cR3, nR = fit_plane_svd(irisR_3d)
    nR = orient_normal_to_camera(nR)
    irisR_2d = np.array([[lms[i].x*W, lms[i].y*H] for i in RIGHT_IRIS_IDX], dtype=np.float32)
    cxR, cyR, rR = min_enclosing_circle_2d(irisR_2d)
    R_e_R = iris_to_eyeball_ratio * rR
    oR = cR3 - nR * R_e_R
    dR = ema_R.update(nR)
    out.update(oR=oR, dR=dR, cR2=(int(cxR),int(cyR)), rR2=rR)
    cv2.circle(frame_bgr, out["cR2"], int(rR), (255,0,255), 2, lineType=cv2.LINE_AA)
    cv2.circle(frame_bgr, out["cR2"], 2, (255,255,255), -1, lineType=cv2.LINE_AA)
    p2 = (int(cxR + dR[0]*200), int(cyR + dR[1]*200))
    cv2.arrowedLine(frame_bgr, out["cR2"], p2, (0,255,255), 2, tipLength=0.18)

    # 양눈 최근접 중점
    w0 = oL - oR
    a = float(np.dot(dL, dL)); b = float(np.dot(dL, dR)); c = float(np.dot(dR, dR))
    d = float(np.dot(dL, w0)); e = float(np.dot(dR, w0))
    denom = a*c - b*b
    if abs(denom) < 1e-6:
        t = 0.0
        s = e/c if abs(c) > 1e-9 else 0.0
    else:
        t = (b*e - c*d) / denom
        s = (a*e - b*d) / denom
    p1 = oL + t*dL
    p2 = oR + s*dR
    pmid = 0.5*(p1 + p2)
    out.update(pmid=pmid, denom=float(denom))

    cv2.putText(frame_bgr, f"P(mid): [{pmid[0]:.1f},{pmid[1]:.1f},{pmid[2]:.1f}] | |den|={abs(denom):.2e}",
                (18, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
    return frame_bgr, out

# =========================
# 입력 오픈 (PyAV 우선)
# =========================
def _open_pyav_or_fallback(
    src: str,
    *,
    av_backend: str,
    pixel_format: str,
    width: int,
    height: int,
    fps: float,
    backend_fallback: str,
    fourcc: str = None,
    exposure: float = None,
    autofocus: int = None,
    flip: bool = False
):
    """
    src가 'video=장치명' 또는 URL이면 PyAV 시도, 아니면 OpenCV 폴백.
    Returns: (read_fn, release_fn, flip_runtime_bool)
    """
    use_pyav = False
    s = str(src).strip()
    if s.lower().startswith("video=") or s.lower().startswith("rtsp://") or s.lower().startswith("http"):
        use_pyav = True
    if use_pyav and not _HAS_PYAV:
        print("[WARN] PyAV import 실패 → OpenCV 폴백으로 진행 (pip install av 필요)")
        use_pyav = False

    if use_pyav:
        grabber = PyAvLatestGrabber(
            device_name_or_url=s,
            backend=av_backend,
            width=width, height=height,
            fps=int(fps),
            pixel_format=pixel_format
        )
        def _read():
            return grabber.read(wait_latest=True, wait_ms=200)
        def _release():
            try: grabber.release()
            except Exception: pass
        return _read, _release, bool(flip)

    # --- OpenCV 폴백 ---
    cap, flip_cv = open_source(s, backend_fallback, width, height, fps, fourcc,
                               flip=flip, exposure=exposure, autofocus=autofocus)
    grabber = LatestFrameGrabber(cap)
    def _read():
        return grabber.read()
    def _release():
        try: grabber.release()
        except Exception: pass
    return _read, _release, bool(flip_cv)

# =========================
# 메인
# =========================
def main():
    ap = argparse.ArgumentParser(description="Eye×World (webcam/RTSP) + VDA streaming (low-latency)")

    # 입력원 (PyAV 우선; 숫자는 OpenCV 폴백)
    ap.add_argument("--eye_cam", type=str, required=True, help="'2' 또는 'video=Iriun Webcam' 또는 'rtsp://...'")
    ap.add_argument("--world_cam", type=str, required=True, help="'1' 또는 'video=USB2.0 Camera' 또는 'rtsp://...'")

    # PyAV 전용 지정(eye/world 분리)
    ap.add_argument("--av_backend_eye", type=str, default="dshow")
    ap.add_argument("--av_backend_world", type=str, default="dshow")
    ap.add_argument("--pixel_format_eye", type=str, default="mjpeg")
    ap.add_argument("--pixel_format_world", type=str, default="mjpeg")

    # 공통 해상도/프레임
    ap.add_argument("--width_eye", type=int, default=640)
    ap.add_argument("--height_eye", type=int, default=480)
    ap.add_argument("--fps_eye", type=float, default=30)
    ap.add_argument("--width_world", type=int, default=512)
    ap.add_argument("--height_world", type=int, default=384)
    ap.add_argument("--fps_world", type=float, default=30)

    # OpenCV 폴백에서만 쓰임
    ap.add_argument("--backend", type=str, default="dshow", choices=list(BACKENDS.keys()))
    ap.add_argument("--fourcc_eye", type=str, default=None)
    ap.add_argument("--fourcc_world", type=str, default=None)
    ap.add_argument("--exposure_eye", type=float, default=None)
    ap.add_argument("--exposure_world", type=float, default=None)
    ap.add_argument("--autofocus_eye", type=int, default=None)
    ap.add_argument("--autofocus_world", type=int, default=None)

    ap.add_argument("--flip_eye", action="store_true")
    ap.add_argument("--flip_world", action="store_true")

    # VDA
    ap.add_argument("--vda_dir", type=str, required=True)
    ap.add_argument("--encoder", type=str, default="vits", choices=["vits","vitb","vitl"])
    ap.add_argument("--metric", action="store_true")
    ap.add_argument("--input_size", type=int, default=392)
    ap.add_argument("--max_res", type=int, default=960)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--segment_secs", type=int, default=1)
    ap.add_argument("--target_fps", type=float, default=20)

    # 월드 intrinsics (px)
    ap.add_argument("--fx_w", type=float, required=True)
    ap.add_argument("--fy_w", type=float, required=True)
    ap.add_argument("--cx_w", type=float, required=True)
    ap.add_argument("--cy_w", type=float, required=True)

    # 외부파라미터: eye → worldCam (meters)
    ap.add_argument("--extrinsic_json", type=str, required=True)

    # 기타
    ap.add_argument("--iris_to_eyeball_ratio", type=float, default=2.1)
    ap.add_argument("--ema", type=float, default=0.25)
    ap.add_argument("--show_fps", action="store_true")
    ap.add_argument("--clahe_eye", action="store_true")
    ap.add_argument("--z_tol", type=float, default=0.05, help="교차 허용오차(미터)")
    ap.add_argument("--t_max", type=float, default=8.0, help="교차 탐색 최대거리(미터)")
    ap.add_argument("--step", type=float, default=0.01, help="교차 탐색 간격(미터)")

    args = ap.parse_args()

    # --- 입력 오픈: PyAV 우선 / OpenCV 폴백 ---
    read_eye, rel_eye, flip_eye_rt = _open_pyav_or_fallback(
        args.eye_cam,
        av_backend=args.av_backend_eye, pixel_format=args.pixel_format_eye,
        width=args.width_eye, height=args.height_eye, fps=args.fps_eye,
        backend_fallback=args.backend, fourcc=args.fourcc_eye,
        exposure=args.exposure_eye, autofocus=args.autofocus_eye,
        flip=args.flip_eye
    )
    read_world, rel_world, flip_world_rt = _open_pyav_or_fallback(
        args.world_cam,
        av_backend=args.av_backend_world, pixel_format=args.pixel_format_world,
        width=args.width_world, height=args.height_world, fps=args.fps_world,
        backend_fallback=args.backend, fourcc=args.fourcc_world,
        exposure=args.exposure_world, autofocus=args.autofocus_world,
        flip=args.flip_world
    )

    # FaceMesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    ema_L = EMA(args.ema); ema_R = EMA(args.ema)

    # Extrinsic (eye→worldCam)
    R_we, t_we = load_extrinsic(args.extrinsic_json)  # R(3x3), t(3,1)

    # VDA 스트리머
    streamer = VDASegmentStreamer(
        vda_dir=args.vda_dir, encoder=args.encoder, metric=args.metric,
        input_size=args.input_size, max_res=args.max_res, fp32=args.fp32,
        segment_secs=args.segment_secs, target_fps=args.target_fps,
        tmp_dir="./_segments", out_root="./_vda_out"
    )

    # HUD
    fps_eye = 0.0; fps_w = 0.0
    fx_w, fy_w, cx_w, cy_w = args.fx_w, args.fy_w, args.cx_w, args.cy_w
    t_last = time.time(); eye_frames = 0; world_frames = 0

    try:
        while True:
            # ---- 최신 프레임 읽기 ----
            okE, frame_eye = read_eye()
            okW, frame_world = read_world()
            if not (okE and okW):
                cv2.waitKey(1); continue
            if flip_eye_rt:   frame_eye = cv2.flip(frame_eye, 1)
            if flip_world_rt: frame_world = cv2.flip(frame_world, 1)

            # ---- Eye: 시선 추정 ----
            t0 = time.time()
            frame_eye_vis, gaze = single_cam_dual_iris_step(
                frame_eye, face_mesh, ema_L, ema_R,
                iris_to_eyeball_ratio=args.iris_to_eyeball_ratio,
                flip=False, clahe=args.clahe_eye
            )
            t1 = time.time()
            fps_eye = 0.9*fps_eye + 0.1*(1.0/max(t1-t0,1e-3))
            eye_frames += 1

            # cyclopean origin & dir (eye 좌표, 상대단위)
            if gaze["oL"] is not None and gaze["oR"] is not None:
                O_eye = 0.5*(gaze["oL"] + gaze["oR"])
                if gaze["pmid"] is not None:
                    d_eye = gaze["pmid"] - O_eye
                else:
                    d_eye = 0.5*(gaze["dL"] + gaze["dR"])
                # 비등방 보정
                H_e, W_e = frame_eye.shape[:2]
                s = np.array([1.0, float(W_e)/float(H_e), 1.0], dtype=np.float32)
                d_eye = d_eye * s
                n = np.linalg.norm(d_eye)
                if n > 1e-8: d_eye = d_eye / n
            else:
                d_eye = None

            # ---- World: VDA 세그먼트 / 최신 깊이 ----
            streamer.push(frame_world)              # 세그먼트에 world 프레임 공급
            depth_m = streamer.get_latest_depth()   # 최신 깊이맵 가져오기

            # eye→worldCam 변환 + 교차 탐색
            hit_txt = "no depth yet"
            if d_eye is not None and depth_m is not None:
                O_w = t_we.reshape(3)   # meters
                d_w = (R_we @ d_eye.reshape(3,1)).reshape(3)
                n = np.linalg.norm(d_w)
                if n > 1e-8: d_w = d_w / n

                P_hit, u, v, Z_d, t_hit, res = ray_depth_intersection(
                    O_w, d_w, depth_m, fx_w, fy_w, cx_w, cy_w,
                    t_min=0.1, t_max=args.t_max, step=args.step,
                    patch=7, z_tol=args.z_tol
                )

                if P_hit is not None and u is not None:
                    # 월드 영상에 응시점 표시
                    cv2.drawMarker(frame_world, (u, v), (0,0,255),
                                   markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
                    Rm = float(np.linalg.norm(P_hit - O_w))
                    hit_txt = f"fix: u={u}, v={v} | Z={Z_d:.2f}m | R={Rm:.2f}m | res={res:.03f}m"
                else:
                    hit_txt = "fix: not found (t_max/step/z_tol 조정)"

            # ---- HUD & 디스플레이 ----
            now = time.time()
            if now - t_last >= 1.0:
                fps_w = world_frames/(now - t_last) if (now-t_last)>0 else 0.0
                world_frames = 0; eye_frames = 0; t_last = now

            if args.show_fps:
                cv2.putText(frame_eye_vis, f"Eye FPS: {fps_eye:4.1f}", (14, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Eye (FaceMesh + Gaze)", frame_eye_vis)

            hud = f"VDA[{args.encoder}{'/metric' if args.metric else ''}] size={args.input_size},max={args.max_res}"
            cv2.putText(frame_world, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame_world, hit_txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow("World (VDA-stream metric + Fixation)", frame_world)
            world_frames += 1

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break

    finally:
        try: rel_eye()
        except Exception: pass
        try: rel_world()
        except Exception: pass
        face_mesh.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
