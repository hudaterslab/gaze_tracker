import os
import cv2
import numpy as np
import json
import argparse
import threading
import queue
import time
import csv
from datetime import datetime
from typing import List, Dict
from collections import deque  # 데이터 스택용 (Smoothing)

# Custom Engine Import
from dx_engine import InferenceEngine, InferenceOption

# Torch/Ultralytics (Used for NMS utilities)
import torch
import torchvision
from ultralytics.utils import ops

# ===============================
# 1. Configuration & Constants
# ===============================
BASE_COLORS = [
    (255,0,0),(0,255,0),(0,0,255),(255,255,0),
    (255,0,255),(0,255,255),(128,0,128),(255,165,0),
    (0,128,128),(128,128,0)
]

def get_color_for_class(class_id:int, num_classes:int):
    base = BASE_COLORS[class_id % len(BASE_COLORS)]
    shift = (class_id // len(BASE_COLORS)) * 30
    return ((base[0]+shift)%256, (base[1]+shift)%256, (base[2]+shift)%256)

class YoloConfig:
    def __init__(self, model_path, classes, score_threshold, iou_threshold, input_size):
        self.model_path = model_path
        self.classes = classes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.input_size = (input_size, input_size)
        self.colors = [get_color_for_class(i,len(classes)) for i in range(len(classes))]

class YoloPoseConfig:
    def __init__(self, model_path, classes, score_threshold, iou_threshold, input_size, num_keypoints):
        self.model_path = model_path
        self.classes = classes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.input_size = (input_size, input_size)
        self.num_keypoints = num_keypoints
        self.colors = [get_color_for_class(i,len(classes)) for i in range(len(classes))]

# ===============================
# 2. GazeEstimator (With Singularity Fix)
# ===============================
class GazeEstimator:
    def __init__(self, img_w, img_h):
        # 3D Model Points
        self.face_3d = np.array([
            [0.0, 0.0, 0.0],          # Nose
            [-35.0, -25.0, -20.0],    # L Eye 
            [35.0, -25.0, -20.0],     # R Eye
            [-70.0, -10.0, -60.0],    # L Ear
            [70.0, -10.0, -60.0]      # R Ear
        ], dtype=np.float64)

        self.cam_matrix = np.array([
            [img_w, 0, img_w / 2],
            [0, img_w, img_h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1))

    def estimate(self, keypoints):
        if keypoints.shape[0] < 5: return None, None, (0,0)
        if np.min(keypoints[:5, 2]) < 0.4: return None, None, (0,0)

        img_pts = np.ascontiguousarray(keypoints[:5, :2], dtype=np.float64)
        
        success, rvec, tvec = cv2.solvePnP(self.face_3d, img_pts, self.cam_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        if not success: return None, None, (0,0)

        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        raw_pitch = angles[0]
        raw_yaw = angles[1]

        # 180도 경계값 보정 (Singularity Fix)
        pitch = raw_pitch
        if pitch < 0:
            pitch += 360
            
        # --- [영점 조절] ---
        # 사용자 환경에 맞춰 미세조정 필요
        target_pitch = 180.0 
        target_yaw = 9.0     
        
        # 민감도 (Smoothing 적용 시 조금 높여도 됨)
        x_gain = 2.0
        y_gain = 2.0
        
        # ==========================================================
        # [FIX] Reversing Left/Right
        # Changed '-' to '+' in the gx calculation below.
        # ==========================================================
        gx = 0.5 + ((raw_yaw - target_yaw) * x_gain / 90.0)
        
        gy = 0.5 - ((pitch - target_pitch) * y_gain / 90.0) 
        
        return gx, gy, (raw_pitch, raw_yaw)

# ===============================
# 3. Analytics Logic (Smoothing Added)
# ===============================
class PersonTrack:
    def __init__(self, track_id, keypoints, bbox):
        self.id = track_id
        self.keypoints = keypoints
        self.bbox = bbox
        self.last_seen = time.time()
        
        self.focus_start_time = None
        self.total_focus_duration = 0.0
        self.is_interested = False
        self.best_snapshot = None
        
        self.gaze_point = (0.5, 0.5) 
        self.is_facing_screen = False

        # --- [Smoothing History] ---
        # 최근 7프레임의 좌표를 저장하여 중앙값 계산
        self.gaze_history = deque(maxlen=7) 

class FocusAnalytics:
    def __init__(self, monitor_w, monitor_h):
        self.tracks: Dict[int, PersonTrack] = {}
        self.next_id = 1
        self.snapshots = []
        self.W = monitor_w
        self.H = monitor_h
        self.gaze_estimator = GazeEstimator(monitor_w, monitor_h)

    def update(self, current_poses, current_bboxes, frame_img):
        now = time.time()
        
        # 1. Prepare Objects
        current_objects = []
        for i, kpts in enumerate(current_poses):
            if kpts[0][2] > 0.0: # Valid nose
                bbox = current_bboxes[i] if i < len(current_bboxes) else [0,0,0,0,0,0]
                current_objects.append({'kpts': kpts, 'bbox': bbox, 'center': (kpts[0][0], kpts[0][1])})

        # 2. Match Tracks
        MAX_DIST = 150
        for obj in current_objects:
            matched_id = None
            min_dist = MAX_DIST
            for tid, track in self.tracks.items():
                if track.keypoints[0][2] > 0:
                    dist = np.linalg.norm(np.array(obj['center']) - np.array(track.keypoints[0][:2]))
                    if dist < min_dist:
                        min_dist = dist
                        matched_id = tid
            
            if matched_id is not None:
                self.tracks[matched_id].keypoints = obj['kpts']
                self.tracks[matched_id].bbox = obj['bbox']
                self.tracks[matched_id].last_seen = now
                self._process_focus(self.tracks[matched_id], frame_img)
            else:
                new_track = PersonTrack(self.next_id, obj['kpts'], obj['bbox'])
                self.tracks[self.next_id] = new_track
                self._process_focus(new_track, frame_img)
                self.next_id += 1

        # 3. Remove Old Tracks
        to_remove = []
        for tid, track in self.tracks.items():
            if now - track.last_seen > 1.0:
                self._finalize_track(track)
                to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

    def _process_focus(self, track: PersonTrack, frame_img):
        # 1. Raw 좌표 계산
        gx_raw, gy_raw, _ = self.gaze_estimator.estimate(track.keypoints)
        
        if gx_raw is None:
            track.is_facing_screen = False
            track.focus_start_time = None
            return

        # 2. History에 추가 (Stacking)
        track.gaze_history.append((gx_raw, gy_raw))

        # 3. Median Filter 적용 (Smoothing)
        history_arr = np.array(track.gaze_history)
        gx = np.median(history_arr[:, 0])
        gy = np.median(history_arr[:, 1])

        track.gaze_point = (gx, gy)

        # 4. 화면 범위 체크
        on_screen = (-0.2 <= gx <= 1.2) and (-0.2 <= gy <= 1.2)
        
        if on_screen:
            track.is_facing_screen = True
            if track.focus_start_time is None:
                track.focus_start_time = time.time()
            
            duration = time.time() - track.focus_start_time
            
            if duration > 3.0:
                if not track.is_interested: 
                    track.is_interested = True
                
                if duration > track.total_focus_duration:
                    track.total_focus_duration = duration
                    if int(duration * 10) % 10 == 0:
                        track.best_snapshot = self._create_blur_snapshot(frame_img, track)
        else:
            track.is_facing_screen = False
            track.focus_start_time = None

    def _create_blur_snapshot(self, frame, track):
        h, w = frame.shape[:2]
        xs, ys = track.keypoints[:, 0], track.keypoints[:, 1]
        valid = track.keypoints[:, 2] > 0.3
        if not np.any(valid): return None
        x1, y1 = max(0, int(np.min(xs[valid]))-30), max(0, int(np.min(ys[valid]))-30)
        x2, y2 = min(w, int(np.max(xs[valid]))+30), min(h, int(np.max(ys[valid]))+30)
        if x2<=x1 or y2<=y1: return None
        
        roi = frame[y1:y2, x1:x2].copy()
        
        nx, ny = track.keypoints[0][:2]
        fx, fy = int(nx - x1), int(ny - y1)
        bx1, by1 = max(0, fx-40), max(0, fy-40)
        bx2, by2 = min(roi.shape[1], fx+40), min(roi.shape[0], fy+40)
        
        try:
            face_part = roi[by1:by2, bx1:bx2]
            if face_part.size > 0:
                roi[by1:by2, bx1:bx2] = cv2.GaussianBlur(face_part, (99, 99), 30)
        except: pass
        return roi

    def _finalize_track(self, track):
        if track.is_interested and track.best_snapshot is not None:
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            fname = f"gaze_log_{ts[:10]}.csv"
            try:
                with open(fname, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([ts, track.id, f"{track.total_focus_duration:.2f}"])
                self.snapshots.insert(0, track.best_snapshot)
                if len(self.snapshots) > 6: self.snapshots.pop()
            except Exception as e:
                print(f"Save Error: {e}")

# ===============================
# 4. Helper Functions (Preprocessing)
# ===============================
def letter_box(image_src, new_shape=(512,512), fill_color=(114,114,114)):
    src_h, src_w = image_src.shape[:2]
    r = min(new_shape[0]/src_h, new_shape[1]/src_w)
    new_unpad = int(round(src_w * r)), int(round(src_h * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if (src_w, src_h) != new_unpad[::-1]: image_src = cv2.resize(image_src, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    return cv2.copyMakeBorder(image_src, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color), r, (dw, dh), (r,r), (dw,dh)

def transform_box(pt1, pt2, ratio, offset, original_shape):
    dw, dh = offset
    pt1[0] = (pt1[0] - dw) / ratio[0]; pt1[1] = (pt1[1] - dh) / ratio[1]
    pt2[0] = (pt2[0] - dw) / ratio[0]; pt2[1] = (pt2[1] - dh) / ratio[1]
    pt1[0] = int(max(0, min(pt1[0], original_shape[1])))
    pt1[1] = int(max(0, min(pt1[1], original_shape[0])))
    pt2[0] = int(max(0, min(pt2[0], original_shape[1])))
    pt2[1] = int(max(0, min(pt2[1], original_shape[0])))
    return pt1, pt2

def postprocess_detection(output, cfg):
    x = np.squeeze(output[0].transpose(0,2,1))
    box, conf = ops.xywh2xyxy(x[...,:4]), np.max(x[...,4:], axis=-1)
    mask = conf > cfg.score_threshold
    if not np.any(mask): return torch.empty((0,6))
    sel = np.concatenate((box[mask], conf[mask, None], np.argmax(x[...,4:], axis=-1)[mask, None]), axis=1)
    sel = sel[np.argsort(-sel[:,4])] 
    keep = torchvision.ops.nms(torch.from_numpy(sel[:,:4]), torch.from_numpy(sel[:,4]), cfg.iou_threshold)
    
    # FIX: Ensure numpy indices and 2D shape
    keep_idxs = keep.numpy()
    result = sel[keep_idxs]
    if result.ndim == 1: result = np.expand_dims(result, axis=0)
    return torch.from_numpy(result)

def postprocess_pose(output, cfg, scale, pad, shape):
    dets = output[0].squeeze(0)
    dets = dets[dets[:,4] > cfg.score_threshold]
    if len(dets)==0: return np.array([]), [], []
    
    # NMS
    boxes_xywh = dets[:, :4]
    boxes_xyxy = boxes_xywh.copy()
    boxes_xyxy[:, 0] -= boxes_xywh[:, 2]/2
    boxes_xyxy[:, 1] -= boxes_xywh[:, 3]/2
    indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), dets[:,4].tolist(), cfg.score_threshold, cfg.iou_threshold)
    if len(indices) == 0: return np.array([]), [], []
    dets = dets[indices.flatten()]
    
    # FIX: Use index 6 for keypoints
    kpts = dets[:, 6:].reshape(-1, cfg.num_keypoints, 3)
    kpts[:,:,0] = (kpts[:,:,0]-pad[0])/scale
    kpts[:,:,1] = (kpts[:,:,1]-pad[1])/scale
    kpts[:,:,0] = np.clip(kpts[:,:,0], 0, shape[1]-1)
    kpts[:,:,1] = np.clip(kpts[:,:,1], 0, shape[0]-1)
    return kpts, dets[:,4], []

# ===============================
# 5. Thread Workers
# ===============================
def frame_grabber(input_source, frame_queue, stop_event):
    # RTSP often needs CAP_FFMPEG
    cap = cv2.VideoCapture(input_source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        # Fallback
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            print(f"Error: Could not open source {input_source}")
            stop_event.set()
            return

    print("Camera connected.")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Stream error. Reconnecting...")
            time.sleep(2)
            cap.release()
            cap = cv2.VideoCapture(input_source, cv2.CAP_FFMPEG)
            continue
            
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except: pass
        frame_queue.put((frame, time.perf_counter()))
    cap.release()

def ai_worker(frame_queue, result_queue, ie_det, cfg_det, ie_pose, cfg_pose, stop_event):
    while not stop_event.is_set():
        try:
            frame, ts = frame_queue.get(timeout=0.1)
            if (time.perf_counter() - ts) > 0.2: continue 
        except queue.Empty: continue

        # Detection
        inp_d, _, _, r_d, o_d = letter_box(frame, cfg_det.input_size)
        out_d = ie_det.run([cv2.cvtColor(inp_d, cv2.COLOR_BGR2RGB)])
        bboxes = postprocess_detection(out_d, cfg_det).numpy()

        # Pose
        inp_p, s_p, p_p, _, _ = letter_box(frame, cfg_pose.input_size)
        out_p = ie_pose.run([inp_p])
        kpts, _, _ = postprocess_pose(out_p, cfg_pose, s_p, p_p, frame.shape)

        if result_queue.full():
            try: result_queue.get_nowait()
            except: pass
        result_queue.put((frame, ts, bboxes, r_d, o_d, kpts))

# ===============================
# 6. Visualizer
# ===============================
def visualizer(result_queue, stop_event):
    analytics = None
    print("=== VISUALIZER STARTED (SMOOTHED) ===")
    
    while not stop_event.is_set():
        try:
            data = result_queue.get(timeout=0.1)
            while not result_queue.empty(): data = result_queue.get(timeout=0.1)
            frame, ts, bboxes, r_d, o_d, kpts = data
        except queue.Empty: continue

        if analytics is None: analytics = FocusAnalytics(frame.shape[1], frame.shape[0])
        
        real_bboxes = []
        if bboxes.ndim == 1 and bboxes.size > 0: bboxes = np.expand_dims(bboxes, axis=0)
        for r in bboxes:
            if len(r) < 6: continue
            p1, p2 = transform_box(r[:2].astype(int), r[2:4].astype(int), r_d, o_d, frame.shape)
            real_bboxes.append([*p1, *p2, r[4], r[5]])

        analytics.update(kpts, real_bboxes, frame)
        
        H, W = frame.shape[:2]
        vis = np.zeros((H+160, W, 3), dtype=np.uint8)
        vis[:H] = frame
        
        cv2.rectangle(vis, (0,0), (W, 60), (0,0,0), -1)
        cv2.putText(vis, "GAZE TRACKING (Median Smoothed)", (20, 40), 0, 0.8, (0,255,255), 2)

        for tid, track in analytics.tracks.items():
            gx, gy = track.gaze_point
            
            screen_x = int(np.clip(gx, 0, 1) * W)
            screen_y = int(np.clip(gy, 0, 1) * H)
            
            nose = track.keypoints[0]
            nx, ny = int(nose[0]), int(nose[1])

            color = (0,255,255) if track.is_facing_screen else (0,0,255)
            cv2.line(vis, (nx, ny), (screen_x, screen_y), color, 2)
            cv2.circle(vis, (screen_x, screen_y), 8, color, -1)
            
            if track.is_interested:
                cv2.putText(vis, "INTERESTED!", (nx, ny-60), 0, 0.7, (0,255,0), 2)
            
            for p in track.keypoints[:5]:
                cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0,255,0), -1)

        sx = 20
        cv2.putText(vis, "CAPTURED:", (10, H+140), 0, 0.5, (200,200,200), 1)
        for snap in analytics.snapshots:
            try:
                sh, sw = snap.shape[:2]
                scale = 120/sh
                ns = cv2.resize(snap, (int(sw*scale), 120))
                vis[H+10:H+130, sx:sx+ns.shape[1]] = ns
                sx += ns.shape[1] + 10
            except: pass

        cv2.imshow("GazeTracker", vis)
        if cv2.waitKey(1) == ord('q'): stop_event.set()
    cv2.destroyAllWindows()

# ===============================
# 7. Main
# ===============================
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--det-model', default='yolov8l-det.dxnn')
    parser.add_argument('--det-config', default='helmet_det_config.json')
    parser.add_argument('--pose-model', default='YOLOV5Pose640_1.dxnn')
    parser.add_argument('--pose-config', default='yolopose_config.json')
    parser.add_argument('--input', default='rtsp://admin:Hu924688@192.168.10.64:554/Streaming/Channels/101')
    args = parser.parse_args()

    print("Loading Models...")
    try:
        with open(args.det_config) as f: det_cfg = json.load(f)
        with open(args.pose_config) as f: pose_cfg = json.load(f)
    except Exception as e:
        print(f"Config Error: {e}")
        exit()

    io = InferenceOption()
    io.set_use_ort(True)
    
    ie_det = InferenceEngine(args.det_model, io)
    ie_pose = InferenceEngine(args.pose_model, io)

    y_det = YoloConfig(args.det_model, det_cfg["output"]["classes"], det_cfg["model"]["param"]["score_threshold"], 0.45, det_cfg["model"]["param"]["input_size"])
    y_pose = YoloPoseConfig(args.pose_model, pose_cfg["output"]["classes"], pose_cfg["model"]["param"]["score_threshold"], 0.45, pose_cfg["model"]["param"]["input_width"], pose_cfg["model"]["param"]["kpt_count"])

    frame_q = queue.Queue(maxsize=30)
    res_q = queue.Queue(maxsize=30)
    stop_ev = threading.Event()

    # 스레드 생성 및 시작
    t1 = threading.Thread(target=frame_grabber, args=(args.input, frame_q, stop_ev))
    t2 = threading.Thread(target=ai_worker, args=(frame_q, res_q, ie_det, y_det, ie_pose, y_pose, stop_ev))
    t3 = threading.Thread(target=visualizer, args=(res_q, stop_ev))

    print(f"Start Gaze Tracking on {args.input}...")
    t1.start(); t2.start(); t3.start()

    try:
        while not stop_ev.is_set(): time.sleep(1)
    except KeyboardInterrupt:
        stop_ev.set()

    t1.join(); t2.join(); t3.join()
    print("Closed.")
