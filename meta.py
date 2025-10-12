# meta.py
import cv2

def meta_counts_string(buf, w: int, h: int, fps: float) -> str:
    try:
        import hailo
        roi = hailo.get_roi_from_buffer(buf)
        if roi is None:
            return f"FPS {fps:.1f} | {w}x{h} | rois=0"
        dets = roi.get_objects_typed(hailo.HAILO_DETECTION)
        lm_count = 0
        for d in dets:
            try:
                lm = d.get_objects_typed(hailo.HAILO_LANDMARKS)
                if lm: lm_count += 1
            except Exception:
                pass
        return f"FPS {fps:.1f} | {w}x{h} | rois=1 objs={len(dets)} landmarks={lm_count}"
    except Exception as e:
        return f"FPS {fps:.1f} | {w}x{h} | [meta err: {e}]"

def draw_kpt_indices(frame, buf, w: int, h: int):
    try:
        import hailo
        roi = hailo.get_roi_from_buffer(buf)
        if roi is None:
            return
        dets = roi.get_objects_typed(hailo.HAILO_DETECTION)
        for d in dets:
            # bbox 정규 좌표(0..1) → 픽셀
            try:
                bbox = d.get_bbox()
                xmin = float(bbox.xmin() if callable(getattr(bbox,"xmin",None)) else bbox.xmin)
                ymin = float(bbox.ymin() if callable(getattr(bbox,"ymin",None)) else bbox.ymin)
                bw   = float(bbox.width() if callable(getattr(bbox,"width",None)) else bbox.width)
                bh   = float(bbox.height() if callable(getattr(bbox,"height",None)) else bbox.height)
            except Exception:
                continue
            try:
                lms = d.get_objects_typed(hailo.HAILO_LANDMARKS)
            except Exception:
                lms = []
            if not lms:
                continue
            try:
                points = lms[0].get_points()
            except Exception:
                continue
            for i, p in enumerate(points):
                try:
                    px = int((p.x() * bw + xmin) * w)
                    py = int((p.y() * bh + ymin) * h)
                    cv2.putText(frame, str(i), (px+3, py-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
                except Exception:
                    pass
    except Exception:
        pass
