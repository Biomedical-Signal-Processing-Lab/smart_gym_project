import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

REQUIRED_ELEMS = [
    "v4l2src","videoconvert","videoscale","tee","appsink",
    "hailocropper","hailoaggregator","hailonet","hailofilter","hailooverlay","autovideosink"
]

def build_pipeline(s) -> str:
    # framerate 캡스 모두 제거 (복제 방지)
    return f"""
v4l2src device={s.CAM} io-mode=2 do-timestamp=true !
video/x-raw,format=YUY2,width=640,height=480 !
videoconvert ! videoscale !
video/x-raw,format=RGB,width={s.SRC_WIDTH},height={s.SRC_HEIGHT},pixel-aspect-ratio=1/1 !
queue name=inference_wrapper_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 !
hailocropper name=crop so-path={s.CROPPER_SO} function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true
hailoaggregator name=agg
crop. ! queue name=bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0 ! agg.sink_0
crop. ! queue name=inf_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 !
videoscale n-threads=2 qos=false ! videoconvert n-threads=2 !
video/x-raw,pixel-aspect-ratio=1/1 !
queue name=inf_hnet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 !
hailonet name=hnet hef-path={s.HEF} batch-size=2 vdevice-group-id=1 force-writable=true !
queue name=inf_post_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 !
hailofilter name=post so-path={s.POST_SO} function-name={s.POST_FUNC} qos=false !
queue name=inf_out_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! agg.sink_1

agg. ! tee name=t


t. ! queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 !
hailooverlay name=ovl !
videoconvert n-threads=2 qos=false !
autovideosink name=preview sync=false

t. ! queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 !
videoconvert n-threads=2 qos=false !
video/x-raw,format=BGR,width={s.SRC_WIDTH},height={s.SRC_HEIGHT} !
appsink name=data_sink emit-signals=true sync=false max-buffers=1 drop=true
"""

def _check_elements():
    missing = [n for n in REQUIRED_ELEMS if Gst.ElementFactory.find(n) is None]
    if missing:
        raise RuntimeError(f"Missing GStreamer elements: {missing}")

def create_pipeline_and_sink(s):
    Gst.init(None)
    _check_elements()
    desc = build_pipeline(s)
    print("=== PIPE ==="); print(desc); print("==============")
    try:
        pipe = Gst.parse_launch(desc)
    except GLib.Error as e:
        raise RuntimeError(f"gst_parse_error: {e.message}")
    data_sink = pipe.get_by_name("data_sink")
    if not data_sink:
        raise RuntimeError("appsink not found: data_sink")
    return pipe, data_sink
