# pipeline.py
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
#,framerate=30/1 
def build_pipeline(s) -> str:
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
agg. ! queue name=ovl_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 !
hailooverlay name=ovl !
videoconvert n-threads=2 qos=false !
video/x-raw,format=BGR,width={s.SRC_WIDTH},height={s.SRC_HEIGHT} !
appsink name=out_sink emit-signals=true sync=false max-buffers=1 drop=true
"""

def create_pipeline_and_sink(s):
    Gst.init(None)
    pipe = Gst.parse_launch(build_pipeline(s))
    sink = pipe.get_by_name("out_sink")
    if sink is None:
        raise RuntimeError("appsink not found")
    return pipe, sink
