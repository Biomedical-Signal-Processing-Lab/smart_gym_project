# display_pipe.py
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

def create_display_pipeline(width: int, height: int):
    """
    파이썬이 그린 BGR 프레임을 appsrc로 밀어 autovideosink로 표시.
    """
    Gst.init(None)
    desc = """
appsrc name=disp_src is-live=true block=true format=time !
videoconvert !
autovideosink name=preview sync=false
"""
    pipe = Gst.parse_launch(desc)
    appsrc = pipe.get_by_name("disp_src")
    # caps는 BGR로 고정 (우리가 그려서 넣는 프레임 포맷)
    caps = Gst.Caps.from_string(f"video/x-raw,format=BGR,width={width},height={height}")
    appsrc.set_property("caps", caps)
    return pipe, appsrc
