"""Compose the recorded demo into one 1920x1080 60 fps H.264 video.

View area (left 1608 px): the JPEG frames the server streamed, each shown from its arrival time, so the video
plays exactly the frame rate the browser got. Control panel (right 312 px): the browser screencast (~20 fps).
On top: caption and a meter (fps over the last second, link rate, server frame time) from the frame log.

usage: python scripts/compose_demo.py <frames dir> <ffmpeg with libx264> <out.mp4>
"""
import bisect, json, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

fdir, ffmpeg, out = sys.argv[1:4]
L = json.load(open(f"{fdir}/log.json"))
W, H, VW, FPS = 1920, 1080, 1608, 60
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
f_cap, f_sub = ImageFont.truetype(F + "-Bold.ttf", 30), ImageFont.truetype(F + ".ttf", 22)
f_fps, f_m = ImageFont.truetype(F + "-Bold.ttf", 22), ImageFont.truetype(F + ".ttf", 15)

screen_t = [s[0] for s in L["screen"]]
ws = L["ws"]
ws_t = [w[0] + 0.008 for w in ws]                       # arrival + decode and draw
caps = L["caps"]
t0, t1 = screen_t[0], screen_t[-1]


class Loader:                                           # decode each JPEG once
    def __init__(self, pattern):
        self.pattern, self.i, self.img = pattern, -1, None

    def get(self, i):
        if i != self.i:
            self.i, self.img = i, Image.open(self.pattern % i).convert("RGB")
        return self.img


scr, view = Loader(f"{fdir}/screen/%06d.jpg"), Loader(f"{fdir}/ws/%06d.jpg")


def caption(d, t):
    k = bisect.bisect_right([c[0] for c in caps], t) - 1
    if k < 0 or not caps[k][1]:
        return
    a = min(1, (t - caps[k][0]) / 0.3)                  # fade in
    if k + 1 < len(caps) and not caps[k + 1][1]:
        a = min(a, max(0, (caps[k + 1][0] - t) / 0.3))  # fade out before an empty caption
    lines = caps[k][1].split("\n")
    fonts = [f_cap] + [f_sub] * (len(lines) - 1)
    sizes = [d.textbbox((0, 0), s, font=f) for s, f in zip(lines, fonts)]
    tw = max(b[2] - b[0] for b in sizes)
    th = sum(b[3] - b[1] for b in sizes) + 10 * (len(lines) - 1)
    x0, y1 = VW // 2 - tw // 2 - 28, H - 70
    y0 = y1 - th - 28
    d.rounded_rectangle((x0, y0, VW - x0, y1), 14, fill=(0, 0, 0, int(190 * a)), outline=(230, 140, 40, int(170 * a)), width=2)
    y = y0 + 14
    for s, f, b in zip(lines, fonts, sizes):
        w = b[2] - b[0]
        d.text((VW // 2 - w // 2, y - b[1]), s, font=f, fill=(255, 255, 255, int(255 * a)) if f is f_cap else (215, 215, 215, int(255 * a)))
        y += b[3] - b[1] + 10


def meter(d, t, i):
    lo = bisect.bisect_left(ws_t, t - 1.0)
    n = i - lo + 1
    fps = (n - 1) / (ws_t[i] - ws_t[lo]) if n > 2 and ws_t[i] - ws_t[lo] > 0.05 and t - ws_t[i] < 0.25 else 0
    mbit = sum(w[4] for w in ws[lo:i + 1]) * 8 / 1e6
    srv = ws[i][5]
    a = f"{fps:4.1f} fps" if fps else "idle"
    b = f"{mbit:5.1f} Mbit/s   server frame {srv:4.1f} ms"
    wa = d.textlength(a, font=f_fps)
    d.rounded_rectangle((10, 10, 30 + wa + d.textlength(b, font=f_m) + 14, 46), 6, fill=(0, 0, 0, 170))
    d.text((20, 15), a, font=f_fps, fill=(230, 140, 40, 255))
    d.text((34 + wa, 21), b, font=f_m, fill=(190, 190, 190, 255))


enc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                        "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
n = int((t1 - t0) * FPS)
canvas, last, buf = Image.new("RGB", (W, H)), None, None
for k in range(n):
    t = t0 + k / FPS
    si = max(0, bisect.bisect_right(screen_t, t) - 1)
    i = bisect.bisect_right(ws_t, t) - 1
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    if i >= 0:
        meter(d, t, i)
    caption(d, t)
    key = (si, i, over.tobytes())
    if key != last:                                     # idle holds repeat the previous frame
        canvas.paste(scr.get(si).crop((VW, 0, W, H)), (VW, 0))
        if i >= 0:
            v = view.get(i)
            canvas.paste(v if v.size == (VW, H) else v.resize((VW, H), Image.BICUBIC), (0, 0))
        else:
            canvas.paste(scr.get(si).crop((0, 0, VW, H)), (0, 0))
        frame = canvas
        box = over.getbbox()
        if box:                                         # blend the overlay only where it has pixels
            frame = canvas.copy()
            frame.paste(Image.alpha_composite(canvas.crop(box).convert("RGBA"), over.crop(box)).convert("RGB"), box[:2])
        buf, last = np.asarray(frame).tobytes(), key
    enc.stdin.write(buf)
    if k % 600 == 0:
        print(f"{k}/{n}", flush=True)
enc.stdin.close()
enc.wait()
print("wrote", out, f"{n / FPS:.1f} s")
