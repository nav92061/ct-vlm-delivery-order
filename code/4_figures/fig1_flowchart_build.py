"""Fig. 1 flowchart: ONE layout spec -> editable fig1_flowchart.drawio + fig1_flowchart.pdf/.png.
Units: points at final size (figure is 504 pt = 7.0 in wide). All numbers are read from the result tables."""
import base64, hashlib, json, os
import numpy as np, pandas as pd
from PIL import Image
from xml.sax.saxutils import escape

B = "figure_bundle/"
CT_INPUT = B + "ct_images/B1_truncation32pct_aligned.jpg"    # shared z-axis view (Stage 1)
CT_MODEL = B + "ct_images/B2_truncation32pct_native.jpg"     # native view = what the model received (Stage 2)

# ---------------- numbers (from tables, never typed) ----------------
ho = pd.read_csv(B + "data/header_override.csv"); mt = pd.read_csv(B + "data/mitigation.csv")
r = ho[(ho.model == "Nemotron-3.5-VL-120B") & (ho["style"] == "native")].iloc[0]
m = mt[(mt.model == "Nemotron-3.5-VL-120B") & (mt["style"] == "native") & (mt.condition == "image+conflict_header")
       & (mt.endpoint == "SFR_defective")].iloc[0]
cs = pd.read_csv("study_cases.csv")
P = lambda x: f"{100 * x:.0f}%"
N = dict(trunc=P(cs[(cs.pair_id == "P11") & (cs.variant == "truncation")].magnitude.iloc[0]),
         img=P(r.SFR_image_only), hdr=P(r.SFR_reassuring_header), ov=int(r.overridden_by_header),
         det=int(r.caught_from_image), ndef=int(r.n_defective), npat=int(r.n_patients),
         mit_base=P(m.original), mit=P(m.mitigated), mit_n=int(m.n), mit_unp=int(m.unparsed_mitigated),
         v_img=float(r.SFR_image_only), v_hdr=float(r.SFR_reassuring_header), v_mit=float(m.mitigated))

# ---------------- truncation boundary measured from the CT render ----------------
def boundary(path):
    im = np.asarray(Image.open(path).convert("L")).astype(float); H, W = im.shape
    on = np.r_[False, (np.abs(im - 90) > 12).mean(0) > 0.15, False]
    e = np.flatnonzero(np.diff(on.astype(int)))
    runs = [(e[i], e[i + 1]) for i in range(0, len(e) - 1, 2) if e[i + 1] - e[i] > 0.2 * W]
    def top(c0, c1):
        p = im[:, c0 + 3:c1 - 3]; panel = (np.abs(p - 90) > 12).mean(1) > 0.5
        rr = np.flatnonzero(panel & (p.max(1) > 40)); return rr.min(), rr.max()
    (a0, a1), (b0, b1) = top(*runs[0]), top(*runs[1])
    return dict(W=W, H=H, A=runs[0], Bc=runs[1], a_top=a0, a_bot=a1, b_top=b0, frac=(b0 - a0) / (a1 - a0 + 1))
BD = boundary(CT_INPUT)
_ts = pd.read_csv("two_stage_metrics.csv")
_t = _ts[(_ts.model == "Nemotron-3.5-VL-120B") & (_ts["style"] == "native") & (_ts.endpoint == "SFR_defective")].iloc[0]
_u = pd.read_csv("uniform_reversals.csv"); _u = _u[(_u.model == "nem") & (_u.condition == "two") & (_u["style"] == "native")].iloc[0]
N.update(two=P(_u.reversed / _u.refused_from_image), v_two=float(_u.reversed / _u.refused_from_image), rev=int(_u.reversed), s1=int(_u.refused_from_image))
_fm = pd.read_csv("frontier_metrics.csv"); _S = "claude-sonnet-5-5"
_sf = _fm[(_fm.model == _S) & (_fm["style"] == "native") & (_fm.set == "synthetic") & (_fm.endpoint == "SFR")].set_index("condition").rate
_pr = pd.read_csv("paired_order_effect.csv"); _pn = _pr[(_pr.model == "Nemotron") & (_pr["style"] == "native")].iloc[0]
_ps = _pr[(_pr.model == "Sonnet") & (_pr["style"] == "native")].iloc[0]
SON = {"A": (float(_sf["image_only"]), "image alone"),
       "B": (float(_ps.one), "%d/%d seen, admitted" % (_ps.k_one, _ps.n)),
       "C": (float(_ps.two), "%d/%d own refusals" % (_ps.k_two, _ps.n)),
       "D": (float(_sf["image+conflict_header+mitigation"]), "with instruction")}
NEM = {"A": (N["v_img"], "image alone"), "B": (float(_pn.one), "%d/%d seen, admitted" % (_pn.k_one, _pn.n)),
       "C": (float(_pn.two), "%d/%d own refusals" % (_pn.k_two, _pn.n)), "D": (N["v_mit"], "with instruction\u2020")}
N.update(son={k: round(v[0], 4) for k, v in SON.items()})

# ---------------- palette / type ----------------
NAVY, TXT, MUTE, LINE = "#1F3A68", "#2B2B2B", "#666666", "#9E9E9E"
RED, RED_L, GRN, GRN_L, YEL, YEL_E, PANEL = "#B2182B", "#FBE9EA", "#1B7837", "#E8F3E6", "#FFF3C4", "#C9A227", "#FAFAFA"
F, FM = "Helvetica", "Courier New"

E = []   # elements
def rect(x, y, w, h, fill="#FFFFFF", stroke=LINE, lw=0.6, r=3, alpha=1.0, dash=False, name=None):
    E.append(dict(t="rect", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, lw=lw, r=r, alpha=alpha, dash=dash, name=name))
def text(x, y, w, h, s, size=6.5, color=TXT, bold=False, italic=False, ha="left", va="center", font=F, name=None, check=True):
    E.append(dict(t="text", x=x, y=y, w=w, h=h, s=s, size=size, color=color, bold=bold, italic=italic, ha=ha, va=va,
                  font=font, name=name, check=check))
def stamp(x, y, w, h, s, rot=-6):
    E.append(dict(t="stamp", x=x, y=y, w=w, h=h, s=s, rot=rot))
def image(x, y, w, path):
    im = Image.open(path); h = w * im.size[1] / im.size[0]
    E.append(dict(t="image", x=x, y=y, w=w, h=h, path=path)); return h
def ellipse(x, y, w, h, fill, stroke="none", lw=0.8):
    E.append(dict(t="ellipse", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, lw=lw))
def tri(x, y, w, h, fill):
    E.append(dict(t="tri", x=x, y=y, w=w, h=h, fill=fill))
def note(x, y, w, h, fill, stroke, lw=0.8):
    E.append(dict(t="note", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, lw=lw))
def line(pts, color=LINE, lw=0.75, arrow=False, dash=False, icon=False):
    E.append(dict(t="line", pts=pts, color=color, lw=lw, arrow=arrow, dash=dash, icon=icon))

def check_icon(cx, cy, d, fill=GRN):
    ellipse(cx - d / 2, cy - d / 2, d, d, fill)
    line([(cx - .26 * d, cy + .02 * d), (cx - .06 * d, cy + .22 * d), (cx + .27 * d, cy - .2 * d)], "#FFFFFF", 1.1, icon=True)
def cross_icon(cx, cy, d, fill=RED):
    ellipse(cx - d / 2, cy - d / 2, d, d, fill); k = .2 * d
    line([(cx - k, cy - k), (cx + k, cy + k)], "#FFFFFF", 1.1, icon=True); line([(cx - k, cy + k), (cx + k, cy - k)], "#FFFFFF", 1.1, icon=True)
def warn_icon(x, y, s, fill="#4D4D4D"):
    tri(x, y, s, s * .9, fill); text(x, y + s * .22, s, s * .68, "!", 6.5, "#FFFFFF", bold=True, ha="center", check=False)
def eye_icon(cx, cy, w):
    ellipse(cx - w / 2, cy - w * .3, w, w * .6, "#FFFFFF", NAVY, 1.0); ellipse(cx - w * .14, cy - w * .14, w * .28, w * .28, NAVY)
def doc_icon(x, y, w, h):
    note(x, y, w, h, "#FFFFFF", RED, 1.0); text(x, y + h * .18, w, h * .7, "!", 7, RED, bold=True, ha="center", check=False)

# ================= LAYOUT (compact, 4 rows) =================
WFIG = 504
X1, X1e = 4, 116          # stage 1
X2, X2e = 128, 300        # stage 2
XV, XVe = 306, 340        # VLM
X3, X3e = 350, 500        # stage 3
Y0 = 22; RH, G = 33, 4
rows = ["A", "B", "C", "D"]
ys = {k: Y0 + n * (RH + G) for n, k in enumerate(rows)}
cy = {k: ys[k] + RH / 2 for k in rows}
yEnd = ys["D"] + RH

text(X1, 3, 150, 10, "1  CT input", 8, NAVY, bold=True)
text(X2, 3, 190, 10, "2  What the model is given", 8, NAVY, bold=True)
text(X3, 3, 150, 10, "3  Defective pairs admitted", 8, NAVY, bold=True)
text(X3 + 2, 12, 72, 8, "Nemotron-3.5-VL", 6.5, NAVY, bold=True, ha="center")
text(X3 + 76, 12, 72, 8, "Claude Sonnet 5.5", 6.5, NAVY, bold=True, ha="center")

# ---- Stage 1: one CT pair, vertically centred
iw = X1e - X1 - 6
im_h = Image.open(CT_INPUT).size[1] * iw / Image.open(CT_INPUT).size[0]
ph = im_h + 40; py = (Y0 + yEnd) / 2 - ph / 2
rect(X1, py, X1e - X1, ph, PANEL, "#D0D0D0", 0.6)
ix, iy = X1 + 3, py + 3
image(ix, iy, iw, CT_INPUT); s = iw / BD["W"]
sx0, sx1 = ix + BD["Bc"][0] * s, ix + BD["Bc"][1] * s
sy0, sy1 = iy + BD["a_top"] * s, iy + BD["b_top"] * s
rect(sx0, sy0, sx1 - sx0, sy1 - sy0, RED, "none", 0, 0, alpha=0.38, name="shade")
line([(ix + BD["A"][0] * s, sy1), (sx1, sy1)], RED, 0.9, dash=True)
text(sx0, sy0, sx1 - sx0, sy1 - sy0, "%s missing" % N["trunc"], 6.5, "#FFFFFF", bold=True, ha="center")
yy = iy + im_h + 3
text(X1 + 4, yy, iw, 16, "Series B truncated;\nblack = no data", 6.5, MUTE)
warn_icon(X1 + 4, yy + 20, 9)
text(X1 + 16, yy + 18, 94, 11, "Ground truth: REFUSE", 6.5, TXT, bold=True)

# ---- Stage 2 cards
HDR = "B: same z-range, 0 gaps"
spec = {"A": ("A  Image only", None), "B": ("B  Image + false header", "hdr"),
        "C": ("C  Image first, false header next turn", "two"), "D": ("D  Image + false header + instruction", "mit")}
TW = round(19.5 * 580 / 348, 1)
for k in rows:
    y = ys[k]; title, kind = spec[k]
    good = kind == "mit"
    rect(X2, y, X2e - X2, RH, "#FFFFFF", GRN if good else "#C8C8C8", 0.8 if good else 0.6)
    if good: rect(X2, y, 2.6, RH, GRN, "none", 0, 0)
    text(X2 + 5, y + 1.5, X2e - X2 - 8, 9, title, 7, GRN if good else NAVY, bold=True)
    image(X2 + 5, y + 11.5, TW, CT_MODEL)
    tx = X2 + 5 + TW + 6; tw = X2e - tx - 4
    if kind is None:
        text(tx, y + 12, tw, 18, "No metadata", 6.5, MUTE, italic=True)
        continue
    if kind == "two":
        text(tx, y + 11, tw, 9, "Turn 1: model answers REFUSE", 6.5, MUTE, italic=True)
        hy = y + 21
    elif kind == "mit":
        hy = y + 11
    else:
        hy = y + 16
    note(tx, hy, 92, 10, YEL, YEL_E, 0.6)
    text(tx + 2, hy, 90, 10, HDR, 6.5, TXT, font=FM)
    stamp(tx + 94, hy - 0.5, 23, 10, "FALSE")
    if kind == "mit":
        text(tx, y + 22, tw, 9, "\u201cIf they disagree, trust the image.\u201d", 6.5, GRN, italic=True)

# ---- fan-out
fx = X1e + 5
for k in rows:
    line([(X1e, (Y0 + yEnd) / 2), (fx, (Y0 + yEnd) / 2), (fx, cy[k]), (X2, cy[k])], LINE, 0.7, arrow=True)

# ---- VLM chip
cw = 26; chx = (XV + XVe) / 2 - cw / 2; mid = (Y0 + yEnd) / 2; chy = mid - cw / 2
for d in (-7, 0, 7):
    line([(chx + cw / 2 + d, chy - 3.2), (chx + cw / 2 + d, chy)], NAVY, 1.0, icon=True)
    line([(chx + cw / 2 + d, chy + cw), (chx + cw / 2 + d, chy + cw + 3.2)], NAVY, 1.0, icon=True)
rect(chx, chy, cw, cw, NAVY, NAVY, 0.6, 3)
text(chx, chy, cw, cw, "VLM", 7, "#FFFFFF", bold=True, ha="center", check=False)
ein = {"A": chy + 4, "B": chy + 10, "C": chy + 16, "D": chy + 22}
xin, xout = X2e + 4, X3 - 5
COL = {k: LINE for k in rows}
for k in rows:
    line([(X2e, cy[k]), (xin, cy[k]), (xin, ein[k]), (chx, ein[k])], LINE, 0.7, arrow=True)
    line([(chx + cw, ein[k]), (xout, ein[k]), (xout, cy[k]), (X3, cy[k])], COL[k], 0.9 if COL[k] == GRN else 1.1, arrow=True)

# ---- Stage 3 outcomes: two models side by side
CW = 72
def cell(x, y, v, sub):
    ok = v < 0.5; col, light = (GRN, GRN_L) if ok else (RED, RED_L)
    rect(x, y, CW, RH, light, col, 0.8)
    (check_icon if ok else cross_icon)(x + 8, y + 7.5, 8)
    text(x + 14, y + 2, CW - 16, 11, "%d%%" % round(100 * v), 8, col, bold=True)
    rect(x + 4, y + 15, CW - 8, 4, "#FFFFFF", "#BDBDBD", 0.4, 0)
    rect(x + 4, y + 15, max((CW - 8) * v, 0.6), 4, col, "none", 0, 0, name="bar")
    text(x + 2.5, y + 21, CW - 3.5, 9, sub, 6.5, TXT if not ok else MUTE)
for k in rows:
    cell(X3 + 2, ys[k], *NEM[k]); cell(X3 + 76, ys[k], *SON[k])
HFIG = yEnd + 3

# ================= drawio writer =================
def _img_uri(path):
    return "data:image/jpeg," + base64.b64encode(open(path, "rb").read()).decode()

def to_drawio():
    cells, k = [], [0]
    def nid(): k[0] += 1; return f"c{k[0]}"
    def v(val, st, x, y, w, h):
        cells.append(f'<mxCell id="{nid()}" value="{escape(val, {chr(34): "&quot;"})}" style="{escape(st)}" vertex="1" parent="1">'
                     f'<mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/></mxCell>')
    for e in E:
        t = e["t"]
        if t == "rect":
            st = (f"rounded={1 if e['r'] else 0};absoluteArcSize=1;arcSize={2 * e['r']};whiteSpace=wrap;html=1;"
                  f"fillColor={e['fill']};strokeColor={e['stroke']};strokeWidth={e['lw']};")
            if e["alpha"] < 1: st += f"opacity={int(e['alpha'] * 100)};"
            v("", st, e["x"], e["y"], e["w"], e["h"])
        elif t == "text":
            fs = (1 if e["bold"] else 0) + (2 if e["italic"] else 0)
            st = (f"text;html=1;whiteSpace=nowrap;spacing=0;fillColor=none;strokeColor=none;fontFamily={e['font']};"
                  f"fontSize={e['size']};fontColor={e['color']};fontStyle={fs};align={e['ha']};"
                  f"verticalAlign={'middle' if e['va'] == 'center' else e['va']};")
            v(e["s"].replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;"), st, e["x"], e["y"], e["w"], e["h"])
        elif t == "stamp":
            st = (f"rounded=1;absoluteArcSize=1;arcSize=3;html=1;fillColor={RED};strokeColor={RED};fontColor=#FFFFFF;"
                  f"fontFamily={F};fontSize=6.5;fontStyle=1;rotation={e['rot']};spacing=0;")
            v(e["s"], st, e["x"], e["y"], e["w"], e["h"])
        elif t == "image":
            st = f"shape=image;html=1;imageAspect=0;aspect=fixed;image={_img_uri(e['path'])};"
            v("", st, e["x"], e["y"], e["w"], e["h"])
        elif t == "ellipse":
            st = f"ellipse;html=1;fillColor={e['fill']};strokeColor={e['stroke']};strokeWidth={e['lw']};"
            v("", st, e["x"], e["y"], e["w"], e["h"])
        elif t == "tri":
            v("", f"triangle;direction=north;html=1;fillColor={e['fill']};strokeColor=none;", e["x"], e["y"], e["w"], e["h"])
        elif t == "note":
            v("", f"shape=note;size=4;html=1;fillColor={e['fill']};strokeColor={e['stroke']};strokeWidth={e['lw']};",
              e["x"], e["y"], e["w"], e["h"])
        elif t == "line":
            p = e["pts"]
            st = (f"html=1;rounded=0;strokeColor={e['color']};strokeWidth={e['lw']};"
                  f"endArrow={'block' if e['arrow'] else 'none'};endFill=1;endSize=4;startArrow=none;"
                  f"{'dashed=1;dashPattern=3 2;' if e['dash'] else ''}")
            mid = "".join(f'<mxPoint x="{x:.2f}" y="{y:.2f}"/>' for x, y in p[1:-1])
            cells.append(f'<mxCell id="{nid()}" style="{escape(st)}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                         f'<mxPoint x="{p[0][0]:.2f}" y="{p[0][1]:.2f}" as="sourcePoint"/>'
                         f'<mxPoint x="{p[-1][0]:.2f}" y="{p[-1][1]:.2f}" as="targetPoint"/>'
                         + (f'<Array as="points">{mid}</Array>' if mid else "") + "</mxGeometry></mxCell>")
    return ('<mxfile host="app.diagrams.net"><diagram name="Fig1" id="fig1"><mxGraphModel dx="800" dy="400" grid="0" '
            f'gridSize="2" page="1" pageWidth="{WFIG}" pageHeight="{int(HFIG) + 1}" background="#FFFFFF"><root>'
            '<mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")

# ================= matplotlib renderer =================
def render(out, dpi=None, interp="none"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.patches as mp
    from matplotlib.transforms import Affine2D
    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.family": "Arial"})
    fam = {F: "Arial", FM: "Courier New"}
    fig = plt.figure(figsize=(WFIG / 72, HFIG / 72)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WFIG); ax.set_ylim(HFIG, 0); ax.axis("off")
    texts, segs = [], []
    for z, e in enumerate(E):
        t = e["t"]
        if t == "rect":
            kw = dict(fc=e["fill"] if e["fill"] != "none" else "none", ec=e["stroke"] if e["stroke"] != "none" else "none",
                      lw=e["lw"], alpha=e["alpha"], zorder=z)
            if e["r"]: ax.add_patch(mp.FancyBboxPatch((e["x"], e["y"]), e["w"], e["h"],
                                                      boxstyle=f"round,pad=0,rounding_size={e['r']}", **kw))
            else: ax.add_patch(mp.Rectangle((e["x"], e["y"]), e["w"], e["h"], **kw))
        elif t == "stamp":
            cx, cy = e["x"] + e["w"] / 2, e["y"] + e["h"] / 2
            p = mp.FancyBboxPatch((e["x"], e["y"]), e["w"], e["h"], boxstyle="round,pad=0,rounding_size=1.5",
                                  fc=RED, ec=RED, lw=0.5, zorder=z)
            p.set_transform(Affine2D().rotate_deg_around(cx, cy, e["rot"]) + ax.transData); ax.add_patch(p)
            ax.text(cx, cy, e["s"], ha="center", va="center", fontsize=6.5, color="#FFFFFF", fontweight="bold",
                    rotation=-e["rot"], family="Arial", zorder=z + 0.5)
        elif t == "image":
            arr = np.asarray(Image.open(e["path"]).convert("RGB"))
            ax.imshow(arr, extent=(e["x"], e["x"] + e["w"], e["y"] + e["h"], e["y"]), interpolation=interp, zorder=z)
        elif t == "ellipse":
            ax.add_patch(mp.Ellipse((e["x"] + e["w"] / 2, e["y"] + e["h"] / 2), e["w"], e["h"], fc=e["fill"],
                                    ec=e["stroke"] if e["stroke"] != "none" else "none", lw=e["lw"], zorder=z))
        elif t == "tri":
            ax.add_patch(mp.Polygon([(e["x"], e["y"] + e["h"]), (e["x"] + e["w"] / 2, e["y"]), (e["x"] + e["w"], e["y"] + e["h"])],
                                    closed=True, fc=e["fill"], ec="none", zorder=z))
        elif t == "note":
            x, y, w, h, c = e["x"], e["y"], e["w"], e["h"], 4
            ax.add_patch(mp.Polygon([(x, y), (x + w - c, y), (x + w, y + c), (x + w, y + h), (x, y + h)], closed=True,
                                    fc=e["fill"], ec=e["stroke"], lw=e["lw"], zorder=z))
        elif t == "line":
            xs, ys = zip(*e["pts"])
            ax.plot(xs, ys, color=e["color"], lw=e["lw"], ls=(0, (3, 2)) if e["dash"] else "-",
                    solid_capstyle="butt", zorder=z)
            if e["arrow"]:
                (x0, y0), (x1, y1) = e["pts"][-2], e["pts"][-1]
                d = np.array([x1 - x0, y1 - y0]); d = d / np.linalg.norm(d); nrm = np.array([-d[1], d[0]])
                tip = np.array([x1, y1]); L, Wd = 4.0, 1.7
                ax.add_patch(mp.Polygon([tip, tip - L * d + Wd * nrm, tip - L * d - Wd * nrm], closed=True,
                                        fc=e["color"], ec="none", zorder=z))
            if not e["icon"]:
                segs += [(e["pts"][i], e["pts"][i + 1]) for i in range(len(e["pts"]) - 1)]
        elif t == "text":
            x = {"left": e["x"], "center": e["x"] + e["w"] / 2, "right": e["x"] + e["w"]}[e["ha"]]
            y = {"top": e["y"], "center": e["y"] + e["h"] / 2, "bottom": e["y"] + e["h"]}[e["va"]]
            tt = ax.text(x, y, e["s"], ha=e["ha"], va=e["va"] if e["va"] != "center" else "center", fontsize=e["size"],
                         color=e["color"], fontweight="bold" if e["bold"] else "normal",
                         fontstyle="italic" if e["italic"] else "normal", family=fam[e["font"]], linespacing=1.12,
                         multialignment=e["ha"], zorder=z + 0.5)
            texts.append((tt, e))
    fig.savefig(out, dpi=dpi)
    # ---- checks on the rendered geometry
    rnd = fig.canvas.get_renderer(); inv = ax.transData.inverted(); issues = []
    bbs = []
    for tt, e in texts:
        bb = tt.get_window_extent(rnd); (x0, y1), (x1, y0) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        bbs.append((x0, y0, x1, y1, e))
        if e["check"] and (x0 < e["x"] - 0.6 or x1 > e["x"] + e["w"] + 0.6 or y0 < e["y"] - 0.8 or y1 > e["y"] + e["h"] + 0.8):
            issues.append(("overflow", e["s"][:30], round(x0 - e["x"], 1), round(e["x"] + e["w"] - x1, 1), round(y0 - e["y"], 1), round(e["y"] + e["h"] - y1, 1)))
        if x1 > WFIG or y1 > HFIG or x0 < 0 or y0 < 0: issues.append(("offpage", e["s"][:30]))
    for i in range(len(bbs)):
        for j in range(i + 1, len(bbs)):
            a, b = bbs[i], bbs[j]
            if a[0] < b[2] - 0.3 and b[0] < a[2] - 0.3 and a[1] < b[3] - 0.3 and b[1] < a[3] - 0.3:
                issues.append(("text-overlap", a[4]["s"][:25], b[4]["s"][:25]))
    def seg_hits(p, q, bb):
        x0, y0, x1, y1 = bb
        for tt_ in np.linspace(0, 1, 60):
            x, y = p[0] + (q[0] - p[0]) * tt_, p[1] + (q[1] - p[1]) * tt_
            if x0 + 0.3 < x < x1 - 0.3 and y0 + 0.3 < y < y1 - 0.3: return True
        return False
    for (x0, y0, x1, y1, e) in bbs:
        for p, q in segs:
            if seg_hits(p, q, (x0, y0, x1, y1)): issues.append(("line-through-text", e["s"][:25]))
    for e in E:
        if e["t"] == "stamp":
            for (x0, y0, x1, y1, te) in bbs:
                if te["font"] == FM and x0 < e["x"] + e["w"] and e["x"] < x1 and y0 < e["y"] + e["h"] and e["y"] < y1:
                    issues.append(("stamp-over-text", te["s"][:25]))
    small = [e["s"][:20] for _, e in texts if e["size"] < 6.5]
    plt.close(fig)
    return issues, small

if __name__ == "__main__":
    open("fig1_flowchart.drawio", "w").write(to_drawio())
    iss, small = render("fig1_flowchart.pdf", interp="none")
    render("fig1_flowchart.png", dpi=600, interp="antialiased")
    json.dump(dict(N={k: v for k, v in N.items()}, boundary_frac=BD["frac"], H_pt=HFIG, issues=iss, small=small),
              open("fig1_checks.json", "w"), indent=1, default=str)
    print("H_pt", round(HFIG, 1), "in", round(HFIG / 72, 2), "| issues", len(iss), "| <6.5pt", small)
    for i in iss: print("  ", i)
