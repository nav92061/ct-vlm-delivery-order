"""Download each NLST pair from IDC S3, extract a physically matched mid-coronal plane per series,
save a compact .npz, then delete the DICOMs. Resumable."""
import os, re, json, time, shutil, numpy as np, pandas as pd, pydicom, requests
from concurrent.futures import ThreadPoolExecutor

B = "https://idc-open-data.s3.amazonaws.com"
S = requests.Session()
urls = json.load(open("main_urls.json"))
main = pd.read_csv("main_pairs.csv"); main = main[main.pair_id.isin(["P10","P11","P12","P13","P14"])]
os.makedirs("planes", exist_ok=True); os.makedirs("dcm_tmp", exist_ok=True)

def list_keys(prefix):
    keys, tok = [], None
    while True:
        p = {"list-type": "2", "prefix": prefix}
        if tok: p["continuation-token"] = tok
        x = S.get(B + "/", params=p, timeout=60).text
        keys += re.findall(r"<Key>([^<]+)</Key>", x)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", x)
        if not m: return keys
        tok = m.group(1)

def getk(k, out):
    fn = os.path.join(out, k.split("/")[-1])
    if os.path.exists(fn) and os.path.getsize(fn) > 1000: return
    for a in range(6):
        try:
            r = S.get(f"{B}/{k}", timeout=90); r.raise_for_status(); open(fn, "wb").write(r.content); return
        except Exception: time.sleep(1 + 2 * a)
    raise RuntimeError(k)

def fetch(uid):
    src = f"dcm/{uid}"
    if os.path.isdir(src) and len(os.listdir(src)) > 50: return src   # pilot cache
    out = f"dcm_tmp/{uid}"; os.makedirs(out, exist_ok=True)
    keys = list_keys(urls[uid].replace("s3://idc-open-data/", "").rstrip("*"))
    with ThreadPoolExecutor(4) as ex: list(ex.map(lambda k: getk(k, out), keys))
    assert len(os.listdir(out)) == len(keys)
    return out

def read(folder):
    ds = [pydicom.dcmread(os.path.join(folder, f)) for f in os.listdir(folder)]
    ds = [d for d in ds if hasattr(d, "ImagePositionPatient")]
    ds.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    z = np.array([float(d.ImagePositionPatient[2]) for d in ds])
    x0, y0 = float(ds[0].ImagePositionPatient[0]), float(ds[0].ImagePositionPatient[1])
    dy, dx = map(float, ds[0].PixelSpacing)
    return ds, z, x0, y0, dx, dy, float(ds[0].SliceThickness)

def plane(ds, y0, dy, y_target):
    row = int(np.clip(round((y_target - y0) / dy), 0, int(ds[0].Rows) - 1))
    return np.stack([d.pixel_array[row].astype(np.float32) * float(d.RescaleSlope) + float(d.RescaleIntercept) for d in ds]).astype(np.int16)

t0 = time.time()
for i, r in main.iterrows():
    out = f"planes/{r.pair_id}.npz"
    if os.path.exists(out): continue
    fa, fb = fetch(r.sa), fetch(r.sb)
    A = read(fa); Bs = read(fb)
    y_t = A[3] + A[5] * int(A[0][0].Rows) / 2                     # physical mid-coronal y of A's FOV
    np.savez_compressed(out,
        ca=plane(A[0], A[3], A[5], y_t), za=A[1], xa0=A[2], dxa=A[4], ta=A[6],
        cb=plane(Bs[0], Bs[3], Bs[5], y_t), zb=Bs[1], xb0=Bs[2], dxb=Bs[4], tb=Bs[6], y_target=y_t)
    for f in (fa, fb):
        shutil.rmtree(f, ignore_errors=True)
    print(i, r.pair_id, round(time.time() - t0), flush=True)
print("DONE", flush=True)
