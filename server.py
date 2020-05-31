import io
import bottle
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from bottle_tools import fill_args
import colorsys

DEFAULT_IMAGE_SIZE = 500


def n_to_escape(c, escape=2, max_iter=100):
    z = complex(0, 0)
    for i in range(max_iter):
        if abs(z) > escape:
            return True, i
        z = z ** 2 + c
    return False, 0


def do_job(numbers):
    results = []
    for c, i, j in numbers:
        results.append((c, i, j, n_to_escape(c)))
    return results


def get_image(xmin, xmax, ymin, ymax, n_cells=DEFAULT_IMAGE_SIZE):
    x_cells = np.linspace(xmin, xmax, num=n_cells)
    y_cells = np.linspace(ymin, ymax, num=n_cells)
    buf_size = 100
    total_args = (len(x_cells) * len(y_cells)) / buf_size

    def get_args():
        buf = []
        for i, x in enumerate(x_cells):
            for j, y in enumerate(y_cells):
                buf.append((complex(x, y), i, j))
                if len(buf) > buf_size:
                    yield buf
                    buf = []
        if len(buf) > buf_size:
            yield buf

    args = get_args()

    image = np.zeros((x_cells.shape[0], y_cells.shape[0]))
    with Pool() as pool:
        work = pool.imap_unordered(do_job, args)
        for results in tqdm(work, total=total_args):
            for c, i, j, (escaped, n) in results:
                if escaped:
                    image[j, i] = n

    buf = io.BytesIO()
    rescaled = (255.0 / image.max() * (image - image.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save(buf, format="png")
    buf.seek(0)
    return buf


app = bottle.Bottle()


@app.get("/image/<xmin>/<xmax>/<ymin>/<ymax>/<step>")
def image(xmin, xmax, ymin, ymax, step=DEFAULT_IMAGE_SIZE):
    img = get_image(float(xmin), float(xmax), float(ymin), float(ymax), int(step))
    resp = bottle.Response(body=img, status=200)
    resp.set_header("Content-Type", "image/png")
    return resp


@app.get("/", name="home")
@fill_args(coerce_types=True)
def home(
    zoom: float = 1.7,
    xm: float = -2,
    xM: float = 1,
    ym: float = -1.5,
    yM: float = 1.5,
    s: int = DEFAULT_IMAGE_SIZE,
    new_x: int = None,
    new_y: int = None,
    newxy: str = None,
):
    if newxy:
        x, y = newxy[1:].split(",")
        return bottle.redirect(
            app.get_url(
                "home", zoom=zoom, xm=xm, xM=xM, ym=ym, yM=yM, s=s, new_x=x, new_y=y,
            )
        )
    if new_x is not None and new_y is not None:
        x_cells = np.linspace(xm, xM, s)
        y_cells = np.linspace(ym, yM, s)
        dx = x_cells[new_x] - x_cells[s // 2]
        dy = y_cells[new_y] - y_cells[s // 2]
        xm += dx
        xM += dx
        ym += dy
        yM += dy
        # ---------- re-scale
        cx, cy = ((xM - xm) / 2) + xm, ((yM - ym) / 2) + ym
        dx, dy = (xM - cx) / zoom, (yM - cy) / zoom
        xm, xM = cx - dx, cx + dx
        ym, yM = cy - dy, cy + dy
        return bottle.redirect(
            app.get_url("home", zoom=zoom, xm=xm, xM=xM, ym=ym, yM=yM, s=s,)
        )
    image_link = f"http://localhost:8080/image/{xm}/{xM}/{ym}/{yM}/{s}"
    href_link = (
        app.get_url("home", zoom=zoom, xm=xm, xM=xM, ym=ym, yM=yM, s=s,) + "&newxy="
    )
    with open("index.html", "r") as fl:
        html = fl.read().format(
            image_link=image_link, href_link=href_link, xm=xm, xM=xM, ym=ym, yM=yM
        )
    resp = bottle.Response(body=html, status=200)
    resp.set_header("Content-type", "text/html")
    return resp


app.run()
