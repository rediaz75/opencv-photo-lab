"""
OpenCV Photo Lab
----------------
A Streamlit app that puts four OpenCV concepts behind live controls:

  1. Annotate              - cv2.line / cv2.circle / cv2.rectangle / cv2.putText
  2. Brightness & Contrast - cv2.add / cv2.subtract / cv2.multiply, with clipping on/off
  3. Thresholding          - global, inverse, adaptive (mean / gaussian), Otsu
  4. Background Remover    - threshold mask, cv2.bitwise_not, cv2.split / cv2.merge -> RGBA PNG

Run locally:  streamlit run app.py
"""

import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="OpenCV Photo Lab", page_icon="🖼️", layout="wide")

MAX_SIDE = 1400  # downscale very large uploads so sliders stay responsive


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def decode_image(data: bytes) -> np.ndarray:
    """Bytes -> BGR uint8 image (same layout cv2.imread gives you)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


@st.cache_data(show_spinner=False)
def sample_image() -> np.ndarray:
    """A built-in demo picture so the app works before anything is uploaded."""
    h, w = 480, 720
    y = np.linspace(0, 1, h)[:, None]
    sky = np.zeros((h, w, 3), np.uint8)
    sky[..., 0] = (235 - 80 * y).astype(np.uint8)      # B
    sky[..., 1] = (200 - 90 * y).astype(np.uint8)      # G
    sky[..., 2] = (150 - 110 * y).astype(np.uint8)     # R
    img = sky.copy()
    cv2.circle(img, (560, 110), 55, (90, 220, 255), -1, cv2.LINE_AA)            # sun
    pts = np.array([[0, 380], [170, 210], [300, 330], [430, 180], [720, 390], [720, 480], [0, 480]])
    cv2.fillPoly(img, [pts], (60, 90, 70), cv2.LINE_AA)                        # mountains
    cv2.rectangle(img, (0, 420), (720, 480), (40, 60, 45), -1)                  # ground
    cv2.rectangle(img, (110, 300), (210, 420), (40, 40, 170), -1)               # house
    tri = np.array([[100, 300], [160, 250], [220, 300]])
    cv2.fillPoly(img, [tri], (30, 30, 90), cv2.LINE_AA)
    return img


def to_rgb(img: np.ndarray) -> np.ndarray:
    """OpenCV stores BGR; Streamlit expects RGB."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def checkerboard(h, w, size=16):
    yy, xx = np.indices((h, w))
    board = (((yy // size) + (xx // size)) % 2).astype(np.uint8)
    board = np.where(board[..., None] == 1, 205, 245).astype(np.uint8)
    return np.repeat(board, 3, axis=2)


def composite_on_checker(bgra: np.ndarray) -> np.ndarray:
    """Show transparency by blending the RGBA result over a checkerboard."""
    h, w = bgra.shape[:2]
    bg = checkerboard(h, w).astype(np.float32)
    alpha = bgra[..., 3:4].astype(np.float32) / 255.0
    fg = bgra[..., :3].astype(np.float32)
    return (fg * alpha + bg * (1 - alpha)).astype(np.uint8)


def before_after(before, after, before_caption="Before (original)", after_caption="After"):
    c1, c2 = st.columns(2)
    with c1:
        st.image(to_rgb(before), caption=before_caption, width="stretch")
    with c2:
        st.image(to_rgb(after), caption=after_caption, width="stretch")


def wrap_uint8(arr_float: np.ndarray) -> np.ndarray:
    """Emulate uint8 overflow: 260 -> 4, -10 -> 246 (what np.uint8() does without clipping)."""
    return np.mod(np.floor(arr_float), 256).astype(np.uint8)


# ------------------------------------------------------------------
# Header + upload
# ------------------------------------------------------------------
st.title("🖼️ OpenCV Photo Lab")
st.write(
    "Upload a photo, pick a tab, and move the controls. The **After** image updates live, "
    "and each tab shows the OpenCV call it is running."
)

uploaded = st.file_uploader("Upload an image (JPG, PNG, WEBP, BMP)", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded is not None:
    image = decode_image(uploaded.getvalue())
    if image is None:
        st.error("Couldn't read that file as an image. Try a JPG or PNG.")
        st.stop()
    base_name = uploaded.name.rsplit(".", 1)[0]
else:
    image = sample_image()
    base_name = "sample"
    st.info("No image uploaded yet, so a built-in sample picture is shown. Upload your own above.")

H, W = image.shape[:2]
st.caption(f"Working image: {W} × {H} px")

tab_annot, tab_arith, tab_thresh, tab_bg = st.tabs(
    ["✏️ Annotate", "☀️ Brightness & Contrast", "⚫ Thresholding", "✂️ Background Remover"]
)

# ------------------------------------------------------------------
# 1. Annotate
# ------------------------------------------------------------------
with tab_annot:
    st.subheader("Draw shapes and add text")
    ctrl, view = st.columns([1, 3])

    with ctrl:
        shape = st.selectbox("Shape", ["Rectangle", "Circle", "Line", "None"])
        shape_color = st.color_picker("Shape color", "#FF00FF")
        thickness = st.slider("Thickness", 1, 20, 3)
        filled = st.checkbox("Filled (thickness = -1)", value=False, disabled=shape in ("Line", "None"))
        line_type_name = st.radio("Line type", ["LINE_AA (anti-aliased)", "LINE_8"], horizontal=False)
        line_type = cv2.LINE_AA if line_type_name.startswith("LINE_AA") else cv2.LINE_8

        st.markdown("**Shape position** (% of width / height)")
        if shape in ("Rectangle", "Line"):
            x1 = st.slider("Start x %", 0, 100, 20)
            y1 = st.slider("Start y %", 0, 100, 25)
            x2 = st.slider("End x %", 0, 100, 60)
            y2 = st.slider("End y %", 0, 100, 80)
        elif shape == "Circle":
            cx = st.slider("Center x %", 0, 100, 50)
            cy = st.slider("Center y %", 0, 100, 50)
            rad = st.slider("Radius % of shorter side", 1, 50, 15)

        st.divider()
        text = st.text_input("Caption text", "Hello, OpenCV!")
        fonts = {
            "HERSHEY_SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
            "HERSHEY_PLAIN": cv2.FONT_HERSHEY_PLAIN,
            "HERSHEY_DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
            "HERSHEY_COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
            "HERSHEY_TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
            "HERSHEY_SCRIPT_SIMPLEX": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        }
        font_name = st.selectbox("Font", list(fonts.keys()))
        font_scale = st.slider("Font scale", 0.3, 6.0, 1.5, 0.1)
        font_thick = st.slider("Text thickness", 1, 10, 2)
        text_color = st.color_picker("Text color", "#00FF00")
        text_pos = st.selectbox("Text position", ["Bottom center", "Top center", "Top left", "Center"])

    # --- apply ---
    annotated = image.copy()
    sc = hex_to_bgr(shape_color)
    t = -1 if filled else thickness
    code_lines = ["image_out = image.copy()"]

    if shape == "Rectangle":
        p1 = (int(x1 / 100 * W), int(y1 / 100 * H))
        p2 = (int(x2 / 100 * W), int(y2 / 100 * H))
        cv2.rectangle(annotated, p1, p2, sc, t, line_type)
        code_lines.append(f"cv2.rectangle(image_out, {p1}, {p2}, {sc}, thickness={t}, lineType=cv2.{line_type_name.split()[0]})")
    elif shape == "Circle":
        c = (int(cx / 100 * W), int(cy / 100 * H))
        r = max(1, int(rad / 100 * min(W, H)))
        cv2.circle(annotated, c, r, sc, t, line_type)
        code_lines.append(f"cv2.circle(image_out, {c}, {r}, {sc}, thickness={t}, lineType=cv2.{line_type_name.split()[0]})")
    elif shape == "Line":
        p1 = (int(x1 / 100 * W), int(y1 / 100 * H))
        p2 = (int(x2 / 100 * W), int(y2 / 100 * H))
        cv2.line(annotated, p1, p2, sc, thickness, line_type)
        code_lines.append(f"cv2.line(image_out, {p1}, {p2}, {sc}, thickness={thickness}, lineType=cv2.{line_type_name.split()[0]})")

    if text.strip():
        font = fonts[font_name]
        (tw, th), base = cv2.getTextSize(text, font, font_scale, font_thick)
        if text_pos == "Bottom center":
            org = ((W - tw) // 2, H - 25)
        elif text_pos == "Top center":
            org = ((W - tw) // 2, th + 25)
        elif text_pos == "Top left":
            org = (25, th + 25)
        else:
            org = ((W - tw) // 2, (H + th) // 2)
        tc = hex_to_bgr(text_color)
        cv2.putText(annotated, text, org, font, font_scale, tc, font_thick, line_type)
        code_lines.append(f"(tw, th), _ = cv2.getTextSize({text!r}, cv2.FONT_{font_name}, {font_scale}, {font_thick})")
        code_lines.append(f"cv2.putText(image_out, {text!r}, {org}, cv2.FONT_{font_name}, {font_scale}, {tc}, {font_thick}, cv2.{line_type_name.split()[0]})")

    with view:
        before_after(image, annotated, after_caption="After (annotated)")
        with st.expander("Show the OpenCV code for this result"):
            st.code("\n".join(code_lines), language="python")
            st.caption("Colors are (B, G, R) because OpenCV stores channels in BGR order.")
        st.download_button("⬇️ Download annotated PNG", png_bytes(annotated),
                           f"{base_name}_annotated.png", "image/png")

# ------------------------------------------------------------------
# 2. Brightness & Contrast (image arithmetic)
# ------------------------------------------------------------------
with tab_arith:
    st.subheader("Image arithmetic: brightness, contrast and clipping")
    ctrl, view = st.columns([1, 3])

    with ctrl:
        brightness = st.slider("Brightness (add / subtract)", -150, 150, 40)
        contrast = st.slider("Contrast (multiply factor)", 0.0, 3.0, 1.2, 0.05)
        clip = st.checkbox("Clip to 0–255 (prevent overflow)", value=True)
        st.caption(
            "Turn clipping **off** to see uint8 overflow: bright pixels wrap around to dark "
            "(e.g. 230 × 1.2 = 276 → 20) and dark pixels wrap to bright when you subtract."
        )

    img_f = image.astype(np.float64)
    # Contrast: cv2.multiply on float64, like the notebook.
    contrasted = cv2.multiply(img_f, np.ones(image.shape) * contrast)
    # Brightness: add or subtract a constant matrix.
    raw = contrasted + brightness  # true (unbounded) values, for the overflow statistics

    if clip:
        c8 = np.uint8(np.clip(contrasted, 0, 255))
        matrix = np.ones(image.shape, dtype="uint8") * abs(brightness)
        result = cv2.add(c8, matrix) if brightness >= 0 else cv2.subtract(c8, matrix)  # saturating
        code = (
            f"img_c = np.uint8(np.clip(cv2.multiply(np.float64(img), np.ones(img.shape) * {contrast}), 0, 255))\n"
            f"matrix = np.ones(img.shape, dtype='uint8') * {abs(brightness)}\n"
            f"img_out = cv2.{'add' if brightness >= 0 else 'subtract'}(img_c, matrix)   # saturates at 0 / 255"
        )
    else:
        result = wrap_uint8(raw)
        code = (
            f"img_c = np.uint8(cv2.multiply(np.float64(img), np.ones(img.shape) * {contrast}))   # no clip -> wraps\n"
            f"matrix = np.ones(img.shape, dtype='uint8') * {abs(brightness)}\n"
            f"img_out = img_c {'+' if brightness >= 0 else '-'} matrix   # NumPy uint8 math wraps modulo 256"
        )

    over = float(np.mean(raw > 255) * 100)
    under = float(np.mean(raw < 0) * 100)

    with view:
        before_after(image, result, after_caption=f"After (contrast ×{contrast}, brightness {brightness:+d}, clipping {'ON' if clip else 'OFF'})")
        m1, m2, m3 = st.columns(3)
        m1.metric("Channel values > 255", f"{over:.1f}%")
        m2.metric("Channel values < 0", f"{under:.1f}%")
        m3.metric("Handled by", "Clipping" if clip else "Wraparound")

        st.markdown("**Pixel inspector:** what happens to a few sample values")
        samples = np.array([20, 100, 150, 200, 230, 250], dtype=np.float64)
        s_raw = samples * contrast + brightness
        s_out = np.clip(s_raw, 0, 255).astype(np.uint8) if clip else wrap_uint8(s_raw)
        rows = [
            {"Original": int(o), "True math": f"{o:.0f} × {contrast:g} {brightness:+d} = {v:.1f}",
             "Stored as uint8": int(s), "Changed by": ("clipped" if clip else "wrapped") if (v > 255 or v < 0) else "—"}
            for o, v, s in zip(samples, s_raw, s_out)
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
        with st.expander("Show the OpenCV code for this result"):
            st.code(code, language="python")
        st.download_button("⬇️ Download adjusted PNG", png_bytes(result),
                           f"{base_name}_adjusted.png", "image/png")

# ------------------------------------------------------------------
# 3. Thresholding
# ------------------------------------------------------------------
with tab_thresh:
    st.subheader("Thresholding: global, inverse, adaptive and Otsu")
    ctrl, view = st.columns([1, 3])

    with ctrl:
        method = st.selectbox(
            "Method",
            ["Global (THRESH_BINARY)", "Inverse (THRESH_BINARY_INV)",
             "Adaptive — mean", "Adaptive — gaussian", "Otsu (automatic)"],
        )
        is_adaptive = method.startswith("Adaptive")
        is_otsu = method.startswith("Otsu")
        thresh_val = st.slider("Threshold value", 0, 255, 127, disabled=is_adaptive or is_otsu)
        block = st.slider("Block size (odd)", 3, 101, 13, 2, disabled=not is_adaptive)
        c_val = st.slider("C (subtracted from local mean)", -30, 30, 7, disabled=not is_adaptive)
        invert = st.checkbox("Invert output", value=False, disabled=method.startswith(("Global", "Inverse")))
        blur = st.checkbox("Gaussian blur first (5×5)", value=False,
                           help="Smoothing removes noise and often helps Otsu and adaptive thresholds.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    src = cv2.GaussianBlur(gray, (5, 5), 0) if blur else gray
    src_name = "cv2.GaussianBlur(gray, (5, 5), 0)" if blur else "gray"
    bin_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    bin_name = "cv2.THRESH_BINARY_INV" if invert else "cv2.THRESH_BINARY"
    retval = None

    if method.startswith("Global"):
        retval, out = cv2.threshold(src, thresh_val, 255, cv2.THRESH_BINARY)
        code = f"retval, out = cv2.threshold({src_name}, {thresh_val}, 255, cv2.THRESH_BINARY)"
    elif method.startswith("Inverse"):
        retval, out = cv2.threshold(src, thresh_val, 255, cv2.THRESH_BINARY_INV)
        code = f"retval, out = cv2.threshold({src_name}, {thresh_val}, 255, cv2.THRESH_BINARY_INV)"
    elif is_adaptive:
        amode = cv2.ADAPTIVE_THRESH_MEAN_C if "mean" in method else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        aname = "cv2.ADAPTIVE_THRESH_MEAN_C" if "mean" in method else "cv2.ADAPTIVE_THRESH_GAUSSIAN_C"
        out = cv2.adaptiveThreshold(src, 255, amode, bin_type, block, c_val)
        code = f"out = cv2.adaptiveThreshold({src_name}, 255, {aname}, {bin_name}, {block}, {c_val})"
    else:
        retval, out = cv2.threshold(src, 0, 255, bin_type + cv2.THRESH_OTSU)
        code = f"retval, out = cv2.threshold({src_name}, 0, 255, {bin_name} + cv2.THRESH_OTSU)"

    with view:
        before_after(image, out, after_caption=f"After ({method})")
        if is_otsu:
            st.success(f"Otsu picked a threshold of **{retval:.0f}** for this image.")
        elif retval is not None:
            st.caption(f"retval = {retval:.0f}. Pixels above it become 255 (white); everything else becomes 0.")
        else:
            st.caption("Adaptive thresholding computes a separate cutoff for every neighborhood, "
                       "so it copes with shadows and uneven lighting that defeat a single global value.")
        white = float(np.mean(out == 255) * 100)
        st.metric("White pixels", f"{white:.1f}%")
        with st.expander("Show the OpenCV code for this result"):
            st.code(f"gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n{code}", language="python")
        st.download_button("⬇️ Download thresholded PNG", png_bytes(out),
                           f"{base_name}_threshold.png", "image/png")

# ------------------------------------------------------------------
# 4. Background Remover (alpha channel + masks)
# ------------------------------------------------------------------
with tab_bg:
    st.subheader("Background remover: masks, bitwise_not and alpha channels")
    ctrl, view = st.columns([1, 3])

    with ctrl:
        bg_kind = st.radio("My background is…", ["Light (white sky, paper, studio)", "Dark"])
        mask_method = st.selectbox("Mask method", ["Global threshold", "Otsu (automatic)"])
        bg_thresh = st.slider("Threshold", 0, 255, 200, disabled=mask_method.startswith("Otsu"))
        flip = st.checkbox("Flip the mask (cv2.bitwise_not)", value=False,
                           help="Use this if the subject disappeared and the background stayed.")
        cleanup = st.slider("Clean-up (morphological open/close kernel)", 0, 15, 3,
                            help="0 = off. Removes speckles and fills small holes in the mask.")
        feather = st.slider("Soften edges (blur radius)", 0, 15, 0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mask_method.startswith("Otsu"):
        used_t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_code = "used_t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)"
    else:
        used_t, mask = cv2.threshold(gray, bg_thresh, 255, cv2.THRESH_BINARY)
        thresh_code = f"used_t, mask = cv2.threshold(gray, {bg_thresh}, 255, cv2.THRESH_BINARY)"

    # After thresholding, bright pixels are 255. For a light background the BACKGROUND is
    # white, so invert it: we want the SUBJECT to be 255 (opaque) in the alpha mask.
    code_lines = ["gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)", thresh_code]
    should_invert = bg_kind.startswith("Light")
    if flip:
        should_invert = not should_invert
    if should_invert:
        mask = cv2.bitwise_not(mask)
        code_lines.append("mask = cv2.bitwise_not(mask)   # subject -> 255 (opaque), background -> 0 (transparent)")

    if cleanup > 0:
        k = np.ones((cleanup, cleanup), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        code_lines.append(f"k = np.ones(({cleanup}, {cleanup}), np.uint8)")
        code_lines.append("mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)")
        code_lines.append("mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)")
    if feather > 0:
        ks = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (ks, ks), 0)
        code_lines.append(f"mask = cv2.GaussianBlur(mask, ({ks}, {ks}), 0)   # soft edges")

    b, g, r = cv2.split(image)
    bgra = cv2.merge([b, g, r, mask])
    code_lines.append("b, g, r = cv2.split(img)")
    code_lines.append("bgra = cv2.merge([b, g, r, mask])   # 4th channel = alpha")
    code_lines.append("cv2.imwrite('no_background.png', bgra)   # PNG keeps transparency")

    with view:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(to_rgb(image), caption="Before (original)", width="stretch")
        with c2:
            st.image(mask, caption=f"Alpha mask (threshold {used_t:.0f}): white = keep",
                     width="stretch")
        with c3:
            st.image(to_rgb(composite_on_checker(bgra)),
                     caption="After (checkerboard = transparent)", width="stretch")
        kept = float(np.mean(mask > 127) * 100)
        st.metric("Pixels kept (opaque)", f"{kept:.1f}%")
        with st.expander("Show the OpenCV code for this result"):
            st.code("\n".join(code_lines), language="python")
        st.download_button("⬇️ Download transparent PNG", png_bytes(bgra),
                           f"{base_name}_no_background.png", "image/png", type="primary")
        st.caption("Works best on photos with a plain, evenly lit background (e.g. a product on white).")

st.divider()
st.caption("Built with Streamlit + OpenCV. Your image is processed in memory and is not stored.")
