# OpenCV Photo Lab

A Streamlit app built from the *Adding Vectors* notebook. Upload a photo and explore four OpenCV concepts with live controls and a before/after view:

| Tab | Concept | OpenCV calls |
|---|---|---|
| ✏️ Annotate | Drawing and text | `cv2.line`, `cv2.circle`, `cv2.rectangle`, `cv2.putText`, `cv2.getTextSize`, `LINE_8` vs `LINE_AA` |
| ☀️ Brightness & Contrast | Image arithmetic | `cv2.add`, `cv2.subtract`, `cv2.multiply`, clipping on/off (uint8 overflow) |
| ⚫ Thresholding | Segmentation | `cv2.threshold` (`THRESH_BINARY`, `THRESH_BINARY_INV`, `THRESH_OTSU`), `cv2.adaptiveThreshold` |
| ✂️ Background Remover | Alpha channels and masks | `cv2.bitwise_not`, `cv2.split`, `cv2.merge` → transparent PNG |

Each tab has a **"Show the OpenCV code"** panel that prints the exact call, with the current slider values, and a download button for the result.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud (public link, no login to view)

1. Create a **public** GitHub repository (e.g. `opencv-photo-lab`) and upload `app.py`, `requirements.txt` and this README to its root.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Pick the repository, branch `main`, and main file `app.py`. Optionally set a custom URL (e.g. `opencv-photo-lab.streamlit.app`). Under **Advanced settings**, choose Python 3.12.
5. Click **Deploy**. The first build takes a few minutes. After that, anyone can open the `*.streamlit.app` link without an account.

Note: free apps go to sleep after a period with no visitors. The next visitor sees a "wake up" button, and the app restarts in under a minute.
