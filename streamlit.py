import streamlit as st
import numpy as np
import cv2
import json
from annotate import annotate_wrinkle, overlay_annotations

st.title("Wrinkle Annotation Extractor")

if "img_result" not in st.session_state:
    st.session_state.img_result = None

if "annotations" not in st.session_state:
    st.session_state.annotations = None

if "json_str" not in st.session_state:
    st.session_state.json_str = ""

upload_bitmoji = st.file_uploader("Upload Image Bitmoji", type=["jpg", "png"])

if upload_bitmoji:
    file_bytes = np.asarray(bytearray(upload_bitmoji.read()), dtype=np.uint8)
    st.session_state.img_result = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if st.button("Annotate Wrinkles") and st.session_state.img_result is not None:
    with st.spinner("Processing..."):
        st.session_state.annotations = annotate_wrinkle(st.session_state.img_result)
        st.session_state.json_str = json.dumps(st.session_state.annotations, indent=4)
        st.success("Annotation Complete!")
        st.json(st.session_state.annotations)
        with open("wrinkle_annotations.json", "w") as f:
            f.write(st.session_state.json_str)

if st.session_state.json_str:
    st.download_button(
        label="Download Annotations JSON",
        data=st.session_state.json_str,
        file_name="wrinkle_annotations.json",
        mime="application/json"
    )

upload_origin = st.file_uploader("Upload Original Image", type=["jpg", "png"])

if upload_origin and st.session_state.annotations is not None:
    file_bytes2 = np.asarray(bytearray(upload_origin.read()), dtype=np.uint8)
    img_origin = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)
    overlayed_img = overlay_annotations(img_origin, st.session_state.annotations)

    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.img_result, caption="Bitmoji Annotated Source", channels="BGR")
    with col2:
        st.image(overlayed_img, caption="Overlay on Original Image", channels="BGR")
