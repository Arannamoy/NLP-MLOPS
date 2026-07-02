# streamlit run app.py --server.port 8080

import streamlit as st
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent))

def init_session_state():
    session_default={
        "image_dir":"path"
    }

    for key,value in session_default.items():
        if key not in st.session_state:
            st.session_state[key]=value


init_session_state()

st.set_page_config(page_title="YOLOv11 Search App",layout="wide")
st.title("CV Powered Application")
option=st.radio("Choose an option",("Process new images","Load existing metadata"),horizontal=True)

if option=="Process new images":
    with st.expander("Process new images",expanded=True):
        col1,col2=st.columns(2)
        with col1:
            image_dir=st.text_input("Image directory path",placeholder="path/to/image")
        with col2:
            model_path=st.text_input("Model weight path:","yolo11m.pt")
        if st.button("Start Inference"):
            if image_dir:
                try:
                    with st.spinner("Running object detection....."):
                        time.sleep(3)
                        st.success(f"Processed new images.")
                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")
            else:
                st.warning(f"Please enter an image directory path")
else:
    with st.expander("Load Existing Metadata",expanded=True):
        metadata_path=st.text_input("Metadata file path",placeholder="/path/to/metadata.json")
        if st.button("Load Metadata"):
            if metadata_path:
                try:
                    pass
                    with st.spinner("Loading metadata....."):
                        time.sleep(3)
                        st.success(f"Successfully loaded metadata")
                except Exception as e:
                    st.error(f"Error loading metadata: {str(e)}")

            else:
                st.warning(f"Please enter a metadata directory path")