#stremlit app component for the upload function

import streamlit as st
from streamlit_ace import st_ace
from os import mkdir
from os.path import join, isdir
import glob
import os
from pathlib import Path

st.set_page_config(page_title= "Test converter", layout='wide')

uploader_expander = st.beta_expander("upload data")

with uploader_expander:
    uploaded = st.file_uploader("upload your zipped experiment")

if uploaded is not None:
    bytes_data = uploaded.read()
    with open("upload/uploaded.zip", "wb") as f:
        f.write(bytes_data)
    
    cmd = 'unzip uploaded/uploaded.zip -d upload'
    os.system(cmd)
    imgs = [x.name for x in Path('upload').rglob('*.tif')]

groups = [x[0] for x in os.walk('data')]

with uploader_expander:
    st.header('Experiment groups')
    col1, __, col2, col3 = st.beta_columns([6,1,3,3])
    with col1:
        groups_diplay = st_ace(
                    value = "\n".join(groups),
                    language = "plain_text",
                    theme = "iplastic",
                    font_size = font_size,
                    show_gutter = True,
                    show_print_margin = False,
                    wrap = True,
                    auto_update= True,
                    key="ace-editor",
                    readonly = True

     )

    with col2:
        group = st.text_input("new group name")
    with col3:
        st.text("  ")
        button = st.button("Create new group")

if button:
    if not os.path.exists(f"data/{button}"):
        os.makedirs(f"data/{button}")
    
        
