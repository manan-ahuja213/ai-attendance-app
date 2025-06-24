import streamlit as st
import pandas as pd
import face_recognition
import numpy as np
import os
from datetime import datetime
from PIL import Image

# --- CONFIG ---
st.set_page_config(page_title="AI Attendance System", layout="centered")

# --- LOGIN SYSTEM ---
def login():
    users = pd.read_csv("users.csv")
    st.title("🔐 School Login")
    school_code = st.text_input("School Code")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        match = users[
            (users["username"] == username) &
            (users["password"] == password) &
            (users["school_code"] == school_code)
        ]
        if not match.empty:
            role = match.iloc[0]["role"]
            st.success(f"Welcome {username} ({role})")
            return username, school_code, role
        else:
            st.error("Invalid credentials")
    return None, None, None

# --- LOAD KNOWN FACES ---
def load_known_faces():
    known_encodings = []
    known_names = []
    for file in os.listdir("sample")
