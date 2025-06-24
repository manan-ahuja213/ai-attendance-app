
import streamlit as st
import pandas as pd
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

def login():
    users_df = pd.read_csv("users.csv")
    st.title("🔐 School Login Portal")
    school_code = st.text_input("Enter School Code")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        user = users_df[
            (users_df["username"] == username) & 
            (users_df["password"] == password) & 
            (users_df["school_code"] == school_code)
        ]
        if not user.empty:
            role = user.iloc[0]["role"]
            st.success(f"Welcome, {username} ({role})")
            return role, school_code, username
        else:
            st.error("Invalid credentials")
            return None, None, None
    return None, None, None

def mark_attendance(name, school_code, marked_by):
    file = "attendance.csv"
    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    entry = pd.DataFrame([[name, time, school_code, marked_by]], columns=["Name", "Time", "SchoolCode", "MarkedBy"])
    if os.path.exists(file):
        df = pd.read_csv(file)
        df = pd.concat([df, entry], ignore_index=True)
    else:
        df = entry
    df.to_csv(file, index=False)

def run_face_recognition(school_code, marked_by):
    path = 'dataset'
    images = []
    names = []
    for img_name in os.listdir(path):
        img = cv2.imread(f'{path}/{img_name}')
        images.append(img)
        names.append(os.path.splitext(img_name)[0])

    def encode_faces(images):
        encodings = []
        for img in images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings.append(face_recognition.face_encodings(img_rgb)[0])
        return encodings

    known_encodings = encode_faces(images)

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        faces_current = face_recognition.face_locations(rgb_small)
        encodings_current = face_recognition.face_encodings(rgb_small, faces_current)

        for encoding, faceLoc in zip(encodings_current, faces_current):
            matches = face_recognition.compare_faces(known_encodings, encoding)
            face_dist = face_recognition.face_distance(known_encodings, encoding)
            matchIndex = np.argmin(face_dist)

            if matches[matchIndex]:
                name = names[matchIndex].upper()
                mark_attendance(name, school_code, marked_by)
                y1, x2, y2, x1 = [v*4 for v in faceLoc]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def show_teacher_dashboard(school_code, teacher):
    st.header(f"📸 Welcome Teacher {teacher}")
    if st.button("Start Attendance"):
        run_face_recognition(school_code, teacher)
    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        df = df[df["SchoolCode"] == school_code]
        st.dataframe(df[df["MarkedBy"] == teacher])

def show_admin_dashboard(school_code):
    st.header("📊 Admin Dashboard")
    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        df = df[df["SchoolCode"] == school_code]
        st.dataframe(df)
        st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name="school_attendance.csv")

role, school_code, username = login()

if role == "teacher":
    show_teacher_dashboard(school_code, username)
elif role == "admin":
    show_admin_dashboard(school_code)
