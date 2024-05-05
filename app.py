import base64
from pickle import load

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from keras.models import model_from_json
from keras.optimizers import Adam

st.set_page_config(page_title="Student Grade Prediction System", page_icon="logo.png", layout="wide")


# ------Read The excel File-----#

@st.cache_data
def get_data_from_excel(url):
    df = pd.read_excel(
        io=url,
        engine="openpyxl",
    )

    return df


url_pivot_table = "pivot_table.xlsx"
url_pivot_sem = "pivot_table_HK.xlsx"
fact_table = "dataset_score.xlsx"
pivot_result = get_data_from_excel(url_pivot_table)
pivot_result_HK = get_data_from_excel(url_pivot_sem)
df_fact_table = get_data_from_excel(fact_table)
pivot_result.set_index('MaSV')
pivot_result_HK.set_index('MaSV')
features = 10
maxSampleClass = int(max(pivot_result[' Algorithms & Data Structures'].value_counts()))
timeSteps = 11

# load the scaler
scaler = load(open('scaler.pkl', 'rb'))
# load the model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
# -----SideBar------#

with open("logo.png", "rb") as f:
    data = base64.b64encode(f.read()).decode("utf-8")

    st.sidebar.markdown(
        f"""
        <div style="display:table;margin-top:-20%;margin-left:10%;">
            <img src="data:image/png;base64,{data}">
        </div>
        """,
        unsafe_allow_html=True,
    )
st.sidebar.header("Please Filter Here")


def createRNNGradeMatrix(pivot_result_HK, pivot_result):
    semesterDict = pivot_result_HK.to_dict(orient='index')
    gradeDict = pivot_result.to_dict(orient='index')
    dataX = []
    maxSemester = 10
    for id, value in semesterDict.items():
        temp_List = []
        value.pop(' Algorithms & Data Structures')
        temp_dict = gradeDict.get(id)
        del temp_dict[' Algorithms & Data Structures']
        for k in range(1, maxSemester + 1):
            temporal_dict = temp_dict
            key_list = [key for key, val in value.items() if int(val) == k]
            semesterGradeList = {key: temp_dict[key] for key in key_list}
            temporal_dict = {x: 0.1 for x in temp_dict}
            temporal_dict.update(semesterGradeList)
            grade_list = temporal_dict.values()
            temp_List.append(list(grade_list))
        dataX.append(np.array(temp_List).T)
    return dataX
def predict_result(studentId):
    student_result = pivot_result[pivot_result.MaSV == studentId]
    student_sem = pivot_result_HK[pivot_result_HK.MaSV == studentId]
    student_data = createRNNGradeMatrix(student_sem.set_index('MaSV'), student_result.set_index('MaSV'))
    student_data = np.array(student_data)
    student_data = np.reshape(student_data, (-1, features))
    student_data = scaler.transform(student_data)
    student_data = np.reshape(student_data, (-1, timeSteps, features))
    print(student_data)
    y_pred = model.predict(student_data)
    print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)

    result = ""
    if y_pred[0] == 0:
        result = "Fail under 50"
    elif y_pred[0] == 1:
        result = "Average From 50 to 80"
    else:
        result = "Good greater than 80"

    return result


student = st.sidebar.text_input("Input the student ID")
predict_button = st.sidebar.button("Predict the Grade of DSA")
course = st.sidebar.multiselect(
    "Select courses (Can select multiple)",
    options=df_fact_table['TenMH'].unique(),
    default=df_fact_table['TenMH'].unique()

)
semester = st.sidebar.multiselect(
    "Select the semester",
    options=df_fact_table['HocKyThu'].unique(),
    default=df_fact_table['HocKyThu'].unique()
)
student_input = False
if len(student) > 0:
    df_selection = df_fact_table.query(
        "MaSV == @student & TenMH==@course & HocKyThu==@semester")
    student_input = True
else:
    df_selection = df_fact_table.query(
        "TenMH==@course & HocKyThu==@semester")

# Check if the dataframe is empty:
if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop()

# ---Main Page---#
st.title(" Student Grade Prediction")
st.markdown("##")

total_courses = df_fact_table['MaMH'].unique().size
total_department = df_fact_table['TenKhoa'].unique().size
total_student = df_fact_table['MaSV'].unique().size

left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Total Courses:")
    st.subheader(f"{total_courses:,}")
with middle_column:
    st.subheader("Total Students:")
    st.subheader(f"{total_student}")
with right_column:
    st.subheader("Total Departments:")
    st.subheader(f"{total_department}")

st.markdown("""---""")

# ----Bar chart---#
left_column, right_column = st.columns(2)
if student_input:
    student_grade = df_selection.sort_values(by="DiemHP", ascending=True).head(8)
    student_grade_px = px.bar(
        student_grade,
        x="DiemHP",
        y="TenMH",
        orientation="h",
        title="<b>Student Academic Grade</b>",
        color_discrete_sequence=["#0083B8"] * len(student_grade),
        template="plotly_white",
    )
    student_grade_px.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )
    student_gpa = df_selection[['HocKyThu', 'DTBHKH4']].drop_duplicates().sort_values(by="HocKyThu", ascending=True)
    print(student_gpa)
    student_GPA_px = px.line(
        student_gpa,
        x="HocKyThu",
        y="DTBHKH4",
        title="<b>Student Academic Grade</b>",
        template="plotly_white",
    )
    student_GPA_px.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
    )
    left_column.plotly_chart(student_grade_px, use_container_width=True)
    right_column.plotly_chart(student_GPA_px, use_container_width=True)
else:
    df_selection["DiemHP"] = df_selection["DiemHP"].replace(to_replace='VT', value=0.0)
    course_mean_grade = df_selection.groupby(by=["TenMH"])[["DiemHP"]].mean().sort_values(by="DiemHP",
                                                                                          ascending=False).head(10)
    course_mean_px = px.bar(
        course_mean_grade,
        x="DiemHP",
        y=course_mean_grade.index,
        orientation="h",
        title="<b>Student Academic Grade</b>",
        color_discrete_sequence=["#0083B8"] * len(course_mean_grade),
        template="plotly_white",
    )
    course_mean_px.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )
    df_selection['DiemChuHP'] = df_selection['DiemChuHP'].str.replace('+', '')
    total_student_of_grade = df_selection.groupby(by=["DiemChuHP"])['DiemChuHP'].count()
    total_student_px = px.bar(
        total_student_of_grade,
        x=total_student_of_grade.index,
        y="DiemChuHP",
        title="<b>Student Academic Grade</b>",
        color_discrete_sequence=["#0083B8"] * len(total_student_of_grade),
        template="plotly_white",
    )
    total_student_px.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )
    left_column.plotly_chart(course_mean_px, use_container_width=True)
    right_column.plotly_chart(total_student_px, use_container_width=True)

if student_input and predict_button:
    predictResult = predict_result(student)
    st.success(predictResult)
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)





