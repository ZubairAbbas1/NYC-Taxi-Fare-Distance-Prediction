import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Students Marks Analysis")
uploaded_file = st.file_uploader("Upload ur following file ", type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Marks preview")
    st.markdown("---")
    st.write(data.head())

    st.subheader("Summary of the Marks")
    st.markdown("---")
    st.write(data.describe())

    st.subheader("Choose Gender")
    st.markdown("---")
    selected_gender = st.selectbox("F / M", data['Gender'].unique())
    if selected_gender == 'Female':
        data.loc[data['Gender'] == 'Female', ['Name', 'Age', 'Section', 'English', 'Science', 'Maths', 'History']]
    else:
        data.loc[data['Gender'] == 'Male', ['Name', 'Age', 'Section', 'English', 'Science', 'Maths', 'History']]

    st.subheader("Highest Marks in subjects ")
    st.markdown("---")
    selected_subject = st.selectbox("Select Subject",["History",'Maths','Science', "English"])
    if selected_subject=='History':
        highest_h=data.sort_values(['History'],  ascending=False).head(15)
        st.write(highest_h[['Name','id','History']])
        st.subheader("Graph ")
        fig,ax=plt.subplots()
        ax.bar(data['id'] , data['History'] , color='skyblue')
        ax.set_xlabel("ID of students")
        ax.set_ylabel("Marks")
        ax.set_title("Marks of students in History")
        st.pyplot(fig)
    elif selected_subject=='Maths':
        highest_h = data.sort_values(['Maths'], ascending=False).head(15)
        st.write(highest_h[['Name', 'id', 'Maths']])
        st.subheader("Graph ")
        fig, ax = plt.subplots()
        ax.bar(data['id'], data['Maths'], color='red')
        ax.set_xlabel("ID of students")
        ax.set_ylabel("Marks")
        ax.set_title("Marks of students in Maths")
        st.pyplot(fig)
    elif selected_subject=='Science':
        highest_h = data.sort_values(['Science'], ascending=False).head(15)
        st.write(highest_h[['Name', 'id', 'Science']])
        st.subheader("Graph ")
        fig, ax = plt.subplots()
        ax.bar(data['id'], data['Science'], color='skyblue')
        ax.set_xlabel("ID of students")
        ax.set_ylabel("Marks")
        ax.set_title("Marks of students in Science")
        st.pyplot(fig)
    else:
        highest_h = data.sort_values(['English'], ascending=False).head(15)
        st.write(highest_h[['Name', 'id', 'English']])
        st.subheader("Graph ")
        fig, ax = plt.subplots()
        ax.bar(data['id'], data['English'], color='skyblue')
        ax.set_xlabel("ID of students")
        ax.set_ylabel("Marks")
        ax.set_title("Marks of students in Science")
        st.pyplot(fig)
