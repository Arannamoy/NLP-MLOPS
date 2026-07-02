import streamlit as st

st.title("Streamlit intro")
st.write("This is just warm up for this project")
name=st.text_input("hello")
if name:
    st.write(f"Welcome, {name}!")

age=st.slider(f"",0,100)

st.write(f"Your age {age}")
gender=st.radio("Gender",("Male","Female"))
tech=st.multiselect("Tech:",("Spring","Django",".Net","Spring Boot","Spring Reactive","DRF",
                             "Tensorflow","Pytorch"))
interest=st.multiselect("Interest",("Microservices","Fintech","AI","ML"))

@st.dialog("Profile")
def open_modal():
    st.write(f"Your name: {name}")
    st.write(f"Your age: {age}")
    st.write(f"Gender: {gender}")
    st.write(f"Interest: {interest}")
if st.button("Create Profile"):
    open_modal()


