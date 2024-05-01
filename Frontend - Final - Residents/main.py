import streamlit as st
from st_pages import Page, show_pages, add_page_title
from PythonFiles.LiveFaceRecognition import LiveFaceRecognition
from contextlib import contextmanager, redirect_stdout
from io import StringIO

def disable(b):
    st.session_state['disabled'] = b

def main():
    st.set_page_config(
        page_title="main page"
    )

    show_pages([
        Page("main.py","Main page"),
        Page("pages/Register.py","Register user"),
        Page("pages/Model.py","Train Model for new user(s)")
    ])
    # st.set_page_config(page_title="Main Page")

    st.title("Welcome to our project")
    
    
    col1,col2= st.columns([1,5],gap="small")
    with col1:
        startModel = st.button("Start recog",key="a", on_click=disable,args=(True,),disabled=st.session_state.get("disabled",False))
    with col2:
        stopModel = st.button("Stop the system",key="b", on_click=disable,args=(False,),disabled= not st.session_state.get("disabled",False))
    codePlaceholder= st.empty()
    placeholder = st.empty()

    if startModel:
        # startModel("Disabled", disabled=True)
        print("model started!")
        LiveFaceRecognition(placeholder = placeholder) 
        # with st_capture(codePlaceholder.code):
        #     LiveFaceRecognition(placeholder = placeholder)
        
    
    with placeholder.container():   
        st.image("Aryan.jpg")
    
       
    
    
if __name__ == "__main__":
    main()