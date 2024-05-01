import streamlit as st
from PythonFiles.newUser import count_folders, TrainNewUsers
import time
from contextlib import contextmanager, redirect_stdout
from io import StringIO
import shutil
st.title("Train Model for new user(s)")
import re
import sys

ansi_escape = re.compile(r'\x1b\[(\d{1,2}(;\d{1,2})*)?m')

@contextmanager
def st_capture(src,output_func):
    with StringIO() as buffer:
        old_write = src.write
        placeholder = st.text("")
        def new_write(string):
            ret = old_write(string)
            buffer.write(string)
            output_with_colors = ansi_escape.sub(convert_ansi_to_html, f"{string}\n")
            output_func.write(f"{output_with_colors}", unsafe_allow_html=True)
            # output_func(stdout.getvalue())
            return ret
        
        src.write = new_write
        
        try:
            yield
        finally:
            src.write = old_write


def convert_ansi_to_html(match):
    codes = match.group(1).split('m')
    style = ''
    output = ''
    for code in codes:
        if code == '0':
            style += 'color: inherit;'
        elif code == '1':
            style += 'font-weight: bold;'
        elif code == '31':
            style += 'color: red;'
        elif code == '32':
            style += 'color: green;'
        elif code == '37':
            style += 'color: black;'
        # if code not in [ '┅', '┉', '┍', '┕', '┝', '┥', '┥', '┝','']:
        #     output += code
        # Add more color codes as needed
    return f'<span style="{style}">'

# @contextmanager
# def st_capture(output_func):
#     with StringIO() as stdout, redirect_stdout(stdout):
#         yield stdout
#         output_func(ansi_escape.sub('', stdout.getvalue()))

def disable(a,b):
    st.session_state['a_disabled'] = a
    st.session_state['b_disabled'] = b

folder_count = count_folders('PythonFiles/newUser')

if(folder_count == 0):
    disable(True,False)
    st.error("Folder is empty please register first")
# else:
#     disable(False,True)

placeholder = st.empty()

startTraining = st.button("Start training",key="a",on_click=disable,args=(True,False,),disabled=st.session_state.get("a_disabled",False))

if(folder_count !=0):
    stopTraining = st.button("Stop training",key="b",on_click=disable,args=(False,True,),disabled=st.session_state.get("b_disabled",True))

trainPlaceholder = st.empty()

if startTraining and folder_count!=0:
    trainPlaceholder.write("![Your Awsome GIF](https://media2.giphy.com/media/3oEjI6SIIHBdRxXI40/200w.gif?cid=6c09b952yx63atue9btnehyn9cu2z2g2o7xcmky4a8i7bjb0&ep=v1_gifs_search&rid=200w.gif&ct=gf)")
    
    # TrainNewUsers()
    with st_capture(sys.stdout,trainPlaceholder):
        TrainNewUsers()

    # output = TrainNewUsers()
    # st.code(output)

    # time.sleep(2)
    # shutil.rmtree(f"PythonFiles/newUser/test")
    trainPlaceholder.success("Training Successfully Completed!")
    disable(False,True)
        
   