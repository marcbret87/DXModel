import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space

st.markdown(st.session_state.ReducePadding, unsafe_allow_html=True)

def set_disease(disease: str, disease_index) -> None:
    st.session_state.Disease = disease
    st.session_state.Disease_Index = disease_index

DISEASE_OPTIONS = ['HIV', 'TB']

if 'Disease_Index' not in st.session_state:
    st.session_state.Disease_Index = 0

with st.container():
    st.header('Disease')

    disease = st.radio('Select the disease to model',
                        options=DISEASE_OPTIONS,
                        index=st.session_state.Disease_Index)

    disease_index = DISEASE_OPTIONS.index(disease)
    
with st.container():
    add_vertical_space(1)
    col1, col2, _ = st.columns(st.session_state.NavButtonCols)

    with col1:
        if st.button('Back', use_container_width=True):
            switch_page('select question')

    with col2:
        next_page = st.button('Continue', on_click=lambda: set_disease(disease, disease_index), use_container_width=True)
        if next_page:
            switch_page('add machines')
