import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space

st.markdown(st.session_state.ReducePadding, unsafe_allow_html=True)

def set_question(question_short: str, question_index: int) -> None:
    """
    Assigns the value of the 'Question' and 'Question_Index' to session state

    Args:
        question_short (str): The selected question.
        question_index (int): The index of the selected question.

    Returns:
        None
    """
    st.session_state.Question = question_short
    st.session_state.Question_Index = question_index

# Define a list of available question options.
questions_dict = {
    'How does our current molecular testing capacity compare to testing demand and need?' : 'capacity - need',
    'Should we invest in new machine(s) or sample transport to underutilized machines? / Should we buy or rent diagnostic machines?' : 'machine vs. transport'
    }

# If the 'Question_Index' session state variable doesn't exist, initialize it to 0
if 'Question_Index' not in st.session_state:
    st.session_state.Question_Index = 0

# Display the radio button group to select the question
with st.container():
    st.header('Question to Model')
    
    question = st.radio('', 
        options=questions_dict.keys(),
        index=st.session_state.Question_Index
        ) 

    question_index = list(questions_dict.keys()).index(question)
    question_short = questions_dict[question]

with st.container():
    add_vertical_space(1)
    col1, col2, _ = st.columns(st.session_state.NavButtonCols)

    with col1:
        if st.button('Back', use_container_width=True):
            switch_page('baseline data')

    with col2:
        next_page = st.button('Continue', on_click=lambda: set_question(question_short, question_index), use_container_width=True)
        if next_page:
            if question_short != 'capacity - need':
                switch_page('select disease')
            else:
                switch_page('verify inputs')
