import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
import pandas as pd
from typing import List
import functions as f

st.markdown(st.session_state.ReducePadding, unsafe_allow_html=True)

@st.cache_data(max_entries=1)
# TargetCountry and RegionsSelected passed to override cache if new country or region set input for baseline data
def LoadAddMachinesData(TableName: str, IGNORE_COLS: List[str], TargetCountry: str, RegionsSelected: List[str]) -> None:
    """
    Load and modify a specified table from the session state, removing unwanted columns.
    The modified table is then saved back into the session state under 'AddMachinesData'.

    Args:
        TableName (str): Name of the table in the session state.
        IGNORE_COLS (List[str]): List of column names to be ignored during the column removal process.
        TargetCountry (str): The target country, currently unused in this function.
        RegionsSelected (List[str]): The regions selected, currently unused in this function.

    Returns:
        None
    """
    df_target = st.session_state[TableName].copy()
    DropList = []
    for col in df_target.columns:
        if col not in IGNORE_COLS:
            col_type = col.split('_')[1]
            TargetList = ['NumMachinesTotal','NumMachinesRented']
            if col_type not in TargetList:
                DropList.append(col)

    df_target.drop(DropList, axis=1, inplace=True)
    st.session_state.AddMachinesData = df_target

if 'AddMachinesData' not in st.session_state:
    st.session_state.AddMachinesData = pd.DataFrame()

if 'AddMachinesData_Edit' not in st.session_state:
    st.session_state.AddMachinesData_Edit = pd.DataFrame()

IGNORE_COLS = ['iso3', 'admin_name']
LoadAddMachinesData('BaselineData', IGNORE_COLS, st.session_state.TargetCountry, st.session_state.RegionsSelected)

with st.container():
    st.header('Add Machines Scenario:')
    st.write("Use the table below to change the number of MolDx machines in subnational regions. \
             To add a machine type not present at baseline, select the machine type from the dropdown menu and click 'Add Machine'. \
             To remove a machine type, select the machine type from the dropdown menu and click 'Remove Machine'.")
    
with st.container():
    ### Create dropdown menu for machine type
    if 'SelectedMachine' not in st.session_state:
        st.session_state.SelectedMachine = ''

    MachineTypes_TargetDisease = list(st.session_state.MachineTypes.loc[st.session_state.MachineTypes['Usable_' + st.session_state.Disease] == 1].index)

    SelectedMachine = st.selectbox('', options=MachineTypes_TargetDisease) 

    st.session_state.SelectedMachine = SelectedMachine
    
    SUFFIXES = ['_NumMachinesTotal', '_NumMachinesRented']

    col1, col2, _ = st.columns(st.session_state.NavButtonCols)
    
    with col1:
        st.button(
            'Add Machine', 
            on_click=f.add_machine_type, 
            args=(st.session_state.SelectedMachine, IGNORE_COLS, SUFFIXES, 'AddMachinesData', 'add_machines'),
            use_container_width=True)
        
    with col2:
        st.button(
            'Remove Machine', 
            on_click=f.remove_machine_type, 
            args=(st.session_state.SelectedMachine, IGNORE_COLS, SUFFIXES, 'AddMachinesData', 'add_machines'),
            use_container_width=True)
    
with st.container():
    # Display editable table
    grid_options = f.get_grid_options_add_machines(st.session_state.AddMachinesData, IGNORE_COLS)
    grid_response = f.get_grid_response(st.session_state.AddMachinesData, grid_options)
    st.session_state.AddMachinesData_Edit = pd.DataFrame(grid_response["data"])

with st.container():
    add_vertical_space(1)
    col1, col2, _ = st.columns(st.session_state.NavButtonCols)

    with col1:
        if st.button('Back', use_container_width=True):
            switch_page('select disease')

    with col2:
        NextPage = st.button('Continue', use_container_width=True)
        if NextPage:
            st.session_state.AddMachinesData = st.session_state.AddMachinesData_Edit
            switch_page('verify inputs')
