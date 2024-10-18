import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
import pandas as pd
import numpy as np
from typing import List
import functions as f


st.markdown(st.session_state.ReducePadding, unsafe_allow_html=True)

@st.cache_data(max_entries=1)
def load_baseline_data(target_country: str, regions_selected: List[str]) -> None:
    """
    Loads baseline data for the selected target country and subnational regions and assigns data to session state.

    Args:
        target_country (str): The selected target country.
        regions_selected (list): The selected subnational regions.

    Returns:
        None
    """
    df_input = st.session_state["MachinesByRegion_Baseline"].copy()
    suffixes = ["_NumMachinesTotal", "_NumMachinesRented", "_TestsPerMachinePerYear_"]

    df = pd.DataFrame(index=regions_selected)
    df.index.rename("admin_name", inplace=True)

    dno_countries = list(df_input["iso3"].unique())

    disease_list = ["HIV", "TB", "Other"]
    if target_country in dno_countries:
        df_input = df_input.loc[
            (df_input["iso3"] == target_country)
            & (df_input["admin_name"].isin(regions_selected))
        ]
        machine_list = list(df_input["Machine"].unique())
        for machine in machine_list:
            for suffix in suffixes:
                if suffix == "_TestsPerMachinePerYear_":
                    for disease in disease_list:
                        df[machine + suffix + disease] = [np.nan] * len(regions_selected)
                else:
                    df[machine + suffix] = [np.nan] * len(regions_selected)

        for _, row in df_input.iterrows():
            machine = row["Machine"]
            col_type = row["ColType"]
            val = row["Val"]
            admin_name = row["admin_name"]
            df.at[admin_name, f"{machine}_{col_type}"] = val

    else:
        df["HOLD"] = 0

    df.reset_index(inplace=True)

    st.session_state.BaselineData = df
    st.session_state.DiseaseList =  disease_list

@st.cache_data
def convert_df(df: pd.DataFrame) -> bytes:
    """
    Convert a pandas DataFrame to a CSV and then encode it to bytes.

    Args:
        df (pandas.DataFrame): A pandas DataFrame to be converted.

    Returns:
        bytes: The encoded CSV byte string.
    """
    return df.to_csv().encode('utf-8')

if 'BaselineData' not in st.session_state:
    st.session_state.BaselineData = pd.DataFrame()

if 'DiseaseList' not in st.session_state:
    st.session_state.DiseaseList = []

if 'BaselineData_Edit' not in st.session_state:
    st.session_state.BaselineData_Edit = pd.DataFrame()

if "SaveClicked" not in st.session_state:
    st.session_state.SaveClicked = False

IGNORE_COLS = ['iso3', 'admin_name', 'HOLD']
load_baseline_data(st.session_state.TargetCountry, st.session_state.RegionsSelected)  

with st.container():
    st.header('Baseline Data')
    st.write("Use the table below to enter the number of machines and tests per machine per year for each subnational region. \
             To add columns for a machine type, select the machine type from the dropdown menu and click 'Add Machine'. \
             To remove columns for a machine type, select the machine type from the dropdown menu and click 'Remove Machine'.")
    st.write("If you have a file with this information from a previous run of the model, you can upload it using the 'Upload' section below. \
            After completing the table, please download the data for later use using the 'Download' section.")

with st.container():
    # Section for uploading and downloading the baseline data table
    col1, col2 = st.columns(2)

    with col1:
        with st.expander('Upload'):
            uploaded_file = st.file_uploader('Only .csv files are accepted', type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)

                exclude_cols = ['admin_name']
                for col in df.columns:
                    if col not in exclude_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(np.nan).astype('float64')

                try:
                    df.drop(columns=['Unnamed: 0'], inplace=True)
                except:
                    pass
                st.session_state.BaselineData = df 
 
    with col2:
        with st.expander('Download'):
            st.caption('Please click Save before downloading.')
            if st.button('Save', use_container_width=True):
                st.session_state.BaselineData = st.session_state.BaselineData_Edit
                st.session_state.SaveClicked = True
            
            st.download_button(
                label='Download as CSV', 
                data=convert_df(st.session_state.BaselineData),
                file_name=f'{st.session_state.TargetCountry}_BaselineData.csv',
                mime='text/csv',
                use_container_width=True)
    
with st.container():
    # Create dropdown menu for machine type
    if 'SelectedMachine' not in st.session_state:
        st.session_state.SelectedMachine = ''

    if 'MachineType_Master' not in st.session_state:
        MachineType_InTable = set(i.split('_')[0] for i in list(st.session_state.BaselineData.columns) if i != 'admin_name')
        MachineType_Master = sorted(set(st.session_state.MachineTypes.index) | MachineType_InTable - {'admin_name'})
        st.session_state.MachineType_Master = MachineType_Master

    selected_machine = st.selectbox('',options=st.session_state.MachineType_Master)                                 

    st.session_state.SelectedMachine = selected_machine
    
    # Display buttons to add or remove machine types. Use callbacks to modify session state for BaselineData
    SUFFIXES = ['_NumMachinesTotal', '_NumMachinesRented', '_TestsPerMachinePerYear_HIV', '_TestsPerMachinePerYear_TB', '_TestsPerMachinePerYear_Other']

    col1, col2, _ = st.columns(st.session_state.NavButtonCols)
    
    with col1:
        st.button(
            'Add Machine', 
            on_click=f.add_machine_type, 
            args=(st.session_state.SelectedMachine, IGNORE_COLS, SUFFIXES, 'BaselineData', 'baseline'),
            use_container_width=True)
        
    with col2:
        st.button(
            'Remove Machine', 
            on_click=f.remove_machine_type, 
            args=(st.session_state.SelectedMachine, IGNORE_COLS, SUFFIXES, 'BaselineData', 'baseline'),
            use_container_width=True)

with st.container():
    grid_options = f.get_grid_options_baseline(st.session_state.BaselineData, IGNORE_COLS)
    grid_response = f.get_grid_response(st.session_state.BaselineData, grid_options)
    st.session_state.BaselineData_Edit = pd.DataFrame(grid_response['data'])

with st.container():
    add_vertical_space(1)
    col1, col2, _ = st.columns(st.session_state.NavButtonCols)

    with col1:
        if st.button('Back', use_container_width=True):
            switch_page('select regions')

    with col2:
        if st.button('Continue', use_container_width=True):
            st.session_state.BaselineData = st.session_state.BaselineData_Edit
            st.session_state.DiseaseList = list(set(i.split('_')[2] for i in list(st.session_state.BaselineData_Edit.columns) if 'TestsPerMachinePerYear' in i))

            switch_page('select question')