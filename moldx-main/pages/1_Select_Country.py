import streamlit as st
import functions as f
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
import numpy as np
from typing import List

st.markdown(st.session_state.ReducePadding, unsafe_allow_html=True)
f.inject_custom_styling()
f.draw_stepper("Country")

@st.cache_data
def create_country_options()-> List[str]:
    """
    Creates a list of country names from the 'EligibleCountries' sheet and returns it.

    Returns:
        List of country names
    """
    Countries = sorted(list(st.session_state.EligibleCountries['name']))
    return Countries

def set_positivity_rate(target_country):
    """
    Defines the TB testing positivity rate for the target country and saves it in session state.
    This is the proportion of MolDx tests for TB that result in a positive diagnosis.

    Returns:
        None
    """
    TB_positivity_rate = st.session_state.EligibleCountries.at[target_country, 'TB_positivity_rate']
    if np.isnan(TB_positivity_rate):
        TB_positivity_rate = st.session_state.EligibleCountries['TB_positivity_rate'].mean()
    st.session_state.TB_PositivityRate = TB_positivity_rate

def set_target_country_and_year(TargetCountry_FullName: str, TargetCountry_Index: int, AnalysisYear: int, AnalysisYear_Index) -> None:
    """
    Sets the target country full name in session state and the analysis year.  Calls set_positivity_rate() because this value is defined at the country level. 

    Args:
        TargetCountry_FullName (str): Full name of the target country selected by the user.
        TargetCountry_Index (int): Index of the target country selected by the user.
        AnalysisYear (int): Year to display costs in.
        AnalysisYear_Index (int): Index of the year to display costs in.

    Returns:
        None
    """
    TargetCountry = st.session_state.EligibleCountries.index[st.session_state.EligibleCountries['name'] == TargetCountry_FullName][0]
    
    country_changed = ('TargetCountry' in st.session_state and st.session_state.TargetCountry != TargetCountry)
    year_changed = ('AnalysisYear' in st.session_state and st.session_state.AnalysisYear != AnalysisYear)

    if country_changed or year_changed:
        if 'apply_inflation_machine_types_called' in st.session_state:
            st.session_state.apply_inflation_machine_types_called = False
        if 'apply_inflation_staff_called' in st.session_state:
            st.session_state.apply_inflation_staff_called = False
        if 'MachineTypesOriginal' in st.session_state:
            st.session_state.MachineTypes = st.session_state.MachineTypesOriginal.copy()

    if country_changed:
        if 'BaselineData' in st.session_state:
            del st.session_state['BaselineData']
        if 'BaselineData_Edit' in st.session_state:
            del st.session_state['BaselineData_Edit']
        if 'AddMachinesData' in st.session_state:
            del st.session_state['AddMachinesData']
        if 'AddMachinesData_Edit' in st.session_state:
            del st.session_state['AddMachinesData_Edit']
        if 'RegionsSelected' in st.session_state:
            del st.session_state['RegionsSelected']
        if 'ValSave' in st.session_state:
            del st.session_state['ValSave']

    st.session_state.TargetCountry = TargetCountry   
    st.session_state.TargetCountry_Index = TargetCountry_Index
    st.session_state.AnalysisYear = AnalysisYear 
    st.session_state.AnalysisYear_Index = AnalysisYear_Index 

    set_positivity_rate(TargetCountry)

# Create a list of country options
CountryOptions = create_country_options()

# Create a list of year options (e.g., 2020 USD, 2021 USD)
YearOptions = [2020, 2021, 2022]

if 'TargetCountry_Index' not in st.session_state:
    st.session_state.TargetCountry_Index = 0

if 'AnalysisYear_Index' not in st.session_state:
    st.session_state.AnalysisYear_Index = 2

# Display selectboxes for the user to choose the target country and the analysis year
with st.container():
    st.header('Target Country')
    TargetCountry_FullName = st.selectbox('Select the country for analysis', options=CountryOptions, index=st.session_state.TargetCountry_Index)
    TargetCountry_Index = CountryOptions.index(TargetCountry_FullName)

    add_vertical_space(1)
    AnalysisYear = st.selectbox('Display costs in [YEAR] USD', options=YearOptions, index=st.session_state.AnalysisYear_Index)
    AnalysisYear_Index = YearOptions.index(AnalysisYear)

with st.container():
    add_vertical_space(1)
    col1, col2, _ = st.columns(st.session_state.NavButtonCols)
    
    with col1:
        if st.button('Back', use_container_width=True):
            switch_page('intro')

    with col2:
        NextPage = st.button('Continue', on_click=lambda: set_target_country_and_year(TargetCountry_FullName, TargetCountry_Index, AnalysisYear, AnalysisYear_Index), use_container_width=True)
        if NextPage:
            switch_page('select regions')





