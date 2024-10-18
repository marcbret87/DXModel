import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
import numpy as np
from typing import List, Dict


st.markdown(st.session_state.ReducePadding, unsafe_allow_html=True)

@st.cache_data
def load_regions(target_country: str) -> List[str]:
    """
    Loads the subnational regions for a given target country.

    Args:
        target_country (str): The target country.

    Returns:
        list: A list of subnational regions for the target country.
    """
    df = st.session_state.DiseaseBurden.copy()
    regions = list(df.loc[df['iso3'] == target_country, 'admin_name'])
    regions.sort()
    return regions

def set_regions(regions_selected: List[str], val_dict: Dict[str, bool]) -> None:
    """
    Sets the selected subnational regions and their corresponding checkbox values
    in st.session_state.

    Args:
        regions_selected (list): The list of selected subnational regions.
        val_dict (dict): A dictionary with the subnational regions as keys and their
                         corresponding checkbox values as values.

    Returns:
        None
    """
    st.session_state.RegionsSelected = regions_selected
    st.session_state.ValSave = val_dict

def display_regions(col, group: List[str], val_dict: Dict[str, bool]) -> None:
    """
    Displays a group of subnational regions as checkboxes.

    Args:
        col (streamlit.Column): The streamlit column to display the checkboxes in.
        group (list): A list of subnational regions to display.
        val_dict (dict): A dictionary with the subnational regions as keys and their
                         corresponding checkbox values as values.

    Returns:
        None
    """
    for region in group:
        val = val_dict.get(region, True)
        if 'ValSave' in st.session_state:
            val = st.session_state.ValSave.get(region, True)
        val_dict[region] = col.checkbox(region, value=val, key=region)

# Load the subnational regions for the target country
regions = load_regions(st.session_state.TargetCountry)

# Divide the regions into three groups for display
group_size = int(np.ceil(len(regions)/3))
regions_1 = regions[0 : group_size]
regions_2 = regions[group_size : (group_size * 2)]
regions_3 = regions[(group_size * 2) : ]

# Initialize the list of selected regions and the dictionary of checkbox values
regions_selected = []
val_dict = {}

with st.container():
    st.header('Subnational Regions')
    st.write('Select subnational regions to include in the analysis.   \
             Regions can be included even if they do not currently have MolDx machines.')

    # Display the subnational regions as checkboxes
    col_1, col_2, col_3 = st.columns(3)
    display_regions(col_1, regions_1, val_dict)
    display_regions(col_2, regions_2, val_dict)
    display_regions(col_3, regions_3, val_dict)

# Update the list of selected regions when the user clicks the "Continue" button
with st.container():
    add_vertical_space(1)
    col1, col2, _ = st.columns(st.session_state.NavButtonCols)

    with col1:
        if st.button('Back', use_container_width=True):
            switch_page('select country')

    with col2:
        if st.button('Continue', use_container_width=True):
            for region, val in val_dict.items():
                if val:
                    regions_selected.append(region)
            set_regions(regions_selected, val_dict)
            switch_page('baseline data')
