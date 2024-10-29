import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
import pandas as pd
import os

# Hide the streamlit sidebar
st.set_page_config(page_title="Modeling Tradeoffs in Molecular Diagnostic Equipment Selection and Use", 
                initial_sidebar_state="collapsed",
                layout="wide")

st.cache_data.clear()

if 'NavButtonCols' not in st.session_state:
    st.session_state.NavButtonCols = [1,1,7]

if 'ReducePadding' not in st.session_state:
    st.session_state.ReducePadding = """
        <style>
               .block-container {
                    padding-top: 2.5rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """

with st.spinner("Loading..."):
    def read_input_data():
        """
        This function reads input data tables from an Excel workbook and
        assigns them to st.session_state if they are not already assigned.

        Returns:
            None
        """
        # Get the current directory path
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        
        # Required to remove 'NA' from the default NaN value list as this is the country code for Namibia
        na_values_new = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a', '<NA>', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', '']
        
        # Dictionary of data file names, sheet names, and index columns
        data_files = {
            "EligibleCountries": {"sheet_name": "EligibleCountries", "index_col": 0},
            "MachineTypes": {"sheet_name": "MachineTypes", "index_col": 0},
            "MachineTypesOriginal": {"sheet_name": "MachineTypes", "index_col": 0},
            "MachinesByRegion_Baseline": {"sheet_name": "MachinesByRegion_Baseline", "index_col": None},
            "DiseaseBurden": {"sheet_name": "DiseaseBurden", "index_col": None},
            "DNOInput": {"sheet_name": "DNOInput", "index_col": None},
            "CPI": {"sheet_name": "CPI", "index_col": 0},
            "Exchange": {"sheet_name": "Exchange", "index_col": 0},
            "DNOTransport1": {"sheet_name": "DNOTransport1", "index_col": None},
            "DNOTransport2": {"sheet_name": "DNOTransport2", "index_col": None},
            "StaffCost": {"sheet_name": "StaffCost", "index_col": [0, 1]},
            "DistanceTime": {"file_name": "GFGX_RegionPairs_Distance_Master.csv", "index_col": None}
        }

        # Loop through each data file and sheet, read it in, and assign it to st.session_state
        for data_file, settings in data_files.items():
            if data_file not in st.session_state:
                if "sheet_name" in settings:
                    data = pd.read_excel(
                        os.path.join(__location__, "GFGX_MainInputs.xlsx"), 
                        sheet_name=settings["sheet_name"],
                        header=0,
                        index_col=settings["index_col"],
                        keep_default_na=False,
                        na_values=na_values_new
                    )
                elif "file_name" in settings:
                    data = pd.read_csv(
                        os.path.join(__location__, settings["file_name"]),
                        header=0,
                        index_col=settings["index_col"]
                    )
                st.session_state[data_file] = data

    # Call the function to read the input data
    read_input_data()

    # Define default values for constants
    DISTANCE_THRESHOLD = 10000 #Maximum km that a sample can travel
    COST_PER_KM = 0.45 #Transport cost per km in USD
    WORK_DAYS_PER_YEAR = 250 #Number of days per year that molecular diagnostic machines are intended to operate (ignoring downtime)
    TRANSPORT_WEEKS_PER_YEAR = 50 #Number of weeks per year that sample transport routes are intended to operate
    TRANSPORT_FREQ_PER_WEEK = 1 #Number of times per week that samples are transported along each route 
    ANALYSIS_YEAR = 2022 #Year of analysis, for costs
    ROUND_TRIP = True #False if one-way distance used for transport cost calculation, True if round-trip distance used for transport cost calculation
    HIV_EID_TESTS_PER_EXPOSED_CHILD = 3 # Either 2 or 3 required https://apps.who.int/iris/bitstream/handle/10665/325961/9789241516211-eng.pdf
    HIV_VL_TESTS_PER_NEWLY_DIAGNOSED = 2 # Annual HIV VL tests per newly diagnosed person https://www.ncbi.nlm.nih.gov/books/NBK572730/figure/ch4.fig3/
    HIV_VL_TESTS_PER_PREVIOUSLY_DIAGNOSED = 1 # Annual HIV VL tests per person diagnosed more than one year ago
    RATE_DENOMINATOR = 100000 # Denominator for incidence, prevalence, and other rates used in the model.  Value per [RATE_DENOMINATOR] population

    constant_dict = {
        "DistanceThreshold": DISTANCE_THRESHOLD,
        "CostPerKm": COST_PER_KM,
        "WorkDaysPerYear": WORK_DAYS_PER_YEAR,
        "TransportWeeksPerYear": TRANSPORT_WEEKS_PER_YEAR,
        "TransportFreqPerWeek": TRANSPORT_FREQ_PER_WEEK,
        "AnalysisYear": ANALYSIS_YEAR,
        "RoundTrip": ROUND_TRIP,
        "HIV_EID_TestsPerExposedChild" : HIV_EID_TESTS_PER_EXPOSED_CHILD,
        "HIV_VL_TestsPerNewlyDiagnosed" : HIV_VL_TESTS_PER_NEWLY_DIAGNOSED,
        "HIV_VL_TestsPerPreviouslyDiagnosed" : HIV_VL_TESTS_PER_PREVIOUSLY_DIAGNOSED,
        "RateDenominator" : RATE_DENOMINATOR
        }

    # Assign the constants to st.session_state
    for key, value in constant_dict.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Fill missing values of rental costs
    rental_cols = ['Processing_Cost_PerSampleVariable_TB_Rented', 'Processing_Cost_PerSampleVariable_HIV_Rented']
    for col in rental_cols:
        if st.session_state.MachineTypes[col].sum() > 0:
            average = st.session_state.MachineTypes[col].mean()
            st.session_state.MachineTypes[col].fillna(average, inplace=True)
    
    st.session_state.MachineTypes['Processing_Cost_PerSampleVariable_TB_Rented'] = st.session_state.MachineTypes['Processing_Cost_PerSampleVariable_TB_Rented'].fillna(st.session_state.MachineTypes['Processing_Cost_PerSampleVariable_HIV_Rented'])

    # Determine the latest year with CPI data
    numeric_columns = st.session_state.CPI.select_dtypes(include=['float64', 'int']).columns
    max_year = max(numeric_columns)
    if 'MaxYearCPI' not in st.session_state:
        st.session_state.MaxYearCPI = max_year

    # Values set to true if items are tradeable (typically imported for use in a lab network)
    # -- Set to false if items are nontradeable (typically not imported for use in a lab network)
    tradeable_dict = {
        'Processing_Cost_AnnualFixed' : True,
        'Processing_Cost_PerSampleVariable_HIV': True,
        'Processing_Cost_PerSampleVariable_TB': True,
        'Processing_Cost_PerSampleVariable_HIV_Rented': True,
        'Processing_Cost_PerSampleVariable_TB_Rented': True,
        'AnnualSalaryUSD' : False,
        'CostPerKm' : False
        }

    if 'TradeableDict' not in st.session_state:
        st.session_state.TradeableDict = tradeable_dict

    # Display title
    st.markdown(st.session_state.ReducePadding, unsafe_allow_html=True)

    with st.container():
        _, col2, _ = st.columns([1,5,1])
        
        with col2:
            st.markdown("<h1 style='text-align: center; color: black;'>Modeling Tradeoffs in Molecular Diagnostic Equipment Selection and Use</h1>", unsafe_allow_html=True)

    with st.container():
        add_vertical_space(2)

        st.write("This model was developed by the William Davidson Institute at the University of Michigan in partnership with the Global Fund and FIND.  It can be used to answer three questions related to investments in HIV and TB molecular diagnostic (MolDx) networks.")
        st.markdown("""
            - **How does our current molecular testing capacity compare to testing demand and need?** The outputs from this modeling question provide context, highlighting mismatches between testing demand/need and testing capacity at the subnational level.
            - **Should we invest in new machine(s) or sample transport to underutilized machines?** The outputs from this modeling question summarize the effects of investing in machines or sample transport on annual cost of molecular testing, annual molecular tests performed, cost per test, percentage of testing need met, and average machine utilization.
            - **Should we buy or rent molecular diagnostic machines?** The outputs from this modeling question summarize the effects of purchasing versus renting machines on annual cost of molecular testing and cost per test.
            """)

    with st.container():
        add_vertical_space(1)
        col1, _, _ = st.columns(st.session_state.NavButtonCols)

        with col1:
            if st.button("Continue", use_container_width=True):
                switch_page('select country')

    with st.container():
        add_vertical_space(7)
        col1, _, col3, _, col5 = st.columns(5)

        # Get the current directory path
        __location__ = os.getcwd()

        # Display logos
        with col1:
            st.image( __location__+"/Logo_WDI.jpg", width=300,)
            
        with col3:
            st.image( __location__+"/Logo_GF.png", width=200)
            
        with col5:
            st.image( __location__+"/Logo_FIND.jpg", width=150)




        




    

    









