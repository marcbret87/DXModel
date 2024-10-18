import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
import pandas as pd
import numpy as np
import functions as f

st.markdown(st.session_state.ReducePadding, unsafe_allow_html=True)

with st.container():
    st.header('Verify Inputs')
    st.write("Please review the inputs below.  If you would like to make changes to an input, click on the cell, change the value, and then press Enter.  When you have finished making changes, click Continue.")

# Create list with unique country-disease combinations in the DNOTransport1 table
if 'TransportDataCountries' not in st.session_state:
    st.session_state.TransportDataCountries = st.session_state.DNOTransport1.groupby(['iso3', 'Disease']).size().reset_index().apply(lambda x: x['iso3'] + '_' + x['Disease'], axis=1).tolist()


## Display inputs
## --Numbers in variable names (e.g., "verify_4") have no significance other than unique identification.  Numbers were assigned based on order of development.

# Display testing inputs
verify_4 = {
    'Number of MolDx HIV EID Tests Per Child Exposed at Birth': 'HIV_EID_TestsPerExposedChild',
    'Number of HIV Viral Load Tests Per Year per for a Newly Identified Case': 'HIV_VL_TestsPerNewlyDiagnosed',
    'Number of HIV Viral Load Tests Per Year per for a Previously Identified Case': 'HIV_VL_TestsPerPreviouslyDiagnosed',
    'TB Positivity Rate: Positive MolDx TB Diagnoses as a Proportion of total MolDx TB Diagnostic Tests Performed': 'TB_PositivityRate',
    }

if st.session_state.Question != 'capacity - need':
    verify_4_series = pd.Series({key: st.session_state[value] for key, value in verify_4.items() if st.session_state.Disease in key})
else:
    verify_4_series = pd.Series({key: st.session_state[value] for key, value in verify_4.items()})
verify_4_df = verify_4_series.to_frame().reset_index()
verify_4_df.columns = ['Variable', 'Value']

grid_options = f.get_grid_options_verify_4()
grid_response = f.get_grid_response(verify_4_df, grid_options)
verify_4_edit = pd.DataFrame(grid_response['data'])

# Display disease burden inputs
col_info = [
    {'field' : 'Population', 'headerName' : 'Population', 'format' : 'zero_decimal', 'editable' : True},
    {'field' : 'HIV_Exposed_Births_Rate_National', 'headerName' : 'HIV-Exposed Births per 100,000 Population', 'format' : 'one_decimal', 'editable' : True},
    {'field' : 'Incidence_Rate_HIV', 'headerName' : 'HIV Incidence Rate per 100,000 Population (All Ages)', 'format' : 'one_decimal', 'editable' : True},
    {'field' : 'Prevalence_Rate_HIV', 'headerName' : 'HIV Prevalence Rate per 100,000 Population (All Ages)', 'format' : 'one_decimal', 'editable' : True},
    {'field' : 'Rate_HIV_DataLevel', 'headerName' : 'Level of HIV Source Data (National or Subnational)', 'format' : 'string', 'editable' : False},
    {'field' : 'Incidence_Rate_TB', 'headerName' : 'TB Incidence Rate per 100,000 Population (All Ages)', 'format' : 'one_decimal', 'editable' : True},
    {'field' : 'Rate_TB_DataLevel', 'headerName' : 'Level of TB Source Data (National or Subnational)', 'format' : 'string', 'editable' : False}]

if st.session_state.Question != 'capacity - need':
    KeepCols = [i['field'] for i in col_info if f'{st.session_state.Disease}' in i['field'] or 'Population' in i['field']]
else:
    KeepCols = [i['field'] for i in col_info]

KeepCols.extend(['admin_name', 'EstimatedHealthFacilities'])

col_info_filtered = [i for i in col_info if i['field'] in KeepCols]

disease_burden_slice = st.session_state.DiseaseBurden.loc[(st.session_state.DiseaseBurden['admin_name'].isin(st.session_state.RegionsSelected)) & (st.session_state.DiseaseBurden['iso3'] == st.session_state.TargetCountry), KeepCols]
disease_burden_slice.reset_index(inplace=True)
disease_burden_slice.sort_values(by=['admin_name'], ascending=True, inplace=True)
grid_options = f.get_grid_options_verify_5(col_info_filtered)
grid_response = f.get_grid_response(disease_burden_slice, grid_options)
verify_5_edit = pd.DataFrame(grid_response['data'])
verify_5_edit.set_index('index', inplace=True)

## Display inputs required for the capacity - need question
if st.session_state.Question == 'capacity - need':
    # Update inputs using the DNOInput table
    f.update_dno_data(st.session_state.DNOInput, 
        st.session_state.TargetCountry)

    # Display value of work days per year
    verify_1 = {
    'Work Days per Year': 'WorkDaysPerYear'
    }

    verify_1_series = pd.Series({key: st.session_state[value] for key, value in verify_1.items()})
    verify_1_df = verify_1_series.to_frame().reset_index()
    verify_1_df.columns = ['Variable', 'Value']

    grid_options = f.get_grid_options_verify_1_capacity_need()
    grid_response = f.get_grid_response(verify_1_df, grid_options)
    verify_1_edit = pd.DataFrame(grid_response['data'])

    # Display values from the MachineType table
    # -- Create list of all machines selected in the BaselineData
    temp = st.session_state.BaselineData.columns.tolist()
    IgnoreCols = ['admin_name']
    temp2 = [i for i in temp if i not in IgnoreCols]
    selected_machines = list(set([i.split('_')[0] for i in temp2]))

    ColInfo = [{'field' : f'Capacity_Day', 'headerName' : 'Capacity per Day (Tests)', 'format' : 'zero_decimal'}]
    KeepCols = [i['field'] for i in ColInfo]

    machine_types_slice = st.session_state.MachineTypes.loc[selected_machines, KeepCols]
    machine_types_slice.reset_index(inplace=True)
    machine_types_slice.sort_values(by=['Machine'], inplace=True)
    grid_options = f.get_grid_options_verify_2(ColInfo)
    grid_response = f.get_grid_response(machine_types_slice, grid_options)
    verify_2_edit = pd.DataFrame(grid_response['data'])
    verify_2_edit.set_index('Machine', inplace=True)

## Display inputs required for the machine vs. transport or the buy vs. rent questions
else:
    # Overwrite default data with DNO data for TargetCountry or ALL
    # Logic for inflation and analysis year
    #  -- Inflation is only applied during the load of the initial dataframe
    #  -- It is assumed that in final set of inputs provided by the user, the costs/wages are aligned with the analysis year
    f.update_dno_data(st.session_state.DNOInput, 
        st.session_state.TargetCountry,
        st.session_state.AnalysisYear, 
        st.session_state.CPI, 
        st.session_state.Exchange, 
        inflation=True,
        max_year_cpi=st.session_state.MaxYearCPI,
        tradeable_dict=st.session_state.TradeableDict)

    # Apply inflation to cost variables in the MachineTypes table
    cost_cols = []
    for col in st.session_state.MachineTypes.columns:
        if col in st.session_state.TradeableDict.keys() and st.session_state.TradeableDict[col]:
            cost_cols.append(col)
    
    if "apply_inflation_machine_types_called" not in st.session_state:
        st.session_state["apply_inflation_machine_types_called"] = False

    # -- Cost categories within processing and machine cost considered to be "tradeable" intead of "non-tradeable".
    # -- Costs have already been converted from LCU to USD at the time of costing.
    # -- At this stage, only US inflation is applied to the costs to express them in USD for the year of the analysis.
    if not st.session_state["apply_inflation_machine_types_called"]:
        f.apply_inflation_machine_types(st.session_state.MachineTypes, 
            st.session_state.CPI,
            st.session_state.Exchange,
            'USA',  
            st.session_state.AnalysisYear, 
            st.session_state.MaxYearCPI,
            st.session_state.TradeableDict,                       
            cost_cols,
            st.session_state.MachineTypesOriginal)
        st.session_state["apply_inflation_machine_types_called"] = True

    # Display variables with single values
    verify_1 = {
        'Work Days per Year': 'WorkDaysPerYear',
        f'Percentage of {st.session_state.Disease} Samples Tested On-Site': f'TestedOnSite_{st.session_state.Disease}',
        'Weeks per Year Sample Transport Operational': 'TransportWeeksPerYear',
        f'Frequency of {st.session_state.Disease} Sample Transport per Week': 'TransportFreqPerWeek',
        f'Maximum Distance a {st.session_state.Disease} Sample Can Travel (km)': 'DistanceThreshold',
        'Cost per Kilometer for Sample Transport (USD)': 'CostPerKm',
        }

    verify_1_series = pd.Series({key: st.session_state[value] for key, value in verify_1.items()})
    verify_1_df = verify_1_series.to_frame().reset_index()
    verify_1_df.columns = ['Variable', 'Value']

    grid_options = f.get_grid_options_verify_1()
    grid_response = f.get_grid_response(verify_1_df, grid_options)
    verify_1_edit = pd.DataFrame(grid_response['data'])

    # Display values from the MachineType table
    # -- Create list of all machines selected in the BaselineData and AddMachinesData tables.
    temp = st.session_state.BaselineData.columns.tolist() + st.session_state.AddMachinesData.columns.tolist()
    IgnoreCols = ['admin_name']
    temp2 = [i for i in temp if i not in IgnoreCols]
    selected_machines = list(set([i.split('_')[0] for i in temp2]))

    ColInfo = [
        {'field' : f'Capacity_Day', 'headerName' : 'Capacity per Day (Tests)', 'format' : 'zero_decimal'},
        {'field' : f'Processing_Cost_PerSampleVariable_{st.session_state.Disease}', 'headerName' : f'Variable Cost Per {st.session_state.Disease} Test on Owned Machines', 'format' : 'currency'},
        {'field' : f'Processing_Cost_AnnualFixed', 'headerName' : 'Annual Machine Cost for Owned Machines', 'format' : 'currency_zero_decimal'},
        {'field' : f'Processing_Cost_PerSampleVariable_{st.session_state.Disease}_Rented', 'headerName' : f'Cost Per {st.session_state.Disease} Test on Rented Machines (Covers Variable and Machine Costs)', 'format' : 'currency'},
        {'field' : 'Rental_MinAnnualTests', 'headerName' : 'Minimum Annual Tests Paid If Machine Rented', 'format' : 'zero_decimal'},
        {'field' : 'FTE_Laboratory Technologist', 'headerName' : 'Full-Time Staff Required to Operate Machine : Laboratory Technologist', 'format' : 'one_decimal'},
        {'field' : 'FTE_Microbiologist', 'headerName' : 'Full-Time Staff Required to Operate Machine : Microbiologist', 'format' : 'one_decimal'},
        {'field' : 'FTE_Janitor', 'headerName' : 'Full-Time Staff Required to Operate Machine : Janitor', 'format' : 'one_decimal'},
        {'field' : 'FTE_Laboratory Assistant', 'headerName' : 'Full-Time Staff Required to Operate Machine : Laboratory Assistant', 'format' : 'one_decimal'},
        {'field' : 'FTE_Nurse', 'headerName' : 'Full-Time Staff Required to Operate Machine : Nurse', 'format' : 'one_decimal'},
        {'field' : 'FTE_Technician', 'headerName' : 'Full-Time Staff Required to Operate Machine : Technician', 'format' : 'one_decimal'},
        {'field' : 'FTE_Doctor', 'headerName' : 'Full-Time Staff Required to Operate Machine : Doctor', 'format' : 'one_decimal'},
        {'field' : 'FTE_Other', 'headerName' : 'Full-Time Staff Required to Operate Machine : Other', 'format' : 'one_decimal'}]
        
    KeepCols = [i['field'] for i in ColInfo]

    machine_types_slice = st.session_state.MachineTypes.loc[selected_machines, KeepCols]
    machine_types_slice.reset_index(inplace=True)
    machine_types_slice.sort_values(by=['Machine'], inplace=True)

    # Adjust rental processing cost if less than 1.2x the owned processing cost.
    machine_types_slice[f'Processing_Cost_PerSampleVariable_{st.session_state.Disease}_Rented'] = np.where(
        machine_types_slice[f'Processing_Cost_PerSampleVariable_{st.session_state.Disease}_Rented'] < machine_types_slice[f'Processing_Cost_PerSampleVariable_{st.session_state.Disease}'] * 1.2,
        machine_types_slice[f'Processing_Cost_PerSampleVariable_{st.session_state.Disease}'] * 1.2,
        machine_types_slice[f'Processing_Cost_PerSampleVariable_{st.session_state.Disease}_Rented']
        )

    grid_options = f.get_grid_options_verify_2(ColInfo)
    grid_response = f.get_grid_response(machine_types_slice, grid_options)
    verify_2_edit = pd.DataFrame(grid_response['data'])
    verify_2_edit.set_index('Machine', inplace=True)

    # Display values from the StaffCost table
    staff_cost_target = st.session_state.StaffCost.loc[(st.session_state.TargetCountry), ['AnnualSalaryUSD', 'Year']]

    if "apply_inflation_staff_called" not in st.session_state:
        st.session_state["apply_inflation_staff_called"] = False

    if not st.session_state["apply_inflation_staff_called"]:
        staff_cost_target = f.apply_inflation(
            staff_cost_target, 
            st.session_state.CPI,
            st.session_state.Exchange, 
            'AnnualSalaryUSD', 
            'Year', 
            st.session_state.TargetCountry, 
            st.session_state.AnalysisYear,
            st.session_state.MaxYearCPI,
            st.session_state.TradeableDict)
        st.session_state["apply_inflation_staff_called"] = True

    staff_cost_target = staff_cost_target['AnnualSalaryUSD']
    verify_3_df = staff_cost_target.to_frame().reset_index()
    verify_3_df.columns = ['Position', 'AnnualSalaryUSD']

    grid_options = f.get_grid_options_verify_3()
    grid_response = f.get_grid_response(verify_3_df, grid_options)
    verify_3_edit = pd.DataFrame(grid_response['data'])
    verify_3_edit['iso3'] = st.session_state.TargetCountry
    verify_3_edit.set_index(['iso3', 'Position'], inplace=True)

    # Display values of distance per machine, tests per route per year, and other transport network variables
    DNO_transport1_target = st.session_state.DNOTransport1.loc[(st.session_state.DNOTransport1['iso3'] == st.session_state.TargetCountry) & (st.session_state.DNOTransport1['Disease'] == st.session_state.Disease)] 
    DNO_transport1_country_only = st.session_state.DNOTransport1.loc[st.session_state.DNOTransport1['iso3'] == st.session_state.TargetCountry]
    DNO_transport1_disease_only = st.session_state.DNOTransport1.loc[st.session_state.DNOTransport1['Disease'] == st.session_state.Disease]

    # -- Avg based on target country and target disease
    if not DNO_transport1_target.empty:
        RoutesPerMachine_Avg = f.calculate_weighted_average(DNO_transport1_target, 'RoutesPerMachine', 'NumMachines')
        DistancePerRoute_Avg = f.calculate_weighted_average(DNO_transport1_target, 'DistancePerRoute_Raw', 'NumRoutes')
        TestsPerRoute_Avg = f.calculate_weighted_average(DNO_transport1_target, 'TestsPerRoutePerYear', 'NumRoutes')
        df_transport = DNO_transport1_target

    # -- Avg based on target country, across all diseases
    elif not DNO_transport1_country_only.empty:
        RoutesPerMachine_Avg = f.calculate_weighted_average(DNO_transport1_country_only, 'RoutesPerMachine', 'NumMachines')
        DistancePerRoute_Avg = f.calculate_weighted_average(DNO_transport1_country_only, 'DistancePerRoute_Raw', 'NumRoutes')
        TestsPerRoute_Avg = f.calculate_weighted_average(DNO_transport1_country_only, 'TestsPerRoutePerYear', 'NumRoutes')
        df_transport = DNO_transport1_country_only

    # -- Avg based on disease only, across all DNO countries
    else:
        RoutesPerMachine_Avg = f.calculate_weighted_average(DNO_transport1_disease_only, 'RoutesPerMachine', 'NumMachines')
        DistancePerRoute_Avg = f.calculate_weighted_average(DNO_transport1_disease_only, 'DistancePerRoute_Raw', 'NumRoutes')
        TestsPerRoute_Avg = st.session_state[f'TestsPerRoutePerYear_{st.session_state.Disease}'] #pre-calculated based on all routes across all DNO countries for target disease
        df_transport = pd.DataFrame()

    # -- Create a new DataFrame with 'admin_name' of selected regions
    verify_regions = pd.DataFrame(st.session_state.RegionsSelected, columns=['admin_name'])
    verify_regions = f.add_transport_table_data(verify_regions, df_transport,'RoutesPerMachine', RoutesPerMachine_Avg, st.session_state.Disease)
    verify_regions = f.add_transport_table_data(verify_regions, df_transport,'DistancePerRoute_Raw', DistancePerRoute_Avg, st.session_state.Disease)
    verify_regions = f.add_transport_table_data(verify_regions, df_transport, 'TestsPerRoutePerYear', TestsPerRoute_Avg, st.session_state.Disease)
    
    # Use input values of TestsPerMachinePerYear and % Tested On Site to bound the values of TestsPerRoutePerYear for countries without DNO transport data
    if f'{st.session_state.TargetCountry}_{st.session_state.Disease}' not in st.session_state.TransportDataCountries:
        NumTestCols = []
        NumMachineCols = []
        for col in st.session_state.BaselineData:
            if 'NumMachinesTotal' in col:
                NumMachineCols.append(col)
            elif f'TestsPerMachinePerYear_{st.session_state.Disease}' in col:
                NumTestCols.append(col)

        new_df = st.session_state.BaselineData.copy()
        ProductCols = []
        i = 1
        for machine_col, test_col in zip(NumMachineCols, NumTestCols):
            new_df[machine_col] = pd.to_numeric(new_df[machine_col], errors='coerce')
            new_df[test_col] = pd.to_numeric(new_df[test_col], errors='coerce')
            new_df['Product' + str(i)] = new_df[machine_col] * new_df[test_col]
            ProductCols.append('Product' + str(i))
            i += 1

        new_df['TotalMachines'] = new_df[NumMachineCols].sum(axis=1)
        new_df['TotalTestsPerYear_Transported'] = (new_df[ProductCols].sum(axis=1)) * (1- st.session_state[f'TestedOnSite_{st.session_state.Disease}'])

        verify_regions = verify_regions.merge(new_df[['admin_name', 'TotalTestsPerYear_Transported', 'TotalMachines']], on='admin_name', how='left')
        verify_regions['EstimatedTestsPerRoute'] = verify_regions['TotalTestsPerYear_Transported'] / (verify_regions[f'RoutesPerMachine_{st.session_state.Disease}'] * verify_regions['TotalMachines']) 
        verify_regions['EstimatedTestsPerRoute'].replace([np.inf, -np.inf], 0, inplace=True)
        verify_regions[f'TestsPerRoutePerYear_{st.session_state.Disease}'] = np.where(
            (verify_regions['EstimatedTestsPerRoute'] > 0), 
            verify_regions['EstimatedTestsPerRoute'], 
            verify_regions[f'TestsPerRoutePerYear_{st.session_state.Disease}'])

    # Add data on maximum routes per region / number of health facilities
    verify_regions = verify_regions.merge(disease_burden_slice[['admin_name', 'EstimatedHealthFacilities']], on='admin_name', how='left')
    if f'{st.session_state.TargetCountry}_{st.session_state.Disease}' in st.session_state.TransportDataCountries:
        verify_regions = verify_regions.merge(DNO_transport1_target[['admin_name', 'NumRoutes']], on='admin_name', how='left')
        verify_regions['EstimatedHealthFacilities'] = np.where(
            verify_regions['NumRoutes'] > verify_regions['EstimatedHealthFacilities'],
            verify_regions['NumRoutes'],
            verify_regions['EstimatedHealthFacilities'])
        
    # Create and display table
    col_info = [
        {'field' : f'RoutesPerMachine_{st.session_state.Disease}', 'headerName' : f'Average Number of {st.session_state.Disease} Sample Transport Routes Serving Each MolDx Machine', 'format': 'zero_decimal'},
        {'field' : f'DistancePerRoute_Raw_{st.session_state.Disease}', 'headerName' : 'Average One-Way Distance Per Sample Transport Route (km)', 'format': 'one_decimal'},
        {'field' : f'TestsPerRoutePerYear_{st.session_state.Disease}', 'headerName' : f'Average Number of {st.session_state.Disease} Samples Moved Along Each Transport Route Per Year', 'format': 'zero_decimal'},
        {'field' : f'EstimatedHealthFacilities', 'headerName' : f'Maximum {st.session_state.Disease} Sample Transport Routes in Region', 'format': 'zero_decimal'}]

    grid_options = f.get_grid_options_verify_regions(col_info)
    grid_response = f.get_grid_response(verify_regions, grid_options)
    verify_regions_edit = pd.DataFrame(grid_response['data'])

    st.caption("If Maximum Routes in Region is less than the number of machines multiplied by the number of routes per machine, the number of routes per machine will be adjusted downward during the analysis to ensure that Maximum Routes in Region is not exceeded.")

    # Calculate DistancePerMachine using data entered by the user
    verify_regions_edit[f'DistancePerMachine_{st.session_state.Disease}'] = verify_regions_edit[f'RoutesPerMachine_{st.session_state.Disease}'] * verify_regions_edit[f'DistancePerRoute_Raw_{st.session_state.Disease}']

    if 'RegionsTransportData' not in st.session_state:
        st.session_state['RegionsTransportData'] = verify_regions_edit

    # Calculate TestsPerRoute average using data entered by the user
    columns_to_sum = st.session_state.BaselineData.filter(regex='_NumMachinesTotal$', axis=1)
    sum_df = columns_to_sum.sum(axis=1)
    sum_df = pd.DataFrame({'admin_name': st.session_state.BaselineData['admin_name'], 'NumMachines': sum_df})
    merged_df = pd.merge(verify_regions_edit, sum_df, on='admin_name')
    TestsPerRoute_Avg = f.calculate_weighted_average(merged_df, f'TestsPerRoutePerYear_{st.session_state.Disease}', 'NumMachines')

    if 'TestsPerRoute_Avg' not in st.session_state:
        st.session_state['TestsPerRoute_Avg'] = TestsPerRoute_Avg
  
with st.container():
    add_vertical_space(1)
    col1, col2, _ = st.columns(st.session_state.NavButtonCols)

    with col1:
        if st.button('Back', use_container_width=True):
            if st.session_state.Question == 'capacity - need':
                switch_page('select question')
            else:
                switch_page('add machines')

    with col2:
        
        if st.button('Continue', use_container_width=True): 
            
            # Save changes to tables and variables applicable to all modeling questions
            for _, row in verify_1_edit.iterrows():
                    st.session_state[verify_1[row['Variable']]] = row['Value']
            for _, row in verify_4_edit.iterrows():
                st.session_state[verify_4[row['Variable']]] = row['Value']
            st.session_state.DiseaseBurden.update(verify_5_edit)
            st.session_state.MachineTypes.update(verify_2_edit)
            
            # Calculate testing need for each disease, adding calculated columns to the disease_burden table
            f.create_testing_rate_table(st.session_state.DiseaseBurden, 
                  st.session_state.RateDenominator,
                  st.session_state.HIV_EID_TestsPerExposedChild,
                  st.session_state.HIV_VL_TestsPerNewlyDiagnosed,
                  st.session_state.HIV_VL_TestsPerPreviouslyDiagnosed,
                  st.session_state.TB_PositivityRate
                  )
            
            if st.session_state.Question == 'capacity - need':
                switch_page('capacity need')
            
            else:
                # Save changes to tables and variables applicable to machine vs. transport and buy vs. rent questions
                st.session_state.StaffCost.loc[st.session_state.TargetCountry, 'AnnualSalaryUSD'] = verify_3_edit.values
                st.session_state.RegionsTransportData = verify_regions_edit
                st.session_state['TestsPerRoute_Avg'] = TestsPerRoute_Avg

                switch_page('machine vs. transport')