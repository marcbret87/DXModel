import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode, DataReturnMode, JsCode

def update_tracking_dict(tracking_dict: Dict[str, Dict[str, float]], 
                         region_capacity: str, 
                         region_need: str, 
                         new_cost: float, 
                         new_variable_cost: float,
                         new_transport_cost: float,
                         new_tests: float)-> Dict[str, Dict[str, float]]:
    """
    Update the tracking dictionary with new cost and new tests.
    
    Args:
        tracking_dict: A dictionary containing the current tracking information.
        region_capacity: The region providing the capacity.
        region_need: The region with the need for tests.
        new_cost: The new cost to be added to the total cost.
        new_variable_cost: The variable cost from processing new samples.
        new_transport_cost: The cost related to transporting new samples.
        new_tests: The number of new tests provided.
    
    Returns:
        The updated tracking dictionary.
    """
    # Update the respective dictionaries in the tracking dictionary
    CostDicts = {
        'TotalCost_Dict': new_cost,
        'VariableCost_Dict': new_variable_cost,
        'TransportCost_Dict': new_transport_cost
        }

    # Try/Except block to handle the case where the region with need is not in the dictionary
    for key, value in CostDicts.items():
        try:
            tracking_dict[key][region_need] += value
        except:
            tracking_dict[key][region_need] = value

    tracking_dict['UtilizedCapacity_Dict'][region_capacity] += new_tests
    tracking_dict['RemainingCapacity_Dict'][region_capacity] -= new_tests
    tracking_dict['RemainingNeed_Dict'][region_need] -= new_tests
    
    return tracking_dict

def stack_table(df: pd.DataFrame, disease_list: List[str]) -> pd.DataFrame:
    """
    Create a table with one row per region and machine type combination.
    
    Args:
        df (pd.DataFrame): Input data frame containing region and machine type information.
        disease_list (List[str]): List of diseases within scope of the analysis; not only the target disease

    Returns:
        pd.DataFrame: Stacked table with one row per region and machine type combination.
    """
    # Initialize an empty list to store temporary DataFrames
    to_stack = []
    
    # Reset the index if the index name is 'admin_name'
    if df.index.name == 'admin_name':
        df.reset_index(inplace=True, names='admin_name')
    
    # Create a list of column names, excluding 'admin_name'
    col_list = list(df.columns)
    col_list.remove('admin_name')
    
    # Iterate through each column name
    for col in col_list:
        # Initialize an empty list to store column names related to tests
        col_test_list = []
        
        # Split the column name by '_' and assign the parts to variables
        machine_type = col.split('_')[0]
        col_type = col.split('_')[1]
    
        # Check if the current column is for the total number of machines
        if col_type == 'NumMachinesTotal':
            
            # Iterate through the diseases and create column names for tests per machine per year
            for disease in disease_list:
                col_tests = f"{machine_type}_TestsPerMachinePerYear_{disease}"
                col_test_list.append(col_tests)
            
            # Create a column name for rented machines
            col_rented = f"{machine_type}_NumMachinesRented"
        
            # Create a list of column names to include in the temporary DataFrame
            cols = ['admin_name', col, *col_test_list]
            cols.append(col_rented)
            
            # Create a temporary DataFrame with the specified columns
            temp = df.loc[:, cols]
            
            # Drop rows with NaN values in the 'col' column
            temp.dropna(subset=[col], inplace=True)
            
            # Add a new column for the machine type
            temp['Machine'] = machine_type
            
            # Rename the columns
            temp.rename(columns={col: 'MachineQuantity'}, inplace=True)
            for c in col_test_list:
                temp.rename(columns={c: f"TestsPerMachinePerYear_{c.split('_')[2]}"}, inplace=True)
            temp.rename(columns={col_rented: 'MachineQuantity_Rented'}, inplace=True)
            
            # Append the temporary DataFrame to the to_stack list
            to_stack.append(temp)

    # Concatenate all the temporary DataFrames in to_stack and create the output DataFrame
    df_out = pd.concat(to_stack, ignore_index=True)
    return df_out

def calculate_percent_below_threshold(df_target: pd.DataFrame, threshold: float) -> Tuple[float, float]:
    """
    Calculate the percentage of tests and total distance below a given distance threshold.
    
    Args:
        df_target (pd.DataFrame): Input DataFrame containing the columns 'DistancePerRoute',
                                  'PercentTotalTests', and 'PercentTotalDistance'.
        threshold (float): The distance threshold to apply for calculating the percentages.

    Returns:
        Tuple[float, float]: A tuple containing the percentage of tests below the threshold and
                             the percentage of total distance below the threshold.
    """
    # Add a new column 'row_num' with the row numbers to the DataFrame
    df_target.insert(0, 'row_num', range(0, len(df_target)))
    
    # Find the row where the 'DistancePerRoute' column exceeds the threshold
    threshold_row = int(df_target.loc[df_target['DistancePerRoute'] > threshold, ['row_num']].head(1).values) - 1

    # Calculate the percentage of tests transported below the threshold distance
    percent_tests_below_threshold = float(df_target.loc[df_target['row_num'] == threshold_row, ['PercentTotalTests']].values)

    # Calculate the percentage of total distance below the threshold distance
    percent_distance_below_threshold = float(df_target.loc[df_target['row_num'] == threshold_row, ['PercentTotalDistance']].values)
    
    return percent_tests_below_threshold, percent_distance_below_threshold

def apply_threshold_tests(df: pd.DataFrame,
                          df_transport: pd.DataFrame,
                          target_country: str,
                          threshold: float,
                          on_site: float,
                          disease: str) -> pd.DataFrame:
    """
    Apply a distance threshold to the tests and distance values in the input DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the columns related to tests and distance.
        df_transport (pd.DataFrame): Transport data DataFrame.
        target_country (str): ISO3 code of the target country.
        threshold (float): The distance threshold to apply.
        on_site (float): The percentage of tests to be performed on site.
        disease (str): Disease for which to apply the threshold.

    Returns:
        pd.DataFrame: DataFrame with updated tests and distance values based on the distance threshold.
    """
    # Create a list of all unique country-disease pairs in df_transport
    temp = list(set(i + j for i, j in zip(df_transport['iso3'], df_transport['Disease'])))

    # Select the target rows from df_transport based on target_country and disease
    if target_country + disease in temp:
        df_target = df_transport.loc[(df_transport['iso3'] == target_country) & (df_transport['Disease'] == disease)]
    elif target_country in list(df_transport['iso3']):
        df_target = df_transport.loc[df_transport['iso3'] == target_country]
    else:
        df_target = df_transport.loc[(df_transport['iso3'] == 'ALL') & (df_transport['Disease'] == disease)]

    # Adjust values for tests and distance based on distance threshold
    if threshold < np.max(df_target['DistancePerRoute']):
        percent_tests_below_threshold, percent_distance_below_threshold = calculate_percent_below_threshold(df_target, threshold)
        
        for col in df.columns:
            if f'TestsPerMachinePerYear_{disease}' in col:
                df[col] = (on_site * df[col]) + ((1 - on_site) * df[col] * percent_tests_below_threshold)
        
        df[f'DistancePerMachine_{disease}'] = df[f'DistancePerMachine_{disease}'] * percent_distance_below_threshold

    return df

def add_staff_costs_per_machine_per_year(df: pd.DataFrame, df_staff: pd.DataFrame, target_country: str) -> pd.DataFrame:
    """
    Add staff costs per machine per year to the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing the columns related to staff.
        df_staff (pd.DataFrame): Staff data DataFrame with index having two levels: country and position.
        target_country (str): Target country to add staff costs for.

    Returns:
        pd.DataFrame: DataFrame with staff costs added.
    """

    # Define a constant for the salary column name
    ANNUAL_SALARY_COLUMN = 'AnnualSalaryUSD'

    # Mapping of staff positions
    staff_map = {
        'FTE_Laboratory Technologist': 'Laboratory Technologist',
        'FTE_Microbiologist': 'Microbiologist',
        'FTE_Janitor': 'Janitor',
        'FTE_Laboratory Assistant': 'Laboratory Assistant',
        'FTE_Nurse': 'Nurse',
        'FTE_Technician': 'Technician',
        'FTE_Doctor': 'Doctor',
        'FTE_Other': 'Other'
    }

    col_list = []
    for col, pos in staff_map.items():
        # Calculate annual staff costs and add them to the DataFrame
        annual_salary = float(df_staff.loc[(df_staff.index.get_level_values(0) == target_country) &
                                           (df_staff.index.get_level_values(1) == pos)][ANNUAL_SALARY_COLUMN].values)
        df[f'AnnualCost_{pos}'] = df[col] * annual_salary
        col_list.append(f'AnnualCost_{pos}')
    
    # Calculate total staff cost per machine per year and add it to the DataFrame
    df['StaffCost_PerMachinePerYear'] = df[col_list].sum(axis=1, skipna=True)
    
    # Drop temporary columns used for calculations
    df.drop(columns=col_list, inplace=True)

    return df

def add_summary_columns(df: pd.DataFrame,
                        cost_per_km: float,
                        transport_freq_per_week: int,
                        transport_weeks_per_year: int,
                        disease: str,
                        disease_list: List[str]) -> pd.DataFrame:
    """
    Add summary columns to the input DataFrame for costs and utilized capacity.

    Args:
        df (pd.DataFrame): Input DataFrame containing machine and processing cost data.
        cost_per_km (float): Cost per kilometer for transportation.
        transport_freq_per_week (int): Transportation frequency per week.
        transport_weeks_per_year (int): Number of weeks per year transportation occurs.
        disease (str): Target disease to calculate summary columns for.
        disease_list (List[str]): List of diseases within scope of the analysis; not only the target disease

    Returns:
        pd.DataFrame: DataFrame with summary columns added.
    """

    df['TestsPerMachinePerYear_All'] = df[[i for i in list(df.columns) if 'TestsPerMachinePerYear' in i]].sum(axis=1)

    # Calculate proportion of total tests performed that are for the target disease
    df['PropTargetDisease'] = np.where(
        df['TestsPerMachinePerYear_All'] != 0, 
        df['TestsPerMachinePerYear_' + disease] / df['TestsPerMachinePerYear_All'], 
        0
    )
    df['PropTargetDisease'].fillna(0, inplace=True)

    # Calculate utilized capacity for each disease
    for d in disease_list:
        df[f'UtilizedCapacity_{d}'] = df['MachineQuantity'] * df[f'TestsPerMachinePerYear_{d}']
        df[f'UtilizedCapacity_{d}_Owned'] = df['MachineQuantity_Owned'] * df[f'TestsPerMachinePerYear_{d}']

    # Calculate total processing cost for the target disease
    df[f'StaffCost_Total_{disease}'] = df['MachineQuantity'] * df['StaffCost_PerMachinePerYear'] * df['PropTargetDisease']
    
    # -Owned machines
    df[f'MachineCost_Total_{disease}_Owned'] = df['MachineQuantity_Owned'] * df['Processing_Cost_AnnualFixed'] * df['PropTargetDisease']
    df[f'VariableCost_Total_{disease}_Owned'] = df[f'Processing_Cost_PerSampleVariable_{disease}'] * df[f'UtilizedCapacity_{disease}_Owned']
    for d in disease_list:
        df.drop(columns=[f'UtilizedCapacity_{d}_Owned'], inplace=True)

    # -Rented machines
    # --Apply threshold for minimum required tests for rented machines.  Threshold compared to all tests performed, not just target disease tests.
    # --For a given subnational region and machine type, there is no distinction between owned and rented machines for TestsPerMachinePerYear
    # --If the value of minimum annual tests is used instead of the original value of TestsPerMachinePerYear, the proprtion of that minimum 
    # -- allocated to the target disease (and thus the target disease cost) is based on df['PropTargetDisease'] 
    df['TestsPerMachinePerYear_Rented_Minimum'] = np.where(df['TestsPerMachinePerYear_All'] < df['Rental_MinAnnualTests'], df['Rental_MinAnnualTests'], df['TestsPerMachinePerYear_All'])
    df[f'MachineAndVariableCost_Total_{disease}_Rented'] = df[f'Processing_Cost_PerSampleVariable_{disease}_Rented'] * df['MachineQuantity_Rented'] * df['TestsPerMachinePerYear_Rented_Minimum'] * df['PropTargetDisease']
    
    processing_cost_total_col = f'ProcessingCost_Total_{disease}'
    df[processing_cost_total_col] = df[[f'StaffCost_Total_{disease}', f'MachineCost_Total_{disease}_Owned', f'VariableCost_Total_{disease}_Owned', f'MachineAndVariableCost_Total_{disease}_Rented']].sum(axis=1)
        
    # Fill missing values with zeros
    df[processing_cost_total_col].fillna(0, inplace=True)
    df[f'TestsPerMachinePerYear_{disease}'].fillna(0, inplace=True)

    # Calculate total transport cost for the target disease
    transport_cost_total_col = f'TransportCost_Total_{disease}'
    df[transport_cost_total_col] = np.where(df[f'TestsPerMachinePerYear_{disease}'] == 0, 0, df['MachineQuantity'] * df[f'DistancePerMachine_{disease}'] * cost_per_km * transport_freq_per_week * transport_weeks_per_year)

    # Calculate total cost for the target disease
    df[f'TotalCost_{disease}'] = df[[processing_cost_total_col, transport_cost_total_col]].sum(axis=1)

    df[f'MachineCost_Total_{disease}'] = df[f'MachineCost_Total_{disease}_Owned']
    # -For rented machines, per test costs that cover the machine and the test are included in the variable cost total
    df[f'VariableCost_Total_{disease}'] = df[[f'VariableCost_Total_{disease}_Owned', f'MachineAndVariableCost_Total_{disease}_Rented']].sum(axis=1)

    return df

def add_calculated_columns(df: pd.DataFrame,
                           df_transport: pd.DataFrame,
                           df_staff: pd.DataFrame,
                           threshold: float,
                           target_country: str,
                           on_site: float,
                           disease: str,
                           disease_list: List[str],
                           cost_per_km: float,
                           transport_freq_per_week: int,
                           transport_weeks_per_year: int) -> pd.DataFrame:
    """
    Add calculated columns to the input DataFrame using multiple processing functions.
    
    Args:
        df (pd.DataFrame): Input DataFrame to add calculated columns.
        df_transport (pd.DataFrame): Transport data DataFrame.
        df_staff (pd.DataFrame): Staff salary data DataFrame.
        threshold (float): Distance threshold to apply for transport calculations.
        target_country (str): ISO3 code of the target country.
        on_site (float): Percentage of tests performed on-site.
        disease (str): Target disease
        disease_list (List[str]): List of diseases within scope of the analysis; not only the target disease
        cost_per_km (float): Cost per kilometer for transport calculations.
        transport_freq_per_week (int): Frequency of transport per week.
        transport_weeks_per_year (int): Number of weeks per year transportation occurs.

    Returns:
        pd.DataFrame: DataFrame with calculated columns added.
    """

    # Apply distance threshold to the annual number of tests and total transport distance values 
    df = apply_threshold_tests(df, df_transport, target_country, threshold, on_site, disease)
    
    # Add staff costs based on FTE counts and staff salaries
    df = add_staff_costs_per_machine_per_year(df, df_staff, target_country)

    # Add summary columns
    df = add_summary_columns(df, cost_per_km, transport_freq_per_week, transport_weeks_per_year, disease, disease_list)

    return df

def calculate_statistics(df: pd.DataFrame, 
                         df_disease_burden: pd.DataFrame, 
                         disease: str, 
                         disease_list: List[str]) -> Dict[str, float]:

    """
    Calculate summary statistics for the input DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing cost and utilized capacity columns.
        df_disease_burden (pd.DataFrame): Dataframe with subnational level disease burden and population data
        disease (str): Target disease
        disease_list (List[str]): List of diseases within scope of the analysis; not only the target disease

    Returns:
        Dict[str, float]: A dictionary containing calculated summary statistics.
    """

    # Initialize a dictionary to store calculated statistics
    stats = {}

    # Calculate total cost, number of samples processed for the target disease, unit cost (cost per test), and percentage of need met
    stats['TotalCost'] = df[f'TotalCost_{disease}'].sum()
    stats['MachineCost'] = df[f'MachineCost_Total_{disease}'].sum()
    stats['StaffCost'] = df[f'StaffCost_Total_{disease}'].sum()
    stats['VariableCost'] = df[f'VariableCost_Total_{disease}'].sum()
    stats['TransportCost'] = df[f'TransportCost_Total_{disease}'].sum()

    stats['NumTests'] = df[f'UtilizedCapacity_{disease}'].sum()

    stats['UnitCost_Total'] = stats['TotalCost'] / stats['NumTests']
    stats['UnitCost_Machine'] = stats['MachineCost'] / stats['NumTests']
    stats['UnitCost_Staff'] = stats['StaffCost'] / stats['NumTests']
    stats['UnitCost_Variable'] = stats['VariableCost'] / stats['NumTests']
    stats['UnitCost_Transport'] = stats['TransportCost'] / stats['NumTests']

    stats['NeedMet'] = df[f'UtilizedCapacity_{disease}'].sum() / df_disease_burden[f'TestingNeed_{disease}'].sum()

    stats['NumTests_All'] = sum([df[f'UtilizedCapacity_{x}'].sum() for x in disease_list]) # Total number of tests per year performed across all diseases, only on machines which can process tests for the target disease
    stats['TotalCapacity'] = df[f'TotalCapacity_{disease}'].sum() # Total annual capacity of all machines which can process tests for the target disease
    stats['UtilizationRate'] = stats['NumTests_All'] / stats['TotalCapacity']

    return stats

def calculate_weighted_average(df: pd.DataFrame, 
                               col_main: Union[str, List[str]], 
                               col_weight: Union[str, List[str]]) -> Union[float, pd.DataFrame]:
    """
    Calculate weighted average based on the given column names or column name patterns in the input DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing columns for weighted average calculation.
        col_main (Union[str, List[str]]): Column name or list of column names for main columns used in the calculation.
        col_weight (Union[str, List[str]]): Column name or list of column names for weight columns used in the calculation.
    
    Returns:
        Union[float, pd.DataFrame]: A float value if col_main and col_weight are strings, or a DataFrame with weighted average calculated for each row if col_main and col_weight are lists.
    
    Raises:
        ValueError: If col_main and col_weight are not of the same type (either both strings or both lists).
    """
    
    # Case when col_main and col_weight are strings (single column names)
    if isinstance(col_main, str) and isinstance(col_weight, str):
        avg = (df[col_main] * df[col_weight]).sum() / df[col_weight].sum()
        return avg
    
    # Case when col_main and col_weight are lists (column name patterns)
    elif isinstance(col_main, list) and isinstance(col_weight, list):
        temp = df.copy()
        
        # Calculate sum of products and sum of weights for each row
        temp['SumProduct'] = temp[col_main].sum(axis=1)
        temp['SumWeight'] = temp[col_weight].sum(axis=1)
        
        # Calculate weighted average for each row
        temp['Avg'] = temp['SumProduct'] / temp['SumWeight']
        
        # Return DataFrame with 'admin_name' and calculated weighted average
        return temp[['admin_name', 'Avg']]
    
    else:
        raise ValueError("col_main and col_weight must both be either str or list")

def calculate_weighted_average_rows(df: pd.DataFrame, colname_main: str, colname_weight: str) -> pd.DataFrame:
    """
    Calculate weighted average for rows based on the given column name patterns in the input DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing columns for weighted average calculation.
        colname_main (str): Column name pattern to identify main columns for calculation.
        colname_weight (str): Column name pattern to identify weight columns for calculation.
    
    Returns:
        pd.DataFrame: DataFrame with weighted average calculated for each row.
    """
    temp = df.copy()
    col_list_main = []
    col_list_weight = []
    
    for col in temp.columns:
        if colname_main in col:
            machine_type = col.split('_')[0]
            
            # Create product columns using f-strings
            product_col = f"{machine_type}_Product"
            weight_col = f"{machine_type}_{colname_weight}"
            temp[product_col] = temp[col] * temp[weight_col]
            
            col_list_main.append(product_col)
            col_list_weight.append(weight_col)
    
    # Call the calculate_weighted_average function to compute the weighted averages
    weighted_averages = calculate_weighted_average(temp, col_list_main, col_list_weight)
    
    return weighted_averages

def calculate_weighted_average_utilization(df: pd.DataFrame, 
                                           df_machine_types: pd.DataFrame, 
                                           colname_main: str, 
                                           colname_weight: str, 
                                           axis: str, 
                                           work_days_per_year: int) -> Union[float, pd.DataFrame]:
    """
    Calculate the weighted average utilization of machines across columns or rows.

    Args:
        df (pd.DataFrame): The input DataFrame containing machine utilization data.
        df_machine_types (pd.DataFrame): A DataFrame containing information about each machine type
        colname_main (str): The main column name to be used for calculating utilization.
        colname_weight (str): The weight column name to be used for calculating the weighted average.
        axis (str): The axis along which the weighted average should be calculated ('row' or 'column').
        work_days_per_year (int): The number of work days per year.

    Returns:
        Union[float, pd.DataFrame]: A DataFrame with weighted average utilization per row if axis is 'row', otherwise returns a float with the overall weighted average utilization.
    """
    col_list = []
    col_list_weight = []
    temp = df.copy()

    # Calculate the utilization for each machine type
    for col in temp:
        if colname_main in col:
            machine_type = col.split('_')[0]
            machine_capacity = df_machine_types.loc[machine_type, 'Capacity_Day'] * work_days_per_year
            temp[f'{machine_type}_Utilization'] = temp[col] / machine_capacity
            col_list.append(f'{machine_type}_Utilization')
            col_list_weight.append(f'{machine_type}_{colname_weight}')
    
    # Calculate the weighted average utilization either per row or overall
    if axis == 'row':
        temp2 = calculate_weighted_average_rows(temp, 'Utilization', colname_weight)
    else:
        avg_list = []
        weight_list = []
        for col in col_list:
            machine_type = col.split('_')[0]
            avg = calculate_weighted_average(temp, col, f'{machine_type}_NumMachinesTotal')
            if np.isnan(avg):
                avg_list.append(0)
            else:
                avg_list.append(avg)
            weight_list.append(temp[f'{machine_type}_NumMachinesTotal'].sum())

        temp2 = np.average(avg_list, weights=weight_list)

    return temp2

def create_route_list_new_tests(new_tests: int, 
                                tests_per_route_per_year: int,
                                new_route_distance: float,
                                num_routes_baseline: float,
                                max_routes: int, 
                              ) -> List[float]:
    
    """
     This function creates a list of new intra-regional routes, where all new routes have the same length.
    The total number of routes is ensured not to exceed the maximum possible routes (max_routes).

    Args:
        new_tests (int): The number of new tests to be conducted.
        tests_per_route_per_year (int): The number of tests per route per year.
        new_route_distance (float): The distance for the new routes to be created.
        num_routes_baseline (float): The initial number of routes.
        max_routes (int): The maximum possible number of routes.

    Returns:
        route_list (List[float]): A list containing distances of new routes.
        max_routes_cap (bool): A boolean value indicating whether the final number of new routes equals the maximum number of new routes.
    """
    max_routes_cap = False
    num_routes_new = int(np.ceil(new_tests / tests_per_route_per_year))
    max_new_routes = max_routes - num_routes_baseline
    num_routes_new = int(min(num_routes_new, max_new_routes))
    if num_routes_new == max_new_routes:
        max_routes_cap = True
    route_list = [new_route_distance] * num_routes_new
    
    return route_list, max_routes_cap

def calculate_transport_cost_new_tests(route_list_below_threshold: List[float], 
                                       cost_per_km: float, 
                                       transport_freq_per_week: float, 
                                       transport_weeks_per_year: float) -> float:
    """
    Calculate the total transport cost for new tests given a list of route distances below a threshold.
    
    Args:
        route_list_below_threshold: A list of route distances below the threshold.
        cost_per_km: The cost per kilometer of transportation.
        transport_freq_per_week: The frequency of transportation per week.
        transport_weeks_per_year: The number of weeks per year when transportation occurs.
    
    Returns:
        The total transport cost for new tests.
    """
    total_cost = sum(route_list_below_threshold) * cost_per_km * transport_freq_per_week * transport_weeks_per_year
    return total_cost

def calculate_num_routes(total_tests: int, 
                         budget: float, 
                         tests_per_route_per_year: int, 
                         route_list: List[float], 
                         cost_per_km: float, 
                         transport_freq_per_week: float, 
                         transport_weeks_per_year: float, 
                         variable_cost_per_test: float,
                         max_routes_cap) -> Tuple[int, float, float]:
    """
    Calculate the number of routes and number of tests that can be allocated within the given budget for testing.

    Args:
        total_tests (int): The total number of tests to be conducted.
        budget (float): The total budget available for testing.
        tests_per_route_per_year (int): The number of tests conducted per route per year.
        route_list (List[float]): A list of route distances.
        cost_per_km (float): The cost per kilometer of transportation.
        transport_freq_per_week (float): The frequency of transportation per week.
        transport_weeks_per_year (float): The number of weeks per year when transportation occurs.
        variable_cost_per_test (float): The variable cost per test conducted.
        max_routes_cap (bool): A boolean value indicating whether the maximum number of new routes has been reached.

    Returns:
        Tuple[List[float], int, float, float, int]: A tuple containing:
        - A list with the distances of each of the retained intra-regional transport routes.
        - The number of routes that can be allocated within the given budget.
        - The variable cost of conducting tests on the allocated routes.
        - The transportation cost of the allocated routes.
        - The total number of tests that can be conducted within the budget.
    """    
    # Calculate the maximum number of routes
    max_routes = int(np.ceil(total_tests / tests_per_route_per_year))
    if len(route_list) < max_routes:
        max_routes = len(route_list)
    
    num_routes = max_routes
    diff = -1
    
    transport_cost = calculate_transport_cost_new_tests(route_list[:num_routes], cost_per_km, transport_freq_per_week, transport_weeks_per_year)
    variable_cost = num_routes * tests_per_route_per_year * variable_cost_per_test
    diff = budget - (transport_cost + variable_cost)
    
    # In this case, there is budget to create all transport routes but not all variable costs
    # Route saturation is reached, and the number of tests moved along each route is higher than tests_per_route_per_year
    if max_routes_cap and diff >= 0:
        budget_variable = budget - transport_cost
        num_tests = np.floor(budget_variable / variable_cost_per_test)
        variable_cost = num_tests * variable_cost_per_test

    # In this case, there is not sufficient budget to create all transport routes
    else:
        while diff < 0:
            transport_cost = calculate_transport_cost_new_tests(route_list[:num_routes], cost_per_km, transport_freq_per_week, transport_weeks_per_year)
            variable_cost = num_routes * tests_per_route_per_year * variable_cost_per_test
            diff = budget - (transport_cost + variable_cost)
            num_routes -= 1

        num_routes += 1
        transport_cost = calculate_transport_cost_new_tests(route_list[:num_routes], cost_per_km, transport_freq_per_week, transport_weeks_per_year)
        num_tests = num_routes * tests_per_route_per_year
        variable_cost = num_tests * variable_cost_per_test
    
    route_list_final = route_list[:num_routes]
    return route_list_final, num_routes, variable_cost, transport_cost, num_tests 

import pandas as pd
from typing import Union

def add_transport_table_data(df: pd.DataFrame, 
                             df_transport: pd.DataFrame, 
                             column_name: str, 
                             avg: Union[float, int], 
                             disease: str) -> pd.DataFrame:
    """
    Adds data to the main data frame from the transport data frame based on a specified column and disease type.

    Args:
        df (pd.DataFrame): The main data frame.
        df_transport (pd.DataFrame): The transport data frame.
        column (str): The column name in the transport data frame to add to the main data frame.
        avg (Union[float, int]): The average value to use if the transport data frame is empty.
        disease (str): The target disease

    Returns:
        pd.DataFrame: The updated main data frame.
    """
    if not df_transport.empty:
        # Merge transport data into main data frame
        # --Works for the CountryOnly dataframe only because there are only two diseases.
        # --More diseases would cause a matching issue (multiple rows with the same admin_name)
        df = df.merge(df_transport[[column_name, 'admin_name']], on='admin_name', how='left')
        df[column_name].fillna(avg, inplace=True)
    else:
        df[column_name] = avg

    # Rename column in the main data frame to include the disease type
    df.rename({column_name: column_name + '_' + disease}, inplace=True, axis=1)

    return df

def get_grid_options_baseline(df: pd.DataFrame, ignore_cols: List[str]) -> Dict[str, object]:
    """Generate options for an AgGrid.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        ignore_cols (List[str]): Columns to ignore while generating options.

    Returns:
        Dict[str, object]: Dictionary containing options for AgGrid.
    """
    grid_options = {}
    
    # Format columns
    ColumnDefsList = []
    ColumnDefsList.append(
            {
                "field": "admin_name", 
                "headerName": "Region",
                "pinned": "left",
                "editable": False
            }
        )
    
    temp = {}
    for col in df.columns:
        if col not in ignore_cols:

            col_type = col.split('_')[1]
            if col_type == "NumMachinesTotal":

                machine_type = col.split('_')[0]
                temp[col] = {
                    "field": machine_type,
                    "headerName": machine_type,
                    "children": [
                        {
                            "field": col, 
                            "headerName": "Total Machines", 
                            "editable": True, 
                            "wrapHeaderText": True,
                            "resizable": True,
                            "autoHeaderHeight": True,
                            "precision": 0,
                            "type": [
                                "numericColumn",
                                "numberColumnFilter",
                                "customNumericFormat"
                                ]
                        },
                        {
                            "field": f"{machine_type}_NumMachinesRented", 
                            "headerName": "Rented Machines", 
                            "editable": True, 
                            "wrapHeaderText": True,
                            "resizable": True,
                            "autoHeaderHeight": True,
                            "precision": 0,
                            "type": [
                                "numericColumn",
                                "numberColumnFilter",
                                "customNumericFormat"
                                ]
                        },
                        {
                            "field": f"{machine_type}_TestsPerMachinePerYear_HIV", 
                            "headerName": "Tests per Machine Per Year : HIV", 
                            "editable": True, 
                            "wrapHeaderText": True,
                            "resizable": True,
                            "autoHeaderHeight": True,
                            "precision": 0,
                            "type": [
                                "numericColumn",
                                "numberColumnFilter",
                                "customNumericFormat"
                                ]
                        },
                        {
                            "field": f"{machine_type}_TestsPerMachinePerYear_TB", 
                            "headerName": "Tests per Machine Per Year : TB", 
                            "editable": True, 
                            "wrapHeaderText": True,
                            "resizable": True,
                            "autoHeaderHeight": True,
                            "precision": 0,
                            "type": [
                                "numericColumn",
                                "numberColumnFilter",
                                "customNumericFormat"
                                ]
                        },
                        {
                            "field": f"{machine_type}_TestsPerMachinePerYear_Other", 
                            "headerName": "Tests per Machine Per Year : Other", 
                            "editable": True, 
                            "wrapHeaderText": True,
                            "resizable": True,
                            "autoHeaderHeight": True,
                            "precision": 0,
                            "type": [
                                "numericColumn",
                                "numberColumnFilter",
                                "customNumericFormat"
                                ]
                        }
                    ]
                }
          
    for group_dict in temp.values():
        ColumnDefsList.append(group_dict)

    grid_options["columnDefs"] = ColumnDefsList

    # Set other options
    grid_options["defaultColDef"] = {
        "minWidth":5,
        "filter":True,
        "resizable":True,
        "sortable":True,
        "aggFunc": "sum",
        }

    grid_options["alwaysShowHorizontalScroll"] = True
    grid_options["alwaysShowVerticalScroll"] = True

    return grid_options

def get_grid_options_add_machines(df: pd.DataFrame, ignore_cols: List[str]) -> Dict[str, object]:
    """Generate options for an AgGrid.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        ignore_cols (List[str]): Columns to ignore while generating options.

    Returns:
        Dict[str, object]: Dictionary containing options for AgGrid.
    """
    grid_options = {}
    
    # Format columns
    ColumnDefsList = []
    ColumnDefsList.append(
            {
                "field": "admin_name", 
                "headerName": "Region",
                "pinned": "left",
                "editable": False
            }
        )

    temp = {}
    for col in df.columns:
        if col not in ignore_cols:

            col_type = col.split('_')[1]
            if col_type == "NumMachinesTotal":

                machine_type = col.split('_')[0]
                temp[col] = {
                    "field": machine_type,
                    "headerName": machine_type,
                    "children": [
                        {
                            "field": col, 
                            "headerName": "Total Machines", 
                            "editable": True, 
                            "wrapHeaderText": True,
                            "resizable": True,
                            "autoHeaderHeight": True,
                            "precision": 0,
                            "type": [
                                "numericColumn",
                                "numberColumnFilter",
                                "customNumericFormat"
                                ]
                        },
                        {
                            "field": f"{machine_type}_NumMachinesRented", 
                            "headerName": "Rented Machines", 
                            "editable": True, 
                            "wrapHeaderText": True,
                            "resizable": True,
                            "autoHeaderHeight": True,
                            "precision": 0,
                            "type": [
                                "numericColumn",
                                "numberColumnFilter",
                                "customNumericFormat"
                                ]
                        },
                        
                    ]
                }
          
    for group_dict in temp.values():
        ColumnDefsList.append(group_dict)

    grid_options["columnDefs"] = ColumnDefsList

    # Set other options
    grid_options["defaultColDef"] = {
        "minWidth":5,
        "filter":True,
        "resizable":True,
        "sortable":True,
        "aggFunc": "sum",
        }

    grid_options["alwaysShowHorizontalScroll"] = True
    grid_options["alwaysShowVerticalScroll"] = True

    return grid_options

def get_grid_options_verify_1() -> Dict[str, object]:
    """
    Generate options for an AgGrid.

    Args:
        None

    Returns:
        Dict[str, object]: Dictionary containing options for AgGrid.
    """
    grid_options = {}
    
    # Format columns
    ColumnDefsList = [

        {
            "field": "Variable", 
            "headerName": "Variable",
            "pinned": "left",
            "editable": False,
        },
        {
            "field": 'Value', 
            "headerName": 'Value',
            "editable": True,
            "wrapHeaderText": True,
            "resizable": True,
            "autoHeaderHeight": True,
            'cellRenderer': js_code1
        }]

    grid_options["columnDefs"] = ColumnDefsList

    # Set other options
    grid_options["defaultColDef"] = {
        "minWidth":10,
        "filter":True,
        "resizable":True,
        "sortable":True,
        }

    grid_options["alwaysShowHorizontalScroll"] = False
    grid_options["alwaysShowVerticalScroll"] = False

    return grid_options

def get_grid_options_verify_1_capacity_need() -> Dict[str, object]:
    """Generate options for an AgGrid.

    Args:
        None

    Returns:
        Dict[str, object]: Dictionary containing options for AgGrid.
    """
    grid_options = {}
    
    # Format columns
    ColumnDefsList = [

        {
            "field": "Variable", 
            "headerName": "Variable",
            "pinned": "left",
            "editable": False,
        },
        {
            "field": 'Value', 
            "headerName": 'Value',
            "editable": True,
            "wrapHeaderText": True,
            "resizable": True,
            "autoHeaderHeight": True,
            'cellRenderer': js_code2
        }]

    grid_options["columnDefs"] = ColumnDefsList

    # Set other options
    grid_options["defaultColDef"] = {
        "minWidth":10,
        "filter":True,
        "resizable":True,
        "sortable":True,
        }

    grid_options["alwaysShowHorizontalScroll"] = False
    grid_options["alwaysShowVerticalScroll"] = False

    return grid_options


def get_grid_options_verify_2(col_info: Dict) -> Dict[str, object]:
    """Generate options for an AgGrid.

    Args:
        col_info (Dict): Dictionary containing information about each column such as the field name, header name, and format.

    Returns:
        Dict[str, object]: Dictionary containing options for AgGrid.
    """
    grid_options = {}
    
    # Format columns
    ColumnDefsList = [

        {
            "field": "Machine", 
            "headerName": "Machine",
            "pinned": "left",
            "editable": False
        }]
        
    for col in col_info:
        temp = {
                "field": col['field'], 
                "headerName": col['headerName'],
                "editable": True,
                "wrapHeaderText": True,
                "resizable": True,
                "autoHeaderHeight": True,
                "precision": 0,
                "type": [
                    "numericColumn",
                    "numberColumnFilter",
                    ]
            }
        if col['format'] == "currency":
            temp["valueFormatter"] = JsCode("""
                function(params) {
                    var value = parseFloat(params.value);
                    if (Number.isNaN(value) || value === 0) {
                        return '';
                    } else {
                        return "$" + value.toFixed(2);
                    }
                }
            """)
        elif col['format'] == "currency_zero_decimal":
            temp["valueFormatter"] = JsCode("""
                            function(params) {
                                if (params.value === 0 || isNaN(params.value)) {
                                    return '';
                                } else {
                                    return "$" + parseFloat(params.value).toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
                                }
                            }
                        """)

        elif col['format'] == "zero_decimal":
            temp["valueFormatter"] = JsCode("""
                    function(params) {
                        var value = parseFloat(params.value);
                        if (Number.isNaN(value) || value === 0) {
                            return '';
                        } else {
                            return (Math.round(value)).toLocaleString();
                        }
                    }
                """)
        elif col['format'] == "one_decimal":
            temp["valueFormatter"] = JsCode("""
                function(params) {
                    var value = parseFloat(params.value);
                    if (Number.isNaN(value) || value === 0) {
                        return '';
                    } else {
                        return parseFloat(params.value).toLocaleString('en-US', {minimumFractionDigits: 1, maximumFractionDigits: 1});
                    }
                }
            """)

        ColumnDefsList.append(temp)

    grid_options["columnDefs"] = ColumnDefsList

    # Set other options
    grid_options["defaultColDef"] = {
        "minWidth":5,
        "filter":True,
        "resizable":True,
        "sortable":True,
        }

    grid_options["alwaysShowHorizontalScroll"] = True
    grid_options["alwaysShowVerticalScroll"] = False

    return grid_options

def get_grid_options_verify_3() -> Dict[str, object]:
    """Generate options for an AgGrid.

    Args:
        None

    Returns:
        Dict[str, object]: Dictionary containing options for AgGrid.
    """
    grid_options = {}
    
    # Format columns
    ColumnDefsList = [

        {
            "field": "Position", 
            "headerName": "Position",
            "pinned": "left",
            "editable": False,
        },
        {
            "field": 'AnnualSalaryUSD', 
            "headerName": 'Annual Salary in USD',
            "editable": True,
            "wrapHeaderText": True,
            "resizable": True,
            "autoHeaderHeight": True,
            "valueFormatter" : JsCode("""
                function(params) {
                    if (params.value === 0 || isNaN(params.value)) {
                        return '';
                    } else {
                        return "$" + parseFloat(params.value).toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
                    }
                }
                """)
        }]

    grid_options["columnDefs"] = ColumnDefsList

    # Set other options
    grid_options["defaultColDef"] = {
        "minWidth":10,
        "filter":True,
        "resizable":True,
        "sortable":True,
        }

    grid_options["alwaysShowHorizontalScroll"] = False
    grid_options["alwaysShowVerticalScroll"] = False

    return grid_options

def get_grid_options_verify_4() -> Dict[str, object]:
    """Generate options for an AgGrid.

    Args:
        None

    Returns:
        Dict[str, object]: Dictionary containing options for AgGrid.
    """
    grid_options = {}
    
    # Format columns
    ColumnDefsList = [

        {
            "field": "Variable", 
            "headerName": "Variable",
            "pinned": "left",
            "editable": False,
        },
        {
            "field": 'Value', 
            "headerName": 'Value',
            "editable": True,
            "wrapHeaderText": True,
            "resizable": True,
            "autoHeaderHeight": True,
            'cellRenderer': js_code3
        }]

    grid_options["columnDefs"] = ColumnDefsList

    # Set other options
    grid_options["defaultColDef"] = {
        "minWidth":10,
        "filter":True,
        "resizable":True,
        "sortable":True,
        }

    grid_options["alwaysShowHorizontalScroll"] = False
    grid_options["alwaysShowVerticalScroll"] = False

    return grid_options

def get_grid_options_verify_5(col_info: Dict) -> Dict[str, object]:
    """Generate options for an AgGrid.

    Args:
        col_info (Dict): Dictionary containing information about each column such as the field name, header name, and format.

    Returns:
        Dict[str, object]: Dictionary containing options for AgGrid.
    """
    grid_options = {}
    
    # Format columns
    ColumnDefsList = [
        {
            "field": "admin_name", 
            "headerName": "Region",
            "pinned": "left",
            "editable": False
        }]
        
    for col in col_info:
        temp = {
                "field": col['field'], 
                "headerName": col['headerName'],
                "editable": col['editable'],
                "wrapHeaderText": True,
                "resizable": True,
                "autoHeaderHeight": True,
                "precision": 0,
                "type": [
                    "numericColumn",
                    "numberColumnFilter",
                    ]
            }
        if col['format'] == "zero_decimal":
            temp["valueFormatter"] = JsCode("""
                    function(params) {
                        var value = parseFloat(params.value);
                        if (Number.isNaN(value) || value === 0) {
                            return '';
                        } else {
                            return (Math.round(value)).toLocaleString();
                        }
                    }
                """)
        elif col['format'] == "one_decimal":
            temp["valueFormatter"] = JsCode("""
                function(params) {
                    var value = parseFloat(params.value);
                    if (Number.isNaN(value) || value === 0) {
                        return '';
                    } else {
                        return parseFloat(params.value).toLocaleString('en-US', {minimumFractionDigits: 1, maximumFractionDigits: 1});
                    }
                }
            """)

        ColumnDefsList.append(temp)

    grid_options["columnDefs"] = ColumnDefsList

    # Set other options
    grid_options["defaultColDef"] = {
        "minWidth":5,
        "filter":True,
        "resizable":True,
        "sortable":True,
        }

    grid_options["alwaysShowHorizontalScroll"] = True
    grid_options["alwaysShowVerticalScroll"] = True

    return grid_options

def get_grid_options_verify_regions(col_info: Dict) -> Dict[str, object]:
    """Generate options for an AgGrid.

    Args:
        col_info (Dict): Dictionary containing information about each column such as the field name, header name, and format.

    Returns:
        Dict[str, object]: Dictionary containing options for AgGrid.
    """
    grid_options = {}
    
    # Format columns
    ColumnDefsList = [
        {
            "field": "admin_name", 
            "headerName": "Region",
            "pinned": "left",
            "editable": False
        }]
    
    for col in col_info:
        temp = {
                "field": col['field'], 
                "headerName": col['headerName'],
                "editable": True,
                "wrapHeaderText": True,
                "resizable": True,
                "autoHeaderHeight": True,
                "precision": 0,
                "type": [
                    "numericColumn",
                    "numberColumnFilter",
                    ]
            }
 
        if col['format'] == "zero_decimal":
            temp["valueFormatter"] = JsCode("""
                    function(params) {
                        var value = parseFloat(params.value);
                        if (Number.isNaN(value) || value === 0) {
                            return '';
                        } else {
                            return (Math.round(value)).toLocaleString();
                        }
                    }
                """)
        elif col['format'] == "one_decimal":
            temp["valueFormatter"] = JsCode("""
                function(params) {
                    var value = parseFloat(params.value);
                    if (Number.isNaN(value) || value === 0) {
                        return '';
                    } else {
                        return parseFloat(params.value).toLocaleString('en-US', {minimumFractionDigits: 1, maximumFractionDigits: 1});
                    }
                }
            """)

        ColumnDefsList.append(temp)

    grid_options["columnDefs"] = ColumnDefsList

    # Set other options
    grid_options["defaultColDef"] = {
        "minWidth":5,
        "filter":True,
        "resizable":True,
        "sortable":True,
        }

    grid_options["alwaysShowHorizontalScroll"] = True
    grid_options["alwaysShowVerticalScroll"] = True

    return grid_options

def get_grid_response(df: pd.DataFrame, grid_options: Dict[str, object]) -> AgGrid:
    """
    Generates an AgGrid object with the given DataFrame and grid options.

    Args:
        df: The pandas DataFrame to display in the grid.
        grid_options: A dictionary of options to customize the behavior and appearance of the grid.

    Returns:
        An AgGrid object with the specified DataFrame and options.
    """
    grid_response = AgGrid(
        df,
        # height=500,
        # width='100%',
        gridOptions=grid_options,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        data_return_mode=DataReturnMode.AS_INPUT,
        theme='streamlit',
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False
    )

    return grid_response
    

def add_machine_type(machine_type: str, ignore_cols: List[str], suffixes: List[str], session_state_key: str, scenario='baseline') -> None:        
    """
    Adds a new machine type to the Baseline or Add Machines data set

    Args:
        machine_type (str): The machine type to add.
        ignore_cols (List[str]): A list of columns to ignore when checking if the machine type
                            already exists in the data.
        suffixes (List[str]): A list of suffixes to add for the new machine type.
        session_state_key (str): The session state key for the data.
        scenario (str, optional): The scenario for which the data is to be updated. Defaults to 'baseline'.

    Raises:
        st.warning: If the machine type already exists in the data.
    """
    
    # Directly reference session state rather than passing to function
    # -- Arg values for on_click are defined at creation of widget (e.g. button)
    # -- Thus, if the user directly clicks Add Machine without making another change to the page
    # -- st.session_state.BaselineData_Edit is still an empty dataframe.  This direct reference to session_state
    # -- fixes the issue.
    if scenario == 'baseline':
        df = st.session_state.BaselineData_Edit
    else:
        df = st.session_state.AddMachinesData_Edit
    
    if any([machine_type in col for col in df.columns if col not in ignore_cols]):
        st.warning("The selected machine type is already in the table")

    else:
        for suffix in suffixes:
            df[machine_type + suffix] = np.nan

        if "HOLD" in df.columns:
            df.drop("HOLD", axis=1, inplace=True)
               
    st.session_state[session_state_key] = df

def remove_machine_type(machine_type: str, ignore_cols: List[str], suffixes: List[str], session_state_key: str, scenario='baseline') -> None:
    """
    Removes a machine type from the Baseline or Add Machines data set

    Args:
        machine_type (str): The machine type to add.
        ignore_cols (List[str]): A list of columns to ignore when checking if the machine type
                            already exists in the data.
        suffixes (List[str]): A list of suffixes to add for the new machine type.
        session_state_key (str): The session state key for the data.
        scenario (str, optional): The scenario for which the data is to be updated. Defaults to 'baseline'.

    Raises:
        st.warning: If the machine type does not exist in the data.
    """
    if scenario == 'baseline':
        df = st.session_state.BaselineData_Edit
    else:
        df = st.session_state.AddMachinesData_Edit
    
    # Check if machine type in table
    machine_list = [col.split('_')[0] for col in df.columns if col not in ignore_cols]

    if machine_type not in machine_list:
        st.warning('The selected machine type is not in the table')
    else:
        drop_list = [machine_type + suffix for suffix in suffixes]
        df.drop(drop_list, axis=1, inplace=True)

        # Add HOLD column, if applicable
        if df.empty:
            df['HOLD'] = 0

    st.session_state[session_state_key] = df

def calculate_total_capacity_target_disease(df: pd.DataFrame, work_days_per_year: int, disease: str) -> pd.DataFrame:
    """
    Calculates the total testing capacity for a target disease and separates machines into purchased and rented.
    
    Args:
        df (pd.DataFrame): The data containing information about the machines and their capabilities.
        work_days_per_year (int): The number of days in a year the machines are used.
        disease (str): The target disease for which the total capacity is to be calculated.

    Returns:
        pd.DataFrame: The updated data frame with calculated capacities for the target disease and separated machine quantities.
    """
    # Separate machines into purchased and rented
    df['MachineQuantity_Rented'].fillna(0, inplace=True)
    df['MachineQuantity_Owned'] = df['MachineQuantity'] - df['MachineQuantity_Rented']
    
    # Calculate total capacity
    df['TotalCapacity'] = df['MachineQuantity'] * df['Capacity_Day'] * work_days_per_year
    df['Capacity_Owned'] = df['MachineQuantity_Owned'] * df['Capacity_Day'] * work_days_per_year
    df['Capacity_Rented'] = df['MachineQuantity_Rented'] * df['Capacity_Day'] * work_days_per_year
    
    # Assume that if tests for the target disease are performed on a machine type at baseline, it overrides the default 
    #   assumption that target disease tests cannot be performed on that machine type.  
    df['Usable_' + disease] = np.where(df['TestsPerMachinePerYear_' + disease] > 0, 1, df['Usable_' + disease])

    df['TotalCapacity_' + disease] = df['TotalCapacity'] * df['Usable_' + disease]  
    df['Capacity_Owned_' + disease] = df['Capacity_Owned'] * df['Usable_' + disease] 
    df['Capacity_Rented_' + disease] = df['Capacity_Rented'] * df['Usable_' + disease] 
    
    # Filter df to only include machine types that can be used for the target disease
    df = df[df['Usable_' + disease] == 1]

    return df  

def calculate_average_variable_cost(df: pd.DataFrame, 
                                    col_capacity_owned: str, 
                                    col_capacity_rented: str, 
                                    col_cost_owned: str, 
                                    col_cost_rented: str) -> pd.DataFrame:
    """
    Calculates the average variable cost based on the capacity and cost of owned and rented machines. 
    
    The function computes products of owned and rented capacity and their respective costs, 
    then calculates the average variable cost for each administrative region. 
    It also fills missing values with the average calculated using data from all regions.

    Args:
        df (pd.DataFrame): The data containing information about the machines and their costs.
        col_capacity_owned (str): The column name in the dataframe for owned capacity.
        col_capacity_rented (str): The column name in the dataframe for rented capacity.
        col_cost_owned (str): The column name in the dataframe for cost of owned machines.
        col_cost_rented (str): The column name in the dataframe for cost of rented machines.

    Returns:
        pd.DataFrame: The updated data frame with calculated average variable cost for each  region.
    """
    df['product_owned'] = df[col_capacity_owned] * df[col_cost_owned]
    df['product_rented'] = df[col_capacity_rented] * df[col_cost_rented]
    
    grouped = df.groupby("admin_name").agg(
        {col_capacity_owned: "sum", col_capacity_rented: "sum", "product_owned": "sum", "product_rented": "sum"}
        )
    
    grouped['Capacity_Both'] = grouped[[col_capacity_owned, col_capacity_rented]].sum(axis=1)
    grouped['Avg_Variable_Cost'] = grouped[['product_owned', 'product_rented']].sum(axis=1) / grouped['Capacity_Both']

    avg_all = calculate_weighted_average(grouped, 'Avg_Variable_Cost', 'Capacity_Both')

    # Fill missing values with average calculated using data from all regions
    grouped['Avg_Variable_Cost'] = grouped['Avg_Variable_Cost'].fillna(avg_all)
    grouped = grouped[['Capacity_Both','Avg_Variable_Cost']]

    return grouped

def apply_inflation(df_main: pd.DataFrame, 
                    df_cpi: pd.DataFrame, 
                    df_exchange: pd.DataFrame, 
                    value_column: str, 
                    year_column: str, 
                    target_country: str, 
                    analysis_year: int, 
                    max_year_cpi: int, 
                    tradeable_dict: Dict[str, bool], 
                    already_inflated: Optional[List[bool]] = None) -> pd.DataFrame:
    """
    Applies inflation to monetary values in the main DataFrame.

    Args:
        df_main (pd.DataFrame): The main DataFrame with monetary values.
        df_cpi (pd.DataFrame): DataFrame containing consumer price index (CPI) data.
        df_exchange (pd.DataFrame): DataFrame containing exchange rate data.
        value_column (str): The name of the column in df_main containing the values to inflate.
        year_column (str): The name of the column in df_main containing the year of each value.
        target_country (str): The target country for the inflation.
        analysis_year (int): The year to inflate the values to.
        max_year_cpi (int): The maximum year for the consumer price index.
        tradeable_dict (Dict[str, bool]): A dictionary indicating whether each item is tradeable or not.
        already_inflated (List[bool], optional): A list indicating whether each item in df_main has already been inflated. 
                                                 If None, it is assumed that none of the items have been inflated.

    Returns:
        pd.DataFrame: The updated DataFrame with the values inflated.
    """
    df = df_main.loc[df_main[year_column].notnull()]
    df[year_column] = np.where(df[year_column] > float(max_year_cpi), float(max_year_cpi), df[year_column])

    original_val_col = []
    new_val_col = []
    for _, row in df.iterrows():
        if value_column in tradeable_dict.keys():
            var = value_column 
        elif row['Row'] in tradeable_dict.keys():
            var = row['Row']
        elif row['Col'] in tradeable_dict.keys():
            var = row['Col'] 
        else:
            var = None

        if var == None:
            original_val_col.append(np.nan)
            new_val_col.append(np.nan)
        
        else:
            # Case where var is tradeable.  US inflation applied to values already in USD
            if tradeable_dict[var]:
                cpi_base = df_cpi.at['USA',row[year_column]]
                cpi_current = df_cpi.at['USA',int(analysis_year)]
                inflation = cpi_current / cpi_base
                original_val_col.append(row[value_column])
                new_val_col.append(row[value_column] * inflation)

            # Case where var is nontradeable.  
            # -- Values in USD converted back to LCU using the exchange rate at the year of costing
            # -- Then local inflation applied
            # -- Then values converted back to USD using the exchange rate at the year of analysis
            # -- The only cost value for 'ALL' countries is cost per km.  Treat as nontradeable using cpi and exchange rate values for the target country 
            else:
                exchange_base = df_exchange.at[target_country,row[year_column]]
                exchange_current = df_exchange.at[target_country,int(analysis_year)]
                cpi_base = df_cpi.at[target_country,row[year_column]]
                cpi_current = df_cpi.at[target_country,int(analysis_year)]
                inflation = cpi_current / cpi_base
                original_val_col.append(row[value_column])
                new_val_col.append((row[value_column] * exchange_base * inflation) / exchange_current)

    df.drop(columns=[value_column], inplace=True)
    
    if already_inflated is not None:
        new_val_col_final = [original_val_col[i] if already_inflated[i] else new_val_col[i] for i in range(len(new_val_col))]
        df[value_column] = new_val_col_final
    else:
        df[value_column] = new_val_col
    
    df_main.update(df)

    return df_main

def create_testing_rate_table(df: pd.DataFrame, 
                              rate_denominator: Union[int, float], 
                              HIV_EID_tests_per_child: Union[int, float], 
                              HIV_VL_tests_per_newly_diagnosed: Union[int, float], 
                              HIV_VL_tests_per_previously_diagnosed: Union[int, float], 
                              TB_positivity_rate: Union[int, float]) -> pd.DataFrame:
    """
    Calculates HIV and TB testing need in terms of tests per year.

    Args:
        df (pd.DataFrame): The original DataFrame with necessary data.
        rate_denominator (int, float): The denominator used to calculate incidence and prevalence rates.
        HIV_EID_tests_per_child (int, float): The number of EID tests performed per HIV-exposed child
        HIV_VL_tests_per_newly_diagnosed (int, float): The number of HIV VL tests performed per newly diagnosed individual (within 1 year)
        HIV_VL_tests_per_previously_diagnosed (int, float): The number of HIV VL tests performed per previously diagnosed individual (more than 1 year ago)
        TB_positivity_rate (int, float): The MolDx TB testing positivity rate

    Returns:
        pd.DataFrame: The DataFrame with added columns for HIV and TB testing rates.
    """
    # HIV
    # - Subnational data from Spectrum Naomi is only for ages 15+.  Still missing VL tests for ages 0-14.
    df['HIV_VL_Tests'] = ((df['Incidence_Rate_HIV'] / rate_denominator) * df['Population'] * HIV_VL_tests_per_newly_diagnosed) + \
    (((df['Prevalence_Rate_HIV'] - df['Incidence_Rate_HIV']) / rate_denominator) * df['Population'] * HIV_VL_tests_per_previously_diagnosed)


    df['HIV_EID_Tests'] = (df['HIV_Exposed_Births_Rate_National']/ rate_denominator) * df['Population'] * HIV_EID_tests_per_child
    df['TestingNeed_HIV'] = df['HIV_VL_Tests'] + df['HIV_EID_Tests']

    # TB
    df['TestingNeed_TB'] = ((df['Incidence_Rate_TB'] / TB_positivity_rate) / rate_denominator) * df['Population']

    return df

@st.cache_data(max_entries=1)
def update_dno_data(dno_input: pd.DataFrame, 
                    target_country: str, 
                    analysis_year: Optional[int] = None, 
                    cpi: Optional[pd.DataFrame] = None, 
                    exchange: Optional[pd.DataFrame] = None, 
                    inflation: bool = False, 
                    max_year_cpi: Optional[int] = None, 
                    tradeable_dict: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
    """
    Updates default data using data from DNO analyses.  Changes are written directly to session state variables.

    Args:
        dno_input (pd.DataFrame): The original DNO data DataFrame.
        target_country (str): ISO3 code of the target country.
        analysis_year (int, optional): The year for the analysis.
        cpi (pd.DataFrame, optional): DataFrame with Consumer Price Index data.
        exchange (pd.DataFrame, optional): DataFrame with exchange rate data.
        inflation (bool, optional): Whether or not to apply inflation. Default is False.
        max_year_cpi (int, optional): The maximum year for CPI data.
        tradeable_dict (Dict[str, bool], optional): A dictionary mapping inputs to whether they are tradeable or not.

    Returns:
        None
    """
    DNOInput_Target = dno_input.loc[(dno_input['iso3'] == target_country) | (dno_input['iso3'] == 'ALL')].copy()

    # Determine which inputs are defined for the target country and which for ALL countries
    TargetList = DNOInput_Target.loc[DNOInput_Target['iso3'] == target_country, 'Row'].unique().tolist()
    AllList = DNOInput_Target.loc[DNOInput_Target['iso3'] == 'ALL', 'Row'].unique().tolist()
    ApplyALL = [i for i in AllList if i not in TargetList]

    # Apply inflation to cost data
    if len(DNOInput_Target['Year'].value_counts()) != 0 and inflation == True:
        DNOInput_Target = apply_inflation(DNOInput_Target, cpi, exchange, 'Val', 'Year', target_country, analysis_year, max_year_cpi, tradeable_dict)

    # Replace data
    for _, row in DNOInput_Target.iterrows():
        Table = row['Table']
        r = row['Row']
        v = row['Val']
        
        ## ...in tables
        if not pd.isnull(Table):
            c = row['Col']

            # Add row to table if not already in index
            if r not in st.session_state[Table].index:
                st.session_state[Table].loc[r, :] = np.nan
            
            st.session_state[Table].at[r, c] = v

        # ...in specific variables 
        else:
            # Only use 'ALL' values if variable not already set using target-country-specific value
            if row['iso3'] == 'ALL':
                if r in ApplyALL:
                    st.session_state[r] = v
            
            else:
                st.session_state[r] = v

@st.cache_data(max_entries=1)
def apply_inflation_machine_types(df: pd.DataFrame, 
                                  df_cpi: pd.DataFrame, 
                                  df_exchange: pd.DataFrame, 
                                  target_country: str, 
                                  analysis_year: int, 
                                  max_year_cpi: int, 
                                  tradeable_dict: dict, 
                                  cost_cols: List[str], 
                                  df_orig: pd.DataFrame) -> pd.DataFrame:
    """
    Apply inflation adjustment to the cost columns of the machine types table

    Args:
        df (pd.DataFrame): The DataFrame containing machine data.
        df_cpi (pd.DataFrame): The DataFrame containing CPI data.
        df_exchange (pd.DataFrame): The DataFrame containing exchange rate data.
        target_country (str): The ISO3 code of the target country.
        analysis_year (int): The year of the analysis.
        max_year_cpi (int): The maximum year for which CPI data is available.
        tradeable_dict (dict): A dictionary mapping inputs to whether they are tradeable or not.
        cost_cols (List[str]): The list of cost column names in the DataFrame.
        df_orig (pd.DataFrame): The original DataFrame before any adjustments.

    Returns:
        pd.DataFrame: The DataFrame with cost columns adjusted for inflation.
    """
    for col in cost_cols:
        already_inflated = []
        for i in range(len(df[col].index)):
            if i < len(df_orig[col].index):
                already_inflated.append(df[col][i] != df_orig[col][i])
            else:
                already_inflated.append(True)

        df = apply_inflation(df, df_cpi, df_exchange, col, 'Cost_Year', target_country, analysis_year, max_year_cpi, tradeable_dict, already_inflated)
    
    return df

def calculate_total_tests_for_disease(df: pd.DataFrame, disease: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculates the total number of tests performed for the target disease across all machine types

    Args:
        df (pd.DataFrame): The DataFrame containing the number of machines and tests per machine per year.
        disease (str): The name of the disease.

    Returns:
        Tuple[pd.DataFrame, List[str]]: The modified DataFrame and a list of column names corresponding to the number of tests per machine per year.
    """
    # Pull relevant columns
    num_machine_cols = []
    num_test_cols = []
    for col in df.columns:
        if 'NumMachinesTotal' in col:
            num_machine_cols.append(col)
        elif f'TestsPerMachinePerYear_{disease}' in col:
            num_test_cols.append(col)

    # Multiply relevant columns to get total number of tests performed for each machine type
    ProductCols = []
    i = 1
    for machine_col, test_col in zip(num_machine_cols, num_test_cols):
        df['Product' + str(i)] = df[machine_col] * df[test_col]
        ProductCols.append('Product' + str(i))
        i += 1
    
    # Sum products across all machine types to get total tests per year for the target disease for each region across all machine types
    df[f'TotalTestsPerYear_{disease}'] = df[ProductCols].sum(axis=1)
    df.drop(columns=ProductCols, inplace=True)

    return df, num_test_cols

def adjust_routes_per_machine(df: pd.DataFrame, df_regions: pd.DataFrame, disease: str) -> pd.DataFrame:
    """
    Adjust number of routes per machine to account for maximum number of routes per region

    Args:
        df (pd.DataFrame): The DataFrame containing the number of machines and tests per machine per year.
        df_regions (pd.DataFrame): The DataFrame containing regional data including routes per machine and estimated health facilities.
        disease (str): The name of the disease.

    Returns:
        pd.DataFrame: The modified regional DataFrame with updated 'RoutesPerMachine' and 'DistancePerMachine' values.
    """
    df_copy = df.copy()
    ignore_cols = ['admin_name']
    df_copy[f'TotalMachinesTesting{disease}'] = 0
    for col in df_copy.columns:
        if col not in ignore_cols and 'NumMachinesTotal' in col:
            machine_type = col.split('_')[0]
            df_copy[f'TotalMachinesTesting{disease}'] = np.where(
                df_copy[f'{machine_type}_TestsPerMachinePerYear_{disease}'] > 0,
                df_copy[[f'TotalMachinesTesting{disease}', col]].sum(axis=1),
                df_copy[f'TotalMachinesTesting{disease}']
            )
             
    df_copy = df_copy.merge(df_regions[['admin_name', f'RoutesPerMachine_{disease}', 'EstimatedHealthFacilities']] , on='admin_name', how='left')
    df_copy['NumRoutes'] = df_copy[f'TotalMachinesTesting{disease}'] * df_copy[f'RoutesPerMachine_{disease}'] 
    df_copy['RoutesPerMachine_Multiplier'] = np.where(df_copy['NumRoutes'] > df_copy['EstimatedHealthFacilities'], df_copy['EstimatedHealthFacilities'] / df_copy['NumRoutes'], 1)

    df_regions_copy = df_regions.copy()
    df_regions_copy[f'RoutesPerMachine_{disease}'] = df_regions_copy[f'RoutesPerMachine_{disease}'] * df_copy['RoutesPerMachine_Multiplier']
    df_regions_copy[f'DistancePerMachine_{disease}'] = df_regions_copy[f'RoutesPerMachine_{disease}'] * df_regions_copy[f'DistancePerRoute_Raw_{disease}']

    return df_regions_copy

def sort_regions(regions: List[str], df_regions: pd.DataFrame, disease: str) -> List[str]:
    """
    Sorts a list of regions first by distance per route (from shortest to longest) and then alphabetically.

    Args:
        regions (List[str]): The list of regions to be sorted.
        df_regions (pd.DataFrame): DataFrame containing regional data including distance per route for a given disease.
        disease (str): The name of the disease.

    Returns:
        List[str]: A sorted list of regions.
    """
    df = pd.DataFrame(regions, columns=['admin_name'])
    df = df.merge(df_regions[['admin_name', f'DistancePerRoute_Raw_{disease}']], on='admin_name', how='left')
    df.sort_values(by=[f'DistancePerRoute_Raw_{disease}', 'admin_name'], ascending=[True, True], inplace=True)
    regions_sorted = df['admin_name'].tolist()

    return regions_sorted

def update_intra_regional_log(Log_IntraRegional: Dict[str, Dict[str, float]], Region: str, UnmetNeed: float, NewTests: float, NumTests_Baseline: float, NumRoutes: float, NewRoute_Distance: float, NumRoutes_Baseline: float, MaxRoutes: float, Disease: str) -> Dict[str, Dict[str, float]]:
    """
    Updates the intra-regional log with information about new routes created and tests performed

    Args:
        Log_IntraRegional (Dict[str, Dict[str, float]]): The existing intra-regional log.
        Region (str): The name of the region.
        UnmetNeed (float): The unmet testing need before transport, measured in tests per year.
        NewTests (float): The number of new tests performed by paying for sample transport and processing, measured in tests per year.
        NumTests_Baseline (float): The baseline number of tests performed in the region per year.
        NumRoutes (float): The number of new routes created in the region.
        NewRoute_Distance (float): The distance of each new route, measured in kilometers.
        NumRoutes_Baseline (float): The baseline number of routes in the region.
        MaxRoutes (float): The maximum number of disease transport routes possible in the region.
        Disease (str): The target disease

    Returns:
        Dict[str, Dict[str, float]]: The updated intra-regional log.
    """
    Log_IntraRegional[Region] = {
        'Unmet Testing Need Before Transport (Tests per Year)' : UnmetNeed,
        'New Tests Performed by Paying for Sample Transport and Processing (Tests per Year)' : NewTests,
        'Total Tests Performed in Region per Year' : NumTests_Baseline + NewTests,
        'New Routes Created in Region' : NumRoutes,
        'Distance of Each New Route (km)' : NewRoute_Distance,
        'Total Routes in Region After the Creation of New Routes' : NumRoutes_Baseline + NumRoutes,
        f'Maximum Number of {Disease} Transport Routes Possible in Region' : MaxRoutes                        
        }
    
    return Log_IntraRegional

# JS code used for formatting AgGrid tables
js_code1 = JsCode("""
function customNumberFormat(params) {
    var value = params.value;
    var rowIndex = params.node.rowIndex;
    switch (rowIndex) {
        case 0:
            return parseFloat(value).toFixed(0);
        case 1:
            return (parseFloat(value) * 100).toFixed(0) + "%";
        case 2:
            return parseFloat(value).toFixed(0);
        case 3:
            return parseFloat(value).toFixed(0);
        case 4:
            return parseFloat(value).toFixed(0);
        case 5:
            return "$" + parseFloat(value).toFixed(2);
        case 6:
            return parseFloat(value).toFixed(1);
        default:
            return value;
    }
}
""")

js_code2 = JsCode("""
function customNumberFormat(params) {
    var value = params.value;
    var rowIndex = params.node.rowIndex;
    switch (rowIndex) {
        case 0:
            return parseFloat(value).toFixed(0);
    }
}
""")

js_code3 = JsCode("""
function customNumberFormat(params) {
    var value = params.value;
    var variable = params.data.Variable;
    
    switch (variable) {
        case 'Number of MolDx HIV EID Tests Per Child Exposed at Birth':
            return parseFloat(value).toFixed(0);
        case 'Number of HIV Viral Load Tests Per Year per for a Newly Identified Case': 
            return parseFloat(value).toFixed(0);
        case 'Number of HIV Viral Load Tests Per Year per for a Previously Identified Case': 
            return parseFloat(value).toFixed(0);
        case 'TB Positivity Rate: Positive MolDx TB Diagnoses as a Proportion of total MolDx TB Diagnostic Tests Performed': 
            return (parseFloat(value) * 100).toFixed(1) + "%";
        default:
            return value;
        }
    }
    """)