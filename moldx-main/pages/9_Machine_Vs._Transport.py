import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
import pandas as pd
import numpy as np
from itertools import product
import plotly_express as px
import plotly.graph_objects as go
import functions as f

st.markdown(st.session_state.ReducePadding, unsafe_allow_html=True)

### Assign session state data to dataframes and variables matching subsequent code
MachineTypes = st.session_state.MachineTypes
DiseaseBurden = st.session_state.DiseaseBurden
EligibleCountries = st.session_state.EligibleCountries
DNOInput = st.session_state.DNOInput
CPI = st.session_state.CPI
DNOTransport1 = st.session_state.DNOTransport1
DNOTransport2 = st.session_state.DNOTransport2
StaffCost = st.session_state.StaffCost
DistanceTime = st.session_state.DistanceTime
RegionsTransportData = st.session_state.RegionsTransportData
RegionsSelected = st.session_state.RegionsSelected

DistanceThreshold = st.session_state.DistanceThreshold
CostPerKm = st.session_state.CostPerKm
WorkDaysPerYear = st.session_state.WorkDaysPerYear
TransportWeeksPerYear = st.session_state.TransportWeeksPerYear
TransportFreqPerWeek = st.session_state.TransportFreqPerWeek
AnalysisYear = st.session_state.AnalysisYear

### Define target country and disease
TargetCountry = st.session_state.TargetCountry
TargetCountryISO2 = EligibleCountries.at[TargetCountry,'iso2']
Disease = st.session_state.Disease
DiseaseList = st.session_state.DiseaseList

### Modify cost per km to account for round-trip sample transport.  All distances used in the model are one-way distances.
### --Simpler than modifying the number of kms
if st.session_state.RoundTrip == True:
    CostPerKm *= 2

### Subset data sets based on TargetCountry and regions selected
MachinesByRegion_Baseline_Target = st.session_state.BaselineData
MachinesByRegion_Scenario_Target = st.session_state.AddMachinesData
DiseaseBurden_Target = DiseaseBurden.loc[(DiseaseBurden['admin_name'].isin(st.session_state.RegionsSelected)) & (DiseaseBurden['iso3'] == TargetCountry)]

MachinesByRegion_Baseline_Target.replace('None', np.nan, inplace=True)
IGNORE_COLS = ['admin_name']
DataSets = [MachinesByRegion_Baseline_Target, MachinesByRegion_Scenario_Target]
for df in DataSets:
    for col in df.columns:
        if col not in IGNORE_COLS:
            df[col] = df[col].astype("float64")

### Adjust testing need, if necessary
## Check if testing need is below the number of tests performed at baseline.  If yes, replace testing need value with number of tests performed at baseline.
TestingNeedAdjusted = False
MachinesByRegion_Baseline_Target, _ = f.calculate_total_tests_for_disease(MachinesByRegion_Baseline_Target, Disease)
DiseaseBurden_Target = DiseaseBurden_Target.merge(MachinesByRegion_Baseline_Target[['admin_name', f'TotalTestsPerYear_{Disease}']], on='admin_name', how='left')
DiseaseBurden_Target[f'TestingNeed_{Disease}_Original'] = DiseaseBurden_Target[f'TestingNeed_{Disease}']
DiseaseBurden_Target['TestingNeed_' + Disease] = DiseaseBurden_Target[[f'TotalTestsPerYear_{Disease}', 'TestingNeed_' + Disease]].max(axis=1)

if (DiseaseBurden_Target['TestingNeed_' + Disease] != DiseaseBurden_Target[f'TestingNeed_{Disease}_Original']).any():
    TestingNeedAdjusted = True

RegionsAdjusted_Need = DiseaseBurden_Target.loc[DiseaseBurden_Target['TestingNeed_' + Disease] != DiseaseBurden_Target[f'TestingNeed_{Disease}_Original'], 'admin_name'].tolist()

DiseaseBurden_Target = DiseaseBurden_Target.drop(columns=[f'TotalTestsPerYear_{Disease}', f'TestingNeed_{Disease}_Original'])
MachinesByRegion_Baseline_Target.drop(columns=[f'TotalTestsPerYear_{Disease}'], inplace=True)

### Adjust number of routes per machine to account for maximum number of routes per region, if necessary
# -- EstimatedHealthFacilities is a proxy for the maximum number of routes per region
RegionsTransportData_Mod = f.adjust_routes_per_machine(MachinesByRegion_Baseline_Target, RegionsTransportData, Disease)

## Baseline analysis
Master_Baseline = f.stack_table(st.session_state.BaselineData, st.session_state.DiseaseList) 
Master_Baseline = Master_Baseline.merge(st.session_state.MachineTypes, left_on='Machine', right_index=True, how='left')
Master_Baseline = Master_Baseline.merge(RegionsTransportData_Mod, on='admin_name', how='left')
Master_Baseline = f.calculate_total_capacity_target_disease(Master_Baseline, WorkDaysPerYear, Disease)
Master_Baseline = f.add_calculated_columns(Master_Baseline, DNOTransport2, StaffCost,  DistanceThreshold, TargetCountry, st.session_state[f'TestedOnSite_{Disease}'], Disease, DiseaseList, CostPerKm, TransportFreqPerWeek, TransportWeeksPerYear)
Stats_Baseline = f.calculate_statistics(Master_Baseline, DiseaseBurden_Target, Disease, DiseaseList)

### Run scenario (add machines and/or rental)
## Add TestsPerMachinePerYear columns to the Scenario table
# Tests per machine columns already present in the Baseline table
JoinCols = [i for i in list(MachinesByRegion_Baseline_Target.columns) if 'TestsPerMachinePerYear' in i]
JoinCols.append('admin_name')
MachinesByRegion_Scenario_Target = MachinesByRegion_Scenario_Target.merge(MachinesByRegion_Baseline_Target[JoinCols], on='admin_name')

# Tests per machine column not present in the Baseline table
IGNORE_COLS = ['admin_name']
MachineList_Scenario = list(set([i.split('_')[0] for i in list(MachinesByRegion_Scenario_Target.columns) if i not in IGNORE_COLS]))
for m in MachineList_Scenario:
    for d in DiseaseList:
        if m + '_TestsPerMachinePerYear_' + d not in list(MachinesByRegion_Scenario_Target.columns):
            MachinesByRegion_Scenario_Target[m + '_TestsPerMachinePerYear_' + d] = np.nan

### Fill values of TestsPerMachinePerYear_[Disease] for machine types not present in a given region at Baseline
MachineList_Baseline = list(set([i.split('_')[0] for i in list(MachinesByRegion_Baseline_Target.columns) if i not in IGNORE_COLS]))
RegionMachineType_Baseline =list(set([i + j for i,j in zip(list(Master_Baseline['admin_name']), list(Master_Baseline['Machine']))]))

AverageDict = {}
for d in DiseaseList:

    ## Calculate the weighted average number of tests per machine for all machines of a given type in all regions
    Avg_SameType_AllRegions_Dict = {}
    for machine_type in MachineList_Baseline:
        # For this calculation, machines without tests for the target disease still contribute to the denominator
        Avg_SameType_AllRegions = f.calculate_weighted_average(MachinesByRegion_Baseline_Target, machine_type + '_TestsPerMachinePerYear_' + d, machine_type + '_NumMachinesTotal') 
        Avg_SameType_AllRegions_Dict[machine_type] = Avg_SameType_AllRegions

    AverageDict[d + 'Avg_SameType_AllRegions'] = Avg_SameType_AllRegions_Dict

    ## Calculate the average number of tests per machine for all machines in each region, regardless of machine type  
    df_AvgTests_Rows = f.calculate_weighted_average_rows(MachinesByRegion_Baseline_Target, 'TestsPerMachinePerYear_' + d, 'NumMachinesTotal')
    AverageDict[d + 'df_AvgTests_Rows'] = df_AvgTests_Rows

    ## Calculate the average utilization of machines in each region, regardless of machine type.  
    df_AvgUtilization_Rows = f.calculate_weighted_average_utilization(MachinesByRegion_Baseline_Target, MachineTypes, 'TestsPerMachinePerYear_' + d, 'NumMachinesTotal', 'row', WorkDaysPerYear)
    AverageDict[d + 'df_AvgUtilization_Rows'] = df_AvgUtilization_Rows

    ## Calculate the weighted average number of tests across all machine types and regions
    Avg_List = []
    Weight_List = []
    for machine_type, tests in Avg_SameType_AllRegions_Dict.items():
        if np.isnan(tests):
            Avg_List.append(0)
        else:
            Avg_List.append(tests)
        Weight_List.append(MachinesByRegion_Baseline_Target[machine_type + '_NumMachinesTotal'].sum())

    Avg_AllTypes_AllRegions = np.average(Avg_List, weights=Weight_List)
    AverageDict[d + 'Avg_AllTypes_AllRegions'] = Avg_AllTypes_AllRegions
    
    ## Calculate average utilization of machines across all regions, regardless of machine type. 
    AvgUtilization_Cols = f.calculate_weighted_average_utilization(MachinesByRegion_Baseline_Target, MachineTypes, 'TestsPerMachinePerYear_' + d, 'NumMachinesTotal', 'col', WorkDaysPerYear)
    AverageDict[d + 'AvgUtilization_Cols'] = AvgUtilization_Cols

## Apply logic to fill in missing values in Scenario table.  All averages are weighted by the number of machines.
# Assumes that tests for diseases other than the target disease are not performed on new [Region]-[MachineType] combinations.
# New machines of a [Region]-[MachineType] on which tests for multiple diseases are performed at baseline will also perform those same tests and quantities
TargetCol = 'TestsPerMachinePerYear_'

for col in list(MachinesByRegion_Scenario_Target.columns):
    if TargetCol in col:
        machine_type = col.split('_')[0] 
        d = col.split('_')[2]
        for i in range(len(MachinesByRegion_Scenario_Target)):
            
            region = MachinesByRegion_Scenario_Target.iloc[i, MachinesByRegion_Scenario_Target.columns.get_loc('admin_name')]

            #Only apply averages to [Region]-[MachineType] combinations not present at baseline
            if region + machine_type not in RegionMachineType_Baseline and not np.isnan(MachinesByRegion_Scenario_Target.iloc[i, MachinesByRegion_Scenario_Target.columns.get_loc(machine_type + '_NumMachinesTotal')]):
                
                # Try/Except block to handle case where there are no machines of a given type in any region at baseline
                try:
                    Tests_AllRegions_OneMachine = AverageDict[d + 'Avg_SameType_AllRegions'][machine_type]
                except:
                    Tests_AllRegions_OneMachine = np.nan
                
                Tests_OneRegion_AllMachines = AverageDict[d + 'df_AvgTests_Rows'].iloc[i, AverageDict[d + 'df_AvgTests_Rows'].columns.get_loc('Avg')]
                UtilizationRate = AverageDict[d + 'df_AvgUtilization_Rows'].iloc[i, AverageDict[d + 'df_AvgUtilization_Rows'].columns.get_loc('Avg')]
                MachineCapacity = MachineTypes.loc[machine_type,'Capacity_Day'] * WorkDaysPerYear

                #Fill with an average calculated using the same machine type in other regions
                if not np.isnan(Tests_AllRegions_OneMachine):
                    val = Tests_AllRegions_OneMachine
                
                #Fill with an average calculated using all machine types in the same region.
                #Select the smaller of: average based on number of tests and average based on utilization rate applied to new machine capacity
                elif not np.isnan(Tests_OneRegion_AllMachines) and not np.isnan(UtilizationRate):
                    Tests_AvgUtilization = MachineCapacity * UtilizationRate
                    if Tests_OneRegion_AllMachines < Tests_AvgUtilization:
                        val = Tests_OneRegion_AllMachines
                    else:
                        val = Tests_AvgUtilization
                
                #Fill with an average calculated using all machine types across all regions
                #Select the smaller of: average based on number of tests and average based on utilization rate applied to new machine capacity
                else:
                    Tests_AvgUtilization = MachineCapacity * AverageDict[d + 'AvgUtilization_Cols']
                    if AverageDict[d + 'Avg_AllTypes_AllRegions'] < Tests_AvgUtilization:
                        val = AverageDict[d + 'Avg_AllTypes_AllRegions']
                    else:
                        val = Tests_AvgUtilization

                MachinesByRegion_Scenario_Target.iloc[i, MachinesByRegion_Scenario_Target.columns.get_loc(col)] = val

### Adjust tests per machine per year, if necessary
## Use testing need to adjust the number of tests performed in the Add Machines scenario so that total tests do not exceed need
MachinesByRegion_Scenario_Target, NumTestCols = f.calculate_total_tests_for_disease(MachinesByRegion_Scenario_Target, Disease)
MachinesByRegion_Scenario_Target = MachinesByRegion_Scenario_Target.merge(DiseaseBurden_Target[['admin_name','TestingNeed_' + Disease]], on='admin_name', how='left')

NumTests_Adjusted = False
MachinesByRegion_Scenario_Target['Testing_Ratio'] = MachinesByRegion_Scenario_Target[f'TotalTestsPerYear_{Disease}'] / MachinesByRegion_Scenario_Target['TestingNeed_' + Disease]
MachinesByRegion_Scenario_Target['Testing_Multiplier'] = np.where(MachinesByRegion_Scenario_Target['Testing_Ratio'] > 1, 1 / MachinesByRegion_Scenario_Target['Testing_Ratio'], 1)
for col in NumTestCols:
     MachinesByRegion_Scenario_Target[col] = MachinesByRegion_Scenario_Target[col] * MachinesByRegion_Scenario_Target['Testing_Multiplier']

# Check if at least one value in 'Testing_Ratio' is greater than 1
if (MachinesByRegion_Scenario_Target['Testing_Ratio'] > 1).any():
    NumTests_Adjusted = True

RegionsAdjusted_Tests = MachinesByRegion_Scenario_Target.loc[MachinesByRegion_Scenario_Target['Testing_Ratio'] > 1, 'admin_name'].tolist()

# Drop new calculated columns
MachinesByRegion_Scenario_Target.drop([f'TotalTestsPerYear_{Disease}', 'TestingNeed_' + Disease, 'Testing_Ratio', 'Testing_Multiplier'], axis=1, inplace=True)

### Adjust number of routes per machine to account for maximum number of routes per region, if necessary
# -- EstimatedHealthFacilities is a proxy for the maximum number of routes per region
RegionsTransportData_Mod = f.adjust_routes_per_machine(MachinesByRegion_Scenario_Target, RegionsTransportData, Disease)

### Add Machines / Rental analysis
Master_Scenario = f.stack_table(MachinesByRegion_Scenario_Target, DiseaseList) 
Master_Scenario = Master_Scenario.merge(MachineTypes, left_on='Machine', right_index=True, how='left')
Master_Scenario = Master_Scenario.merge(RegionsTransportData_Mod, on='admin_name', how='left')
Master_Scenario = f.calculate_total_capacity_target_disease(Master_Scenario, WorkDaysPerYear, Disease)
Master_Scenario = f.add_calculated_columns(Master_Scenario, DNOTransport2, StaffCost, DistanceThreshold, TargetCountry, st.session_state[f'TestedOnSite_{Disease}'], Disease, DiseaseList, CostPerKm, TransportFreqPerWeek, TransportWeeksPerYear)
Stats_Scenario = f.calculate_statistics(Master_Scenario, DiseaseBurden_Target, Disease, DiseaseList)

### Calculate incremental cost and set as the budget
Budget = Stats_Scenario['TotalCost'] - Stats_Baseline['TotalCost']
if Budget > 0:
    budget_depleted=False
else:
    budget_depleted=True

### Apply incremental cost to sample transport toward underutilized machines
## Calculate available capacity in each region
# Remove capacity utilized for all diseases, not just the target disease
# Only count available capacity applicable to the target disease.  
Master_Baseline['UtilizedCapacity'] = Master_Baseline[[i for i in list(Master_Baseline.columns) if 'UtilizedCapacity' in i]].sum(axis=1)
Master_Baseline['AvailableCapacity_' + Disease] = (Master_Baseline[f'TotalCapacity_{Disease}'] - Master_Baseline['UtilizedCapacity'])

### Determine the regions in which machines were added in the Add Machines scenario and create a sorted list
Temp = Master_Baseline[['admin_name', 'Machine', 'MachineQuantity' ]].merge(Master_Scenario[['admin_name', 'Machine', 'MachineQuantity']], on=['admin_name', 'Machine'], how='outer', suffixes=('_Baseline', '_Scenario'))
Regions_MachinesAdded = Temp.loc[(Temp['MachineQuantity_Scenario'] > Temp['MachineQuantity_Baseline']) | np.isnan(Temp['MachineQuantity_Baseline']), 'admin_name'].unique().tolist()
Regions_MachinesAdded_Sorted = f.sort_regions(Regions_MachinesAdded, RegionsTransportData, Disease)

Regions_MachinesNotAdded = list(set(RegionsSelected) - set(Regions_MachinesAdded)) # Use RegionsSelected to ensure selected regions that do not have machines at baseline or to which machines are not added still appear in the sort list
Regions_MachinesNotAdded_Sorted = f.sort_regions(Regions_MachinesNotAdded, RegionsTransportData, Disease)

Regions_Sorted = Regions_MachinesAdded_Sorted + Regions_MachinesNotAdded_Sorted

## Clean up table and group by region
df = Master_Baseline.copy()

# For later calculation of total number of routes, count only machines for which some tests of the target disease are performed at baseline
df['MachineQuantity_ForRouteCalc']=np.where(df[f'TestsPerMachinePerYear_{Disease}'] > 0, df['MachineQuantity'], 0)

KeepCols = ['admin_name', 'TotalCapacity', 'UtilizedCapacity', 
            'TotalCapacity_' + Disease, 'AvailableCapacity_' + Disease,
            'Capacity_Owned_' + Disease, 'Capacity_Rented_' + Disease,
            'MachineCost_Total_' + Disease, 'StaffCost_Total_' + Disease, 
            'VariableCost_Total_' + Disease, 'TransportCost_Total_' + Disease, 
            'TotalCost_' + Disease,
            'Processing_Cost_PerSampleVariable_' + Disease,
            f'Processing_Cost_PerSampleVariable_{Disease}_Rented',
            'TestsPerRoutePerYear_' + Disease,
            f'RoutesPerMachine_{Disease}',
            'MachineQuantity_ForRouteCalc',
            f'DistancePerRoute_Raw_{Disease}']

for d in DiseaseList:
    KeepCols.append('UtilizedCapacity_' + d)

df = df[KeepCols]
df.rename({'UtilizedCapacity' : 'UtilizedCapacity_Baseline',
           'UtilizedCapacity_' + Disease : 'UtilizedCapacity_' + Disease + '_Baseline',
           'VariableCost_Total_' + Disease: 'VariableCost_Total_' + Disease + '_Baseline',
           'TransportCost_Total_' + Disease: 'TransportCost_Total_' + Disease + '_Baseline',
           'TotalCost_' + Disease : 'TotalCost_' + Disease + '_Baseline', 
           }, 
           axis=1, inplace=True
    )

## Calculate average sample processing cost for each region, weighted by the available capacity of each machine
## -Weights are the available capacity of each machine because we assume new samples processed will be allocated to machines proportional to the available capacity of each.
df_Avg = f.calculate_average_variable_cost(df, 'Capacity_Owned_' + Disease, 'Capacity_Rented_' + Disease, 'Processing_Cost_PerSampleVariable_' + Disease, f'Processing_Cost_PerSampleVariable_{Disease}_Rented')

## Group by region
SumList = df.columns.tolist()
SumList.remove('admin_name')
MeanList = ['Processing_Cost_PerSampleVariable_' + Disease,
            f'Processing_Cost_PerSampleVariable_{Disease}_Rented',
            'TestsPerRoutePerYear_' + Disease,
            f'RoutesPerMachine_{Disease}',
            f'DistancePerRoute_Raw_{Disease}']
for item in MeanList:
    SumList.remove(item)

func_dict = {n: 'sum' for n in SumList}
func_dict['admin_name'] = 'first' #'first' keeps the first instance of a cell value when aggregating
func_dict.update({n: 'mean' for n in MeanList})
df = df.groupby(['admin_name'], as_index=False).agg(func_dict) #func_dict specifies the function to use for each column when grouping

df = df.merge(df_Avg, on='admin_name', how='left')
df.drop(columns=['Capacity_Both'], inplace=True)

## Add other columns to df
df = df.merge(DiseaseBurden_Target[['admin_name','TestingNeed_' + Disease]], on='admin_name', how='left')
df = df.merge(RegionsTransportData[['admin_name','EstimatedHealthFacilities']], on='admin_name', how='left') #Unlike DiseaseBurden_Target, RegionsTransportData contains user-adjusted values of EstimatedHealthFacilities
df['UnmetNeed'] = np.where(df['TestingNeed_' + Disease] > df['UtilizedCapacity_' + Disease + '_Baseline'], df['TestingNeed_' + Disease] - df['UtilizedCapacity_' + Disease + '_Baseline'], 0) 
df['NumRoutes_Baseline'] = df['MachineQuantity_ForRouteCalc'] * df[f'RoutesPerMachine_{Disease}']

## Assign latent testing demand to underutilized machines
## First assign latent demand to machines in the same region that have available capacity
## Then assign remaining latent demand by moving samples to regions with machines that have available capacity.
## --For this second step, include latent demand for regions that do not have any machines at Baseline.
TotalCost_Dict = dict(zip(df['admin_name'], df['TotalCost_' + Disease + '_Baseline']))
VariableCost_Dict = dict(zip(df['admin_name'], df['VariableCost_Total_' + Disease + '_Baseline']))
TransportCost_Dict = dict(zip(df['admin_name'], df['TransportCost_Total_' + Disease + '_Baseline']))
UtilizedCapacity_Dict = dict(zip(df['admin_name'], df['UtilizedCapacity_' + Disease + '_Baseline']))
RemainingCapacity_Dict = dict(zip(df['admin_name'], df['AvailableCapacity_' + Disease]))
RemainingNeed_Dict = dict(zip(df['admin_name'], df['UnmetNeed']))
RemainingNeedBeyondThreshold_Dict = dict(zip(df['admin_name'], [0]*len(df)))

TrackingDict = {
    'TotalCost_Dict' : TotalCost_Dict,
    'VariableCost_Dict' : VariableCost_Dict,
    'TransportCost_Dict' : TransportCost_Dict,
    'UtilizedCapacity_Dict' : UtilizedCapacity_Dict,
    'RemainingCapacity_Dict' : RemainingCapacity_Dict,
    'RemainingNeed_Dict' : RemainingNeed_Dict
    }

Log_IntraRegional = {}
Log_InterRegional = {}

## Sort df, defining the order in which intra-regional transport routes will be added
df['Region_Order'] = df['admin_name'].apply(lambda x: Regions_Sorted.index(x) if x in Regions_Sorted else len(Regions_Sorted))
df = df.sort_values(by=['Region_Order', 'admin_name'])
df = df.drop(columns=['Region_Order'])

#Intra-regional transport calculation
if Budget > 0:

    for _, row in df.iterrows():
        Region = row['admin_name']
        AvailableCapacity = row['AvailableCapacity_' + Disease]
        UnmetNeed = row['UnmetNeed']
        VariableCostPerTest = row['Avg_Variable_Cost']
        NewRoute_Distance = row[f'DistancePerRoute_Raw_{Disease}']
        NumRoutes_Baseline = round(row['NumRoutes_Baseline'],0)
        MaxRoutes = row['EstimatedHealthFacilities'] # assume that saturation achieved at 1 route per health facility per disease
        NumTests_Baseline = row['UtilizedCapacity_' + Disease + '_Baseline']

        TestsPerRoutePerYear = row['TestsPerRoutePerYear_' + Disease]
        if TestsPerRoutePerYear == 0:
            TestsPerRoutePerYear = st.session_state.TestsPerRoute_Avg

        if UnmetNeed > 0 and NewRoute_Distance <= DistanceThreshold:  
               
            if AvailableCapacity >= UnmetNeed:
                RouteList, MaxRoutes_Cap = f.create_route_list_new_tests(UnmetNeed, TestsPerRoutePerYear, NewRoute_Distance, NumRoutes_Baseline, MaxRoutes)      
                NumRoutes = len(RouteList)
                NewTests = UnmetNeed
                if NumRoutes * TestsPerRoutePerYear < UnmetNeed and not MaxRoutes_Cap:
                    NewTests = NumRoutes * TestsPerRoutePerYear 
                VariableCost_Total = NewTests * VariableCostPerTest 
                TransportCost_Total = f.calculate_transport_cost_new_tests(RouteList, CostPerKm, TransportFreqPerWeek, TransportWeeksPerYear)
                NewCost = TransportCost_Total + VariableCost_Total
                
                if NewCost <= Budget:
                    TrackingDict = f.update_tracking_dict(TrackingDict, Region, Region, NewCost, VariableCost_Total, TransportCost_Total, NewTests) 
                    Budget -= NewCost

                    if (NumRoutes_Baseline + NumRoutes) < MaxRoutes:
                        MaxRoutes_Cap = False 
                    Log_IntraRegional = f.update_intra_regional_log(Log_IntraRegional, Region, UnmetNeed, NewTests, NumTests_Baseline, NumRoutes, NewRoute_Distance, NumRoutes_Baseline, MaxRoutes, Disease)
                    
                else:             
                    RouteList_Final, NumRoutes, VariableCost_Total, TransportCost_Total, NewTests = f.calculate_num_routes(UnmetNeed, Budget, TestsPerRoutePerYear, RouteList, CostPerKm, TransportFreqPerWeek, TransportWeeksPerYear, VariableCostPerTest, MaxRoutes_Cap) 
                    NewCost = VariableCost_Total + TransportCost_Total
                    TrackingDict = f.update_tracking_dict(TrackingDict, Region, Region, NewCost, VariableCost_Total, TransportCost_Total, NewTests) 
                    budget_depleted = True
                    Budget = 0 #Approximation; some budget will be left over, but it will not be enough to add another transport route 

                    if (NumRoutes_Baseline + NumRoutes) < MaxRoutes:
                        MaxRoutes_Cap = False 
                    Log_IntraRegional = f.update_intra_regional_log(Log_IntraRegional, Region, UnmetNeed, NewTests, NumTests_Baseline, NumRoutes, NewRoute_Distance, NumRoutes_Baseline, MaxRoutes, Disease)
                    
                    break
                    
            else:
                RouteList, MaxRoutes_Cap = f.create_route_list_new_tests(AvailableCapacity, TestsPerRoutePerYear, NewRoute_Distance, NumRoutes_Baseline, MaxRoutes) 
                NumRoutes = len(RouteList)
                # If the number of new routes created pushes the total number of routes to the saturation point for the region,
                # --as many tests as needed can move along each route, up to the number that utilize all available capacity
                if MaxRoutes_Cap:
                    NewTests = AvailableCapacity
                else:
                    TestsBelowThreshold = NumRoutes * TestsPerRoutePerYear 
                    if AvailableCapacity >= TestsBelowThreshold:
                        NewTests = TestsBelowThreshold
                    else:
                        NewTests = AvailableCapacity
                        
                TransportCost_Total = f.calculate_transport_cost_new_tests(RouteList[:NumRoutes], CostPerKm, TransportFreqPerWeek, TransportWeeksPerYear)
                VariableCost_Total = NewTests * VariableCostPerTest 
                NewCost = TransportCost_Total + VariableCost_Total

                if NewCost <= Budget:
                    TrackingDict = f.update_tracking_dict(TrackingDict, Region, Region, NewCost, VariableCost_Total, TransportCost_Total, NewTests) 
                    Budget -= NewCost
                    
                    if (NumRoutes_Baseline + NumRoutes) < MaxRoutes:
                        MaxRoutes_Cap = False 
                    Log_IntraRegional = f.update_intra_regional_log(Log_IntraRegional, Region, UnmetNeed, NewTests, NumTests_Baseline, NumRoutes, NewRoute_Distance, NumRoutes_Baseline, MaxRoutes, Disease)
                
                else:
                    RouteList_Final, NumRoutes, VariableCost_Total, TransportCost_Total, NewTests = f.calculate_num_routes(AvailableCapacity, Budget, TestsPerRoutePerYear, RouteList, CostPerKm, TransportFreqPerWeek, TransportWeeksPerYear, VariableCostPerTest, MaxRoutes_Cap) 
                    NewCost = VariableCost_Total + TransportCost_Total
                    TrackingDict = f.update_tracking_dict(TrackingDict, Region, Region, NewCost, VariableCost_Total, TransportCost_Total, NewTests) 
                    budget_depleted = True
                    Budget = 0 #Approximation; some budget will be left over, but it will not be enough to add another transport route 
                    
                    if (NumRoutes_Baseline + NumRoutes) < MaxRoutes:
                        MaxRoutes_Cap = False 
                    Log_IntraRegional = f.update_intra_regional_log(Log_IntraRegional, Region, UnmetNeed, NewTests, NumTests_Baseline, NumRoutes, NewRoute_Distance, NumRoutes_Baseline, MaxRoutes, Disease)
                    
                    break

#Inter-regional transport calculation      
# Assume that MaxRoutes_Cap (route saturation in a region) does not apply to inter-regional transport          
if Budget > 0:

    ## Add latent demand for regions that do not have any machines at Baseline to RemainingNeed from the previous code block
    # Start with RemainingNeed as the full estimate of testing need (true for regions without machines at Baseline)
    # Replace values of RemainingNeed using outputs from the previous code block
    RemainingNeed_New = dict(zip(DiseaseBurden_Target['admin_name'], DiseaseBurden_Target['TestingNeed_' + Disease]))
    for region, need in TrackingDict['RemainingNeed_Dict'].items():
        RemainingNeed_New[region] = need

    #Subtract remaining need that is beyond the distance threshold (not eligible for inter-regional transport)
    for region, need in RemainingNeedBeyondThreshold_Dict.items():
        RemainingNeed_New[region] -= need

    Temp = { k:v for k, v in RemainingNeed_New.items() if v != 0 }
    TrackingDict['RemainingNeed_Dict'] = Temp

    ## Drop regions with no remaining capacity
    Temp = { k:v for k, v in TrackingDict['RemainingCapacity_Dict'].items() if v != 0 }
    TrackingDict['RemainingCapacity_Dict'] =  Temp

    df2 = DistanceTime
    DistanceTime_Dict = {}

    if len(TrackingDict['RemainingNeed_Dict']) > 0 and len(TrackingDict['RemainingCapacity_Dict']) > 0:
        RegionsWithNeed = list(TrackingDict['RemainingNeed_Dict'].keys())
        RegionsWithCapacity = list(TrackingDict['RemainingCapacity_Dict'].keys())
        RegionPairs = list(product(RegionsWithNeed, RegionsWithCapacity))

        # Pull distance and travel time for each pair
        for pair in RegionPairs:
            RegionNeed = pair[0]
            RegionCapacity = pair[1]

            Temp = (df2[(df2['orig_country code'] == TargetCountry) & ((df2['orig_admin_name'] == RegionNeed) | (df2['orig_admin_name'] == RegionCapacity)) & ((df2['dest_admin_name'] == RegionNeed) | (df2['dest_admin_name'] == RegionCapacity))]).squeeze()
            Distance = Temp['Km']
            Seconds = Temp['seconds']
            Time = Seconds / 60 # to minutes 

            DistanceTime_Dict[pair] = (Distance, Time)
    
        df3 = pd.DataFrame.from_dict(DistanceTime_Dict, orient='index', columns=['Distance', 'Time']) 
        
        # Drop entries with distance beyond thresholds
        # This operation also drops region pairs with no distance data (NaN).
        df3 = df3.loc[(df3['Distance'] <= DistanceThreshold)]

        # Sort entries, shortest distance first
        df3.sort_values(by=['Distance'], ascending=True, inplace=True)

        # Loop through each pair of region with need and region with capacity.  Testing allocation and cost tracked through dictionaries, which are then appended to the main df
        budget_depleted = False
        for i in range(len(df3)):
            RegionNeed = df3.index[i][0]
            RegionCapacity =df3.index[i][1]

            RemainingNeed = TrackingDict['RemainingNeed_Dict'][RegionNeed]
            RemainingCapacity = TrackingDict['RemainingCapacity_Dict'][RegionCapacity]

            DistancePerRoute = df3.iloc[i, df3.columns.get_loc('Distance')]
            if RegionNeed in df['admin_name'].values:
                TestsPerRoutePerYear = df.query('admin_name == @RegionNeed').squeeze()['TestsPerRoutePerYear_' + Disease]
            else:
                TestsPerRoutePerYear = st.session_state.TestsPerRoute_Avg
            
            if RemainingNeed > 0 and RemainingCapacity > 0: 

                VariableCostPerTest = df.query('admin_name == @RegionCapacity').squeeze()['Avg_Variable_Cost']

                if RemainingCapacity >= RemainingNeed:
                    NewTests = RemainingNeed
                    TransportCost_Total = np.ceil(NewTests/TestsPerRoutePerYear) * DistancePerRoute * CostPerKm * TransportFreqPerWeek * TransportWeeksPerYear
                    VariableCost_Total = NewTests * VariableCostPerTest
                    NewCost = TransportCost_Total + VariableCost_Total
                    
                    if NewCost <= Budget:
                        TrackingDict = f.update_tracking_dict(TrackingDict, RegionCapacity, RegionNeed, NewCost, VariableCost_Total, TransportCost_Total, NewTests)
                        Budget -= NewCost

                        Log_InterRegional[(RegionNeed,RegionCapacity)] = {
                            'New Routes Created' : np.ceil(NewTests/TestsPerRoutePerYear),
                            'New Tests Performed by Paying for Sample Transport and Processing (Tests per Year)' : NewTests,
                            'Distance of Each New Route (km)' : DistancePerRoute}

                    else:
                        NumRoutes = np.floor(Budget / ((DistancePerRoute * CostPerKm * TransportFreqPerWeek * TransportWeeksPerYear) + (TestsPerRoutePerYear * VariableCostPerTest))) 
                        TransportCost_Total = NumRoutes * DistancePerRoute * CostPerKm * TransportFreqPerWeek * TransportWeeksPerYear
                        VariableCost_Total = NumRoutes * TestsPerRoutePerYear * VariableCostPerTest
                        NewCost = VariableCost_Total + TransportCost_Total
                        TrackingDict = f.update_tracking_dict(TrackingDict, RegionCapacity, RegionNeed, NewCost, VariableCost_Total, TransportCost_Total, NewTests)
                        budget_depleted = True
                        Budget = 0 #Approximation; some budget will be left over, but it will not be enough to add another transport route
                        
                        Log_InterRegional[(RegionNeed,RegionCapacity)] = {
                            'New Routes Created' : NumRoutes,
                            'New Tests Performed by Paying for Sample Transport and Processing (Tests per Year)' : NumRoutes * TestsPerRoutePerYear,
                            'Distance of Each New Route (km)' : DistancePerRoute}
                        
                        break 

                else:
                    NewTests = RemainingCapacity
                    TransportCost_Total = np.ceil(NewTests/TestsPerRoutePerYear) * DistancePerRoute * CostPerKm * TransportFreqPerWeek * TransportWeeksPerYear
                    VariableCost_Total = NewTests * VariableCostPerTest
                    NewCost = TransportCost_Total + VariableCost_Total

                    if NewCost <= Budget:
                        TrackingDict = f.update_tracking_dict(TrackingDict, RegionCapacity, RegionNeed, NewCost, VariableCost_Total, TransportCost_Total, NewTests)
                        Budget -= NewCost 

                        Log_InterRegional[(RegionNeed,RegionCapacity)] = {
                            'New Routes Created' : np.ceil(NewTests/TestsPerRoutePerYear),
                            'New Tests Performed by Paying for Sample Transport and Processing (Tests per Year)' : NewTests,
                            'Distance of Each New Route (km)' : DistancePerRoute}

                    else:
                        NumRoutes = np.floor(Budget / ((DistancePerRoute * CostPerKm * TransportFreqPerWeek * TransportWeeksPerYear) + (TestsPerRoutePerYear * VariableCostPerTest))) 
                        TransportCost_Total = NumRoutes * DistancePerRoute * CostPerKm * TransportFreqPerWeek * TransportWeeksPerYear
                        VariableCost_Total = NumRoutes * TestsPerRoutePerYear * VariableCostPerTest
                        NewCost = VariableCost_Total + TransportCost_Total
                        TrackingDict = f.update_tracking_dict(TrackingDict, RegionCapacity, RegionNeed, NewCost, VariableCost_Total, TransportCost_Total, NewTests)
                        budget_depleted = True
                        Budget = 0 #Approximation; some budget will be left over, but it will not be enough to add another transport route 
                        
                        Log_InterRegional[(RegionNeed,RegionCapacity)] = {
                            'New Routes Created' : NumRoutes,
                            'New Tests Performed by Paying for Sample Transport and Processing (Tests per Year)' : NumRoutes * TestsPerRoutePerYear,
                            'Distance of Each New Route (km)' : DistancePerRoute}
                    
                        break 

### Determine reason for stopping the "pay for transport" algorithms
if budget_depleted or Budget == 0:
    Constraint = 'Budget depleted'
elif sum(TrackingDict['RemainingNeed_Dict'].values()) == 0:
    Constraint = 'No remaining testing need'
else:
    Constraint = 'No more machine capacity within distance threshold'

### Calculate summary statistics for the "pay for transport" scenario
df.set_index('admin_name', inplace=True)
df['TotalCost_' + Disease] = df.index.map(TrackingDict['TotalCost_Dict'])
df['VariableCost_Total_' + Disease] = df.index.map(TrackingDict['VariableCost_Dict'])
df['TransportCost_Total_' + Disease] = df.index.map(TrackingDict['TransportCost_Dict'])
df['UtilizedCapacity_' + Disease] = df.index.map(TrackingDict['UtilizedCapacity_Dict'])

## Handling of regions with need that do not contain machines at baseline
## -- These regions create entries for cost in TrackingDict but are not included in the Master_Baseline dataframe
missing_keys = set(TrackingDict['TotalCost_Dict']).difference(df.index)
missing_data = {key: {subkey: TrackingDict[subkey][key] for subkey in TrackingDict if key in TrackingDict[subkey]} for key in missing_keys}
missing_df = pd.DataFrame(missing_data).transpose()

missing_df.rename({'TotalCost_Dict' : 'TotalCost_' + Disease,
        'TransportCost_Dict' : 'TransportCost_Total_' + Disease,
        'VariableCost_Dict' : 'VariableCost_Total_' + Disease,
        },
        axis=1, inplace=True
    )
try:
    missing_df.drop(['RemainingNeed_Dict'], axis=1, inplace=True)
except:
    pass
df = pd.concat([df, missing_df])

Stats_PayTransport = f.calculate_statistics(df, DiseaseBurden_Target, Disease, DiseaseList)

### Create dataframe with analysis results
MachineTransport_Results = pd.DataFrame(
    np.array([
        [Stats_Baseline['TotalCost'], Stats_Scenario['TotalCost'], Stats_PayTransport['TotalCost']],
        [Stats_Baseline['MachineCost'], Stats_Scenario['MachineCost'], Stats_PayTransport['MachineCost']],
        [Stats_Baseline['StaffCost'], Stats_Scenario['StaffCost'], Stats_PayTransport['StaffCost']],
        [Stats_Baseline['VariableCost'], Stats_Scenario['VariableCost'], Stats_PayTransport['VariableCost']],
        [Stats_Baseline['TransportCost'], Stats_Scenario['TransportCost'], Stats_PayTransport['TransportCost']],
        [Stats_Baseline['UnitCost_Total'], Stats_Scenario['UnitCost_Total'], Stats_PayTransport['UnitCost_Total']],
        [Stats_Baseline['UnitCost_Machine'], Stats_Scenario['UnitCost_Machine'], Stats_PayTransport['UnitCost_Machine']],
        [Stats_Baseline['UnitCost_Staff'], Stats_Scenario['UnitCost_Staff'], Stats_PayTransport['UnitCost_Staff']],
        [Stats_Baseline['UnitCost_Variable'], Stats_Scenario['UnitCost_Variable'], Stats_PayTransport['UnitCost_Variable']],
        [Stats_Baseline['UnitCost_Transport'], Stats_Scenario['UnitCost_Transport'], Stats_PayTransport['UnitCost_Transport']],
        [Stats_Baseline['UtilizationRate'], Stats_Scenario['UtilizationRate'], Stats_PayTransport['UtilizationRate']],
        [Stats_Baseline['NumTests'], Stats_Scenario['NumTests'], Stats_PayTransport['NumTests']],
        [Stats_Baseline['NeedMet'], Stats_Scenario['NeedMet'], Stats_PayTransport['NeedMet']],
        ]),
    columns=['Baseline', 'Add Machines', 'Pay for Transport'],
    index=[f'Total Annual Cost : {Disease}', '-- Machine', '-- Staff', '-- Variable', '-- Transport',
            f'Cost per Test : {Disease}', ' -- Machine', ' -- Staff', ' -- Variable', ' -- Transport', 
            'Average Machine Utilization (All Diseases)',
            f'Total {Disease} MolDx Tests Per Year',
            f'Percentage of {Disease} MolDx Testing Need Met'] 
    )

if 'MachineTransport_Results' not in st.session_state:
    st.session_state.MachineTransport_Results = pd.DataFrame()

st.session_state.MachineTransport_Results = MachineTransport_Results

### Add figures
## Total cost
data = []
Labels = ['Machine Cost', 'Staff Cost', 'Variable Cost', 'Transport Cost']
DictKeys = ['MachineCost', 'StaffCost', 'VariableCost', 'TransportCost']

# Loop through the dictionaries and keys, and append each row to the data list
for stat_dict, scenario in [(Stats_Baseline, 'Baseline'), (Stats_Scenario, 'Add Machines'), (Stats_PayTransport, 'Pay for Transport')]:
    for label, key in zip(Labels, DictKeys):
        data.append((scenario, label, stat_dict[key]))

# Create the dataframe from the data list
df = pd.DataFrame(data, columns=['Scenario', 'Cost Component', 'Value'])
dfs = df.groupby('Scenario').sum()

# Main figure
fig = px.bar(df, 
    x='Scenario', 
    y='Value', 
    color='Cost Component', 
    color_discrete_sequence=['#0d0887', '#9c179e','#ed7953','#f0f921'])

# Total over each bar
fig.add_trace(go.Scatter(
    x=dfs.index, 
    y=dfs['Value'],
    text=[f'${val:,.0f}' for val in dfs['Value']],
    mode='text',
    textposition='top center',
    textfont=dict(
        size=18,
    ),
    showlegend=False
))

fig.update_layout(legend={'orientation': 'h'})

# Change position of legend
fig.update_layout(
    autosize=True,
    legend=dict(
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5,
        title=''
    ))

fig.update_layout(title_text=f'Total Annual Cost of {Disease} MolDx Testing', title_x=0.4)
fig.update_xaxes(title_text="")
fig.update_yaxes(title_text="", tickprefix="$", range=[0,dfs['Value'].max()*1.2])

## Unit cost
data = []
Labels = ['Machine Cost', 'Staff Cost', 'Variable Cost', 'Transport Cost']
DictKeys = ['UnitCost_Machine', 'UnitCost_Staff', 'UnitCost_Variable', 'UnitCost_Transport']

# Loop through the dictionaries and keys, and append each row to the data list
for stat_dict, scenario in [(Stats_Baseline, 'Baseline'), (Stats_Scenario, 'Add Machines'), (Stats_PayTransport, 'Pay for Transport')]:
    for label, key in zip(Labels, DictKeys):
        data.append((scenario, label, stat_dict[key]))

# Create the dataframe from the data list
df = pd.DataFrame(data, columns=['Scenario', 'Cost Component', 'Value'])
dfs = df.groupby('Scenario').sum()

fig2 = px.bar(df, 
    x='Scenario', 
    y='Value', 
    color='Cost Component', 
    color_discrete_sequence=['#0d0887', '#9c179e','#ed7953','#f0f921'])

fig2.add_trace(go.Scatter(
    x=dfs.index, 
    y=dfs['Value'],
    text=[f'${val:,.2f}' for val in dfs['Value']],
    mode='text',
    textposition='top center',
    textfont=dict(
        size=18,
    ),
    showlegend=False
))

fig2.update_layout(legend={'orientation': 'h'})

fig2.update_layout(
    autosize=True,
    legend=dict(
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5,
        title=''
    ))

fig2.update_layout(title_text=f'Cost Per {Disease} MolDx Test', title_x=0.4)
fig2.update_xaxes(title_text="")
fig2.update_yaxes(title_text="", tickprefix="$", range=[0,dfs['Value'].max()*1.2])

## % Need Met
data = []
Labels = ['Percentage of Need Met']
DictKeys = ['NeedMet']

# Loop through the dictionaries and keys, and append each row to the data list
for stat_dict, scenario in [(Stats_Baseline, 'Baseline'), (Stats_Scenario, 'Add Machines'), (Stats_PayTransport, 'Pay for Transport')]:
    for label, key in zip(Labels, DictKeys):
        data.append((scenario, label, stat_dict[key]))

df = pd.DataFrame(data, columns=['Scenario', 'Component', 'Value'])

fig3 = px.bar(df, 
    x='Scenario', 
    y='Value',
    text='Value', 
    color_discrete_sequence=['#0d0887'])

fig3.update_layout(title_text=f'Percentage of {Disease} MolDx Testing Need Met', title_x=0.3)
fig3.update_xaxes(title_text="")
fig3.update_traces(texttemplate='%{text:.0%}', textposition='outside', textfont_size=18)
fig3.update_yaxes(title_text="", tickformat=".0%", range=[0,df['Value'].max()*1.2])

## % Utilization
data = []
Labels = ['Average Machine Utilization']
DictKeys = ['UtilizationRate']

# Loop through the dictionaries and keys, and append each row to the data list
for stat_dict, scenario in [(Stats_Baseline, 'Baseline'), (Stats_Scenario, 'Add Machines'), (Stats_PayTransport, 'Pay for Transport')]:
    for label, key in zip(Labels, DictKeys):
        data.append((scenario, label, stat_dict[key]))

df = pd.DataFrame(data, columns=['Scenario', 'Component', 'Value'])

fig4 = px.bar(df, 
    x='Scenario', 
    y='Value',
    text='Value', 
    color_discrete_sequence=['#0d0887'])

fig4.update_layout(title_text='Average Machine Utilization (All Diseases)', title_x=0.3)
fig4.update_xaxes(title_text="")
fig4.update_traces(texttemplate='%{text:.0%}', textposition='outside', textfont_size=18)
fig4.update_yaxes(title_text="", tickformat=".0%", range=[0,df['Value'].max()*1.2])

## Number of Tests
data = []
Labels = ['Number of Tests']
DictKeys = ['NumTests']

# Loop through the dictionaries and keys, and append each row to the data list
for stat_dict, scenario in [(Stats_Baseline, 'Baseline'), (Stats_Scenario, 'Add Machines'), (Stats_PayTransport, 'Pay for Transport')]:
    for label, key in zip(Labels, DictKeys):
        data.append((scenario, label, stat_dict[key]))

df = pd.DataFrame(data, columns=['Scenario', 'Component', 'Value'])

fig5 = px.bar(df, 
    x='Scenario', 
    y='Value',
    text='Value', 
    color_discrete_sequence=['#0d0887'])

fig5.update_layout(title_text=f'Total {Disease} MolDx Tests Per Year', title_x=0.3)
fig5.update_xaxes(title_text="")
fig5.update_traces(texttemplate='%{text:,.0f}', textposition='outside', textfont_size=18)
fig5.update_yaxes(title_text="", tickformat=",.0f", range=[0,df['Value'].max()*1.2])

with st.container():
    col1, col2, col3 = st.columns(3)
    with st.container():
        with col1:
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.plotly_chart(fig2, use_container_width=True)
        
        with col3:
            st.plotly_chart(fig4, use_container_width=True)
            st.caption(f'Total tests for all diseases performed on machines that can perform {Disease} tests divided by total capacity of all machines that can perform {Disease} tests.')

with st.container():
    _, col2, col3, _ = st.columns([1,2,2,1])
    with st.container():
        with col2:
            st.plotly_chart(fig5, use_container_width=True)

        with col3:
            st.plotly_chart(fig3, use_container_width=True)

with st.container():
    # Format table values
    format_mapping = {
        "${:,.0f}": [f'Total Annual Cost : {Disease}', '-- Machine', '-- Staff', '-- Variable', '-- Transport'],
        "${:,.2f}": [f'Cost per Test : {Disease}', ' -- Machine', ' -- Staff', ' -- Variable', ' -- Transport'],
        "{:.0%}": ['Average Machine Utilization (All Diseases)', f'Percentage of {Disease} MolDx Testing Need Met'],
        "{:,.0f}": [f'Total {Disease} MolDx Tests Per Year']
        }

    for format_string, indices in format_mapping.items():
        for idx in indices:
            if idx in MachineTransport_Results.index:
                MachineTransport_Results.loc[idx] = MachineTransport_Results.loc[idx].apply(
                    lambda x: format_string.format(x) if pd.notnull(x) else ""
                )

    st.table(MachineTransport_Results)
    st.caption('Constraint in the Pay for Transport scenario: ' + Constraint)
    if TestingNeedAdjusted:
        st.caption(f'The number of {Disease} tests at baseline is higher than estimated testing need for the following regions: {", ".join(RegionsAdjusted_Need)}.  For these regions, the original testing need estimates were replaced with the number of tests performed at baseline.')

    if NumTests_Adjusted:
        st.caption(f'For the following regions, in the Add Machines scenario, the number of {Disease} tests was adjusted downward so as not to exceed estimated testing need: {", ".join(RegionsAdjusted_Tests)}')

# Display the "pay for transport" scenario logs
with st.container():

    LogList = [Log_IntraRegional, Log_InterRegional]
    for log in LogList:
        # Round values to zero decimal places
        for region_dict in log.values():
            for stat, val in region_dict.items():
                if stat == 'RouteList':
                    for i, route in enumerate(val):
                        val[i] = int(round(route, 0))
                else:
                    region_dict[stat] = int(round(val, 0))

    df_intra = pd.DataFrame(Log_IntraRegional).T
    df_inter = pd.DataFrame(Log_InterRegional).T
    
    if not df_intra.empty:
        df_intra.index.name = 'Region'
        df_intra = df_intra.reset_index()
        add_vertical_space(1)
        st.write('Pay for Transport: Intra-Regional Log')
        st.table(df_intra)

    if not df_inter.empty:
        df_inter.index.names = ['Region with Testing Need', 'Region with Testing Capacity']
        df_inter = df_inter.reset_index()
        add_vertical_space(1)
        st.write('Pay for Transport: Inter-Regional Log')
        st.table(df_inter)

with st.container():
    add_vertical_space(1)
    col1, _, _ = st.columns(st.session_state.NavButtonCols)

    with col1:
        if st.button('Back', use_container_width=True):
            switch_page('verify inputs')