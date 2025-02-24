import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import json
import functions as f
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# Initialize the Google Drive API
def get_gdrive_service():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

# Function to search for a file by name within a specific folder
def get_file_id_by_name(file_name, folder_id):
    service = get_gdrive_service()
    query = f"'{folder_id}' in parents"  # Only query for the folder
    
    results = service.files().list(
        q=query,
        spaces='drive',
        fields="files(id, name)"
    ).execute()
    
    files = results.get('files', [])
    
    # Print all files in the folder for debugging
    st.write("Files in folder:")
    for file in files:
        st.write(f"File Name: '{file['name']}', File ID: {file['id']}")  # Displaying the files for verification

    # Now, check if the specific file name exists (case insensitive)
    for file in files:
        if file['name'].lower() == file_name.lower():
            return file['id']
    
    st.error(f"No file found with the name: {file_name} in folder ID: {folder_id}")
    return None


# Function to download a file by its ID
def download_file(file_id, file_name):
    service = get_gdrive_service()
    request = service.files().get_media(fileId=file_id)
    
    # Create a BytesIO stream to hold the downloaded content
    fh = io.BytesIO()
    
    # Use MediaIoBaseDownload to download the file
    downloader = MediaIoBaseDownload(fh, request)
    
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        st.write(f"Download {int(status.progress() * 100)}% complete.")

    # Write the content to a file after download is complete
    with open(file_name, "wb") as f:
        f.write(fh.getbuffer())

    return file_name

# Load GeoJSON by file name and folder ID
def load_geojson(file_name, folder_id):
    file_id = get_file_id_by_name(file_name, folder_id)
    if not file_id:
        return None
    file_path = download_file(file_id, "temp.geojson")
    return gpd.read_file(file_path)
    
# Function to list all files in a specific folder
def list_files_in_folder(folder_id):
    service = get_gdrive_service()
    query = f"'{folder_id}' in parents"
    
    results = service.files().list(
        q=query,
        spaces='drive',
        fields="files(id, name)"
    ).execute()
    
    files = results.get('files', [])
    if not files:
        st.write("No files found in the folder.")
    else:
        for file in files:
            st.write(f"File Name: {file['name']}, File ID: {file['id']}")


st.markdown(st.session_state.ReducePadding, unsafe_allow_html=True)

with st.spinner("Loading..."):
    ### Assign session state data to dataframes matching subsequent code
    DNOInput = st.session_state.DNOInput
    TargetCountry = st.session_state.TargetCountry
    MachineTypes = st.session_state.MachineTypes
    DiseaseBurden = st.session_state.DiseaseBurden
    WorkDaysPerYear = st.session_state.WorkDaysPerYear
    DiseaseList = st.session_state.DiseaseList
    EligibleCountries = st.session_state.EligibleCountries

    ### Load and subset data sets
    MachinesByRegion_Baseline_Target = st.session_state.BaselineData
    MachinesByRegion_Baseline_Target.index.name = 'admin_name'
    DiseaseBurden_Target = DiseaseBurden.loc[(DiseaseBurden['admin_name'].isin(st.session_state.RegionsSelected)) & (DiseaseBurden['iso3'] == TargetCountry)]

    DiseaseList_Orig =  DiseaseList.copy()
    DiseaseList = ['HIV', 'TB']

    ### Create table with one row per region and machine type combination
    MachinesByRegion_Baseline_Target.index.rename('index', inplace=True)
    Master_Baseline = f.stack_table(MachinesByRegion_Baseline_Target, DiseaseList_Orig) 
    Master_Baseline = Master_Baseline.merge(MachineTypes, left_on='Machine', right_index=True, how='left')

    ### Calculate testing capacity
    ## Add calculated columns at the [Region]-[Machine Type] level
    # Reduce total capacity by these tests that are performed for other diseases (not HIV or TB)
    Master_Baseline['UtilizedCapacity_Other'] = Master_Baseline['TestsPerMachinePerYear_Other'] * Master_Baseline['MachineQuantity']
    Master_Baseline['UtilizedCapacity_Other'].fillna(0, inplace=True)
    Master_Baseline['Capacity_Day'] = pd.to_numeric(Master_Baseline['Capacity_Day'], errors='coerce')

    Master_Baseline['AnnualCapacity_HIV_Only'] = np.where(
        (Master_Baseline['Usable_TB'] == 0) & (Master_Baseline['Usable_HIV'] == 1), 
        (Master_Baseline['MachineQuantity'] * Master_Baseline['Capacity_Day'] * WorkDaysPerYear) - Master_Baseline['UtilizedCapacity_Other'], 
        0)
    Master_Baseline['AnnualCapacity_TB_Only'] = np.where(
        (Master_Baseline['Usable_TB'] == 1) & (Master_Baseline['Usable_HIV'] == 0), 
        (Master_Baseline['MachineQuantity'] * Master_Baseline['Capacity_Day'] * WorkDaysPerYear) - Master_Baseline['UtilizedCapacity_Other'], 
        0)
    Master_Baseline['AnnualCapacity_HIV_TB'] = np.where(
        (Master_Baseline['Usable_TB'] == 1) & (Master_Baseline['Usable_HIV'] == 1), 
        (Master_Baseline['MachineQuantity'] * Master_Baseline['Capacity_Day'] * WorkDaysPerYear) - Master_Baseline['UtilizedCapacity_Other'], 
        0)

    ### Calculate testing demand
    for d in DiseaseList:
        Master_Baseline['AnnualTests_' + d] = Master_Baseline['TestsPerMachinePerYear_' + d] * Master_Baseline['MachineQuantity']

    ## Aggregate at the [Region] level
    KeepList = ['admin_name', 'MachineQuantity', 'AnnualCapacity_HIV_Only','AnnualCapacity_TB_Only', 'AnnualCapacity_HIV_TB']
    for d in DiseaseList:
        KeepList.append('AnnualTests_' + d)

    CapacityNeed = Master_Baseline[KeepList].copy()

    ColList = CapacityNeed.columns.tolist()
    RemoveList = ['admin_name']
    for item in RemoveList:
        ColList.remove(item)
    func_dict = {n: 'sum' for n in ColList}
    func_dict['admin_name'] = 'first'
    CapacityNeed = CapacityNeed.groupby(['admin_name'], as_index=False).agg(func_dict)

    ### Create dict to hold national-level stats
    NationalStats = {}

    ### Calculate current annual test demand as percent of capacity
    for d in DiseaseList:
        # Calculate subnational-level stats
        CapacityNeed['AnnualTests_PctCapacity_' + d] = CapacityNeed['AnnualTests_' + d] / CapacityNeed[['AnnualCapacity_' + d + '_Only', 'AnnualCapacity_HIV_TB']].sum(axis=1)

        # Calculate national-level stat
        NationalStats['AnnualTests_PctCapacity_' + d] = CapacityNeed['AnnualTests_' + d].sum() / (CapacityNeed['AnnualCapacity_' + d + '_Only'].sum() + CapacityNeed['AnnualCapacity_HIV_TB'].sum())


    # Demand for a given test (HIV or TB) should not exceed capacity for that test since AnnualTests is based on historical testing
    # For this reason, a simple calculation like the one below is valid.  It does not, however, inform the user of the proportion of remaining capacity that can only be used for one of the two diseases.
    if 'HIV' in DiseaseList and 'TB' in DiseaseList:    
        # Calculate subnational-level stats
        CapacityNeed['AnnualTests_PctCapacity_HIV_TB'] = CapacityNeed[['AnnualTests_TB', 'AnnualTests_HIV']].sum(axis=1) / CapacityNeed[['AnnualCapacity_HIV_Only', 'AnnualCapacity_TB_Only', 'AnnualCapacity_HIV_TB']].sum(axis=1)

        # Calculate national-level stat
        NationalStats['AnnualTests_PctCapacity_HIV_TB'] = (CapacityNeed['AnnualTests_TB'].sum() + CapacityNeed['AnnualTests_HIV'].sum()) / (CapacityNeed['AnnualCapacity_HIV_Only'].sum() + CapacityNeed['AnnualCapacity_TB_Only'].sum() + CapacityNeed['AnnualCapacity_HIV_TB'].sum())

    ### Calculate current annual test need as percent of capacity
    # Lookup target testing rates
    MergeList = ['admin_name']
    for d in DiseaseList:
        MergeList.append('TestingNeed_' + d)

    CapacityNeed = CapacityNeed.merge(DiseaseBurden_Target[MergeList], on='admin_name', how='left')

    # Adjust testing need if the number of tests at baseline exceeds estimated need
    TestingNeedAdjusted = False
    RegionsAdjusted = {}
    #for d in DiseaseList:
    #    CapacityNeed[f'TestingNeed_{d}_Original'] = CapacityNeed[f'TestingNeed_{d}']
    #    CapacityNeed['TestingNeed_' + d] = np.where(
    #        CapacityNeed['AnnualTests_' + d] > CapacityNeed['TestingNeed_' + d], 
    #        CapacityNeed['AnnualTests_' + d],
    #        CapacityNeed['TestingNeed_' + d])
    #    
    #    if (CapacityNeed[f'TestingNeed_{d}'] != CapacityNeed[f'TestingNeed_{d}_Original']).any():
    #        TestingNeedAdjusted = True
    #    
    #    # Make list of CapacityNeed['admin_name'] for which (CapacityNeed[f'TestingNeed_{d}'] != CapacityNeed[f'TestingNeed_{d}_Original'])
    #    RegionsAdjusted[d] = CapacityNeed.loc[CapacityNeed[f'TestingNeed_{d}'] != CapacityNeed[f'TestingNeed_{d}_Original'], 'admin_name'].tolist()
        
    # Calculate percentages
    NeedDict = {}
    for d in DiseaseList:
        # Calculate subnational-level stats
        CapacityNeed['AnnualNeed_PctCapacity_' + d] = CapacityNeed['TestingNeed_' + d] / CapacityNeed[['AnnualCapacity_' + d + '_Only', 'AnnualCapacity_HIV_TB']].sum(axis=1)

        # Calculate national-level stat
        # Selected regions without capacity need to be considered for testing need values
        # DiseaseBurden_Target already includes only selected regions
        NeedDict['TestingNeed_' + d] = DiseaseBurden_Target['TestingNeed_' + d].sum()
        NationalStats['AnnualNeed_PctCapacity_' + d] = NeedDict['TestingNeed_' + d] / (CapacityNeed['AnnualCapacity_' + d + '_Only'].sum() + CapacityNeed['AnnualCapacity_HIV_TB'].sum())


    if 'HIV' in DiseaseList and 'TB' in DiseaseList:

        ## Calculate subnational-level stats
        # Step 1: Allocate disease-specific capacity
        CapacityNeed['HIV_tests_HIV_only_machines'] = CapacityNeed[['TestingNeed_HIV', 'AnnualCapacity_HIV_Only']].min(axis=1)
        CapacityNeed['TB_tests_TB_only_machines'] = CapacityNeed[['TestingNeed_TB', 'AnnualCapacity_TB_Only']].min(axis=1)

        # Step 2: Calculate remaining need
        CapacityNeed['Remaining_HIV_need'] = CapacityNeed['TestingNeed_HIV'] - CapacityNeed['HIV_tests_HIV_only_machines']
        CapacityNeed['Remaining_TB_need'] = CapacityNeed['TestingNeed_TB'] - CapacityNeed['TB_tests_TB_only_machines']

        # Step 3: Allocate remaining demand to shared machines
        CapacityNeed['HIV_tests_shared_machines'] = 0
        CapacityNeed['TB_tests_shared_machines'] = 0

        hiv_higher_demand_mask = CapacityNeed['Remaining_HIV_need'] > CapacityNeed['Remaining_TB_need']
        tb_higher_demand_mask = ~hiv_higher_demand_mask

        # Step 3: Allocate remaining demand to shared machines
        CapacityNeed.loc[hiv_higher_demand_mask, 'HIV_tests_shared_machines'] = CapacityNeed.loc[hiv_higher_demand_mask, ['Remaining_HIV_need', 'AnnualCapacity_HIV_TB']].min(axis=1)
        CapacityNeed.loc[hiv_higher_demand_mask, 'TB_tests_shared_machines'] = CapacityNeed.loc[hiv_higher_demand_mask].apply(lambda row: min(row['Remaining_TB_need'], row['AnnualCapacity_HIV_TB'] - row['HIV_tests_shared_machines']), axis=1)

        CapacityNeed.loc[tb_higher_demand_mask, 'TB_tests_shared_machines'] = CapacityNeed.loc[tb_higher_demand_mask, ['Remaining_TB_need', 'AnnualCapacity_HIV_TB']].min(axis=1)
        CapacityNeed.loc[tb_higher_demand_mask, 'HIV_tests_shared_machines'] = CapacityNeed.loc[tb_higher_demand_mask].apply(lambda row: min(row['Remaining_HIV_need'], row['AnnualCapacity_HIV_TB'] - row['TB_tests_shared_machines']), axis=1)

        # Step 4: Calculate total tests performed
        CapacityNeed['Total_HIV_tests_performed'] = CapacityNeed[['HIV_tests_HIV_only_machines', 'HIV_tests_shared_machines']].sum(axis=1)
        CapacityNeed['Total_TB_tests_performed'] = CapacityNeed[['TB_tests_TB_only_machines', 'TB_tests_shared_machines']].sum(axis=1)

        # Step 5: Calculate testing need as a percentage of capacity
        # If all the needed tests for each disease can be performed with the available capacity, perform simple calculation and assume excess capacity remains (though without specifying which disease this excess capacity can be used for)
        mask = (CapacityNeed['Total_HIV_tests_performed'] == CapacityNeed['TestingNeed_HIV']) & (CapacityNeed['Total_TB_tests_performed'] == CapacityNeed['TestingNeed_TB'])
        CapacityNeed.loc[mask, 'AnnualNeed_PctCapacity_HIV_TB'] = (CapacityNeed.loc[mask, 'TestingNeed_HIV'] + CapacityNeed.loc[mask, 'TestingNeed_TB']) / (CapacityNeed.loc[mask, 'AnnualCapacity_HIV_Only'] + CapacityNeed.loc[mask, 'AnnualCapacity_TB_Only'] + CapacityNeed.loc[mask, 'AnnualCapacity_HIV_TB'])

        # Otherwise, drop any remaining capacity, as it is not usable for the estimated need across both diseases. Tests performed reflects capacity usable for the need (i.e., excludes capacity that can only be used for the other disease)
        CapacityNeed.loc[~mask, 'AnnualNeed_PctCapacity_HIV_TB'] = (CapacityNeed.loc[~mask, 'TestingNeed_HIV'] + CapacityNeed.loc[~mask, 'TestingNeed_TB']) / (CapacityNeed.loc[~mask, 'Total_HIV_tests_performed'] + CapacityNeed.loc[~mask, 'Total_TB_tests_performed'])

        #-----------------------------------------------------------------------------------------------------------------------

        ## Calculate national-level stat
        AnnualCapacity_HIV_Only = CapacityNeed['AnnualCapacity_HIV_Only'].sum()
        AnnualCapacity_TB_Only = CapacityNeed['AnnualCapacity_TB_Only'].sum()
        AnnualCapacity_HIV_TB = CapacityNeed['AnnualCapacity_HIV_TB'].sum()
        TestingNeed_HIV = NeedDict['TestingNeed_HIV']
        TestingNeed_TB = NeedDict['TestingNeed_TB']

        # Step 1: Allocate disease-specific capacity
        HIV_tests_HIV_only_machines = min(TestingNeed_HIV, AnnualCapacity_HIV_Only)
        TB_tests_TB_only_machines = min(TestingNeed_TB, AnnualCapacity_TB_Only)

        # Step 2: Calculate remaining need
        Remaining_HIV_need = TestingNeed_HIV - HIV_tests_HIV_only_machines
        Remaining_TB_need = TestingNeed_TB - TB_tests_TB_only_machines

        # Step 3: Allocate remaining demand to shared machines
        HIV_tests_shared_machines = 0
        TB_tests_shared_machines = 0

        if Remaining_HIV_need > Remaining_TB_need:
            HIV_tests_shared_machines = min(Remaining_HIV_need, AnnualCapacity_HIV_TB - HIV_tests_HIV_only_machines)
            TB_tests_shared_machines = min(Remaining_TB_need, AnnualCapacity_HIV_TB - HIV_tests_shared_machines)
        else:
            TB_tests_shared_machines = min(Remaining_TB_need, AnnualCapacity_HIV_TB - TB_tests_TB_only_machines)
            HIV_tests_shared_machines = min(Remaining_HIV_need, AnnualCapacity_HIV_TB - TB_tests_shared_machines)

        # Step 4: Calculate total tests performed
        Total_HIV_tests_performed = HIV_tests_HIV_only_machines + HIV_tests_shared_machines
        Total_TB_tests_performed = TB_tests_TB_only_machines + TB_tests_shared_machines

        # Step 5: Calculate testing need as a percentage of capacity
        if Total_HIV_tests_performed == TestingNeed_HIV and Total_TB_tests_performed == TestingNeed_TB:
            AnnualNeed_PctCapacity_HIV_TB = (TestingNeed_HIV + TestingNeed_TB) / (AnnualCapacity_HIV_Only + AnnualCapacity_TB_Only + AnnualCapacity_HIV_TB)
        else:
            AnnualNeed_PctCapacity_HIV_TB = (TestingNeed_HIV + TestingNeed_TB) / (Total_HIV_tests_performed + Total_TB_tests_performed)

        NationalStats['AnnualNeed_PctCapacity_HIV_TB'] = AnnualNeed_PctCapacity_HIV_TB
        
    # Check if picking the right folder
    list_files_in_folder("1LJo-H0ToFc6igpX6gYR8dYUSgK1coVXH")

    # Usage example: Replace 'sample.geojson' with the name of the file you want to load
    file_name = st.session_state.TargetCountry + '.geojson'  # Ensure this is the correct filename
    geo_df = load_geojson(file_name, '1LJo-H0ToFc6igpX6gYR8dYUSgK1coVXH')

    #if geo_df is not None:
    #	    st.write(geo_df)
    #else:
    #	    st.write("File not found.")
    ## Import geojson file as geopandas dataframe
    #geo_df = gpd.read_file('geojson/' + st.session_state.TargetCountry + '.geojson')

    ## Subset by regions selected
    geo_df = geo_df[geo_df['admin_name'].isin(st.session_state.RegionsSelected)]

    ColsTemp = ['admin_name',
        'AnnualTests_PctCapacity_HIV', 
        'AnnualTests_PctCapacity_TB',
        'AnnualTests_PctCapacity_HIV_TB',
        'AnnualNeed_PctCapacity_HIV',
        'AnnualNeed_PctCapacity_TB',
        'AnnualNeed_PctCapacity_HIV_TB'
        ]

    IGNORE = ['admin_name']
    for col in ColsTemp:
        if col not in IGNORE:
            CapacityNeed[col].replace([np.inf, -np.inf], np.nan, inplace=True)

    JoinCols = [x for x in ColsTemp if (x not in IGNORE)]
    JoinCols.append(*IGNORE)

    geo_df = geo_df.reset_index().merge(CapacityNeed[JoinCols], on='admin_name', how='left')
    ColsTemp.append('geometry')
    geo_df = geo_df[ColsTemp]

    ### Show maps
    ## Dictionary with column names as keys and html for display of map titles as values
    TitleDict = {
        'AnnualTests_PctCapacity_HIV' : """<p style="text-align:left;">Annual MolDx <strong>HIV Tests</strong> as a Percentage<br>of MolDx <strong>HIV Testing Capacity</strong></p6>""", 
        'AnnualTests_PctCapacity_TB' : """<p style="text-align:left;">Annual MolDx <strong>TB Tests</strong> as a Percentage<br>of MolDx <strong>TB Testing Capacity</strong></p6>""",
        'AnnualTests_PctCapacity_HIV_TB' : """<p style="text-align:left;">Annual MolDx <strong>HIV and TB Tests</strong> as a Percentage<br>of MolDx <strong>HIV and TB Testing Capacity</strong></p6>""",
        'AnnualNeed_PctCapacity_HIV' : """<p style="text-align:left;">Annual MolDx <strong>HIV Testing Need</strong> as a Percentage<br>of MolDx <strong>HIV Testing Capacity</strong></p6>""", 
        'AnnualNeed_PctCapacity_TB' : """<p style="text-align:left;">Annual MolDx <strong>TB Testing Need</strong> as a Percentage<br>of MolDx <strong>TB Testing Capacity</strong></p6>""",
        'AnnualNeed_PctCapacity_HIV_TB' : """<p style="text-align:left;">Annual MolDx <strong>HIV and TB Testing Need</strong> as a Percentage<br>of MolDx <strong>HIV and TB Testing Capacity</strong></p6>""",
        }

    def format_percentage(x, num_decimals):
        if isinstance(x, (int, float)) and not np.isnan(x):
            return f"{x:.{num_decimals}%}"
        else:
            return x

    ## Determine bin size for color scale
    bins = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 100]

    ## Define center coordinates for country
    lat = EligibleCountries.at[TargetCountry, 'center_lat']
    lon = EligibleCountries.at[TargetCountry, 'center_lon']

    ## Create maps
    zoom = int(EligibleCountries.at[TargetCountry, 'map_zoom'])
    map_dict = {}
    for col_name, title in TitleDict.items():

        # Create the base map centered on a latitude and longitude
        m = folium.Map(
            location=[lat, lon], 
            zoom_start=zoom,
            tiles=None,
            )
        
        # Add the map tiles to the base map
        folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(m)

        # Create the choropleth map
        c = folium.Choropleth(
            geo_data=geo_df,
            name="choropleth",
            data=geo_df,
            columns=["admin_name", col_name],
            key_on="feature.properties.admin_name",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.4,
            line_weight=1.5,
            smooth_factor=1,
            bins=bins,
            reset=True,
            nan_fill_color='black',
            nan_fill_opacity=1
        )
        
        # Remove the legend from the choropleth map
        for key in c._children:
            if key.startswith('color_map'):
                del(c._children[key])

        # Add the choropleth map to the base map        
        c.add_to(m)

        # Define functions for the interactive layer
        style_function = lambda x: {'fillColor': '#ffffff', 
                                    'color':'#000000', 
                                    'fillOpacity': 0.1, 
                                    'weight': 0.1}
        
        highlight_function = lambda x: {'fillColor': '#000000', 
                                        'color':'#000000', 
                                        'fillOpacity': 0.50, 
                                        'weight': 0.1}


        # Convert the GeoDataFrame to a JSON object
        geo_json_data = json.loads(geo_df.to_json())

        # Format as percentage, for display in the interactive layer
        for feature in geo_json_data['features']:
            feature['properties'][col_name] = format_percentage(feature['properties'][col_name], 1)

        # Create tooltip (popup for interactive layer)
        tooltip = folium.features.GeoJsonTooltip(
            fields=['admin_name', col_name],
            aliases=['Region: ', 'Percentage of Capacity: '],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),
            localize=True,
            sticky=False
        )

        # Create the interactive layer
        INTERACTIVE = folium.features.GeoJson(
            geo_json_data,
            style_function=style_function,
            control=False,
            highlight_function=highlight_function,
            tooltip=tooltip
        )

        # Add the interactive layer to the base map
        m.add_child(INTERACTIVE)

        # Ensure that the interactive layer is always on top
        m.keep_in_front(INTERACTIVE)

        # Add the map and title to the dictionary
        map_dict[title] = m

    # Create a custom legend applicable to all maps
    legend_html = """
    <div style="position: relative; width: 100%; height: 40px; background-color: white; border:2px solid grey; z-index:9999; font-size:14px; display: flex; flex-direction: column;">
        <div style="margin-bottom: 10px;"></div>
        <div style="display: flex; flex-direction: row; justify-content: space-around;">
    """

    colormap = ['#FEFECC', '#FDE6A6', '#FDCD8C', '#FCB682', '#FC8E76', '#EC6D6D', '#CC5D73']

    for i, bin_value in enumerate(bins[:-1]):
        bin_start = bin_value
        bin_end = bins[i + 1]
        
        # Label the final bin differently
        if i == len(bins) - 2:
            bin_end_str = "≥150%"
        else:
            bin_end_str = format_percentage(bin_end, 0)
        
        # For the final bin, don't show the start percentage
        if i == len(bins) - 2:
            legend_html += f'<div style="display: flex; align-items: center;"><i style="background-color: {colormap[i]}; width: 18px; height: 18px; margin-right: 8px; opacity: 1;"></i> {bin_end_str}</div>'
        else:
            legend_html += f'<div style="display: flex; align-items: center;"><i style="background-color: {colormap[i]}; width: 18px; height: 18px; margin-right: 8px; opacity: 1;"></i> {format_percentage(bin_start, 0)} to <{bin_end_str}</div>'

    legend_html += '<div style="display: flex; align-items: center;"><i style="background-color: black; width: 18px; height: 18px; margin-right: 8px; opacity: 0.7;"></i> No known testing capacity for disease(s)</div>'

    legend_html += "</div></div>"

    # Create containers and columns.  Display titles, maps, and legend
    map_height = 400
    map_width = 400
    vertical_space = 2

    with st.container():
        columns1 = st.columns(3)
        for i, (title, map) in enumerate(map_dict.items()):
            if i < 3:
                with columns1[i]:
                    st.write(title, unsafe_allow_html=True)
                    folium_static(map, height=map_height, width=map_width)
        st.write(legend_html, unsafe_allow_html=True)
        add_vertical_space(vertical_space)

    with st.container():
        columns2 = st.columns(3)
        for i, (title, map) in enumerate(map_dict.items()):
            if i >= 3:
                with columns2[i - 3]:
                    st.write(title, unsafe_allow_html=True)
                    folium_static(map, height=map_height, width=map_width)
                        
        add_vertical_space(vertical_space)
        st.caption("""These statistics are based on the capabilities of available machines (e.g., "Can a machine run a TB test?") not ownership (e.g., "Is the machine owned by the National TB Program?").  Capacity of machines that can perform both HIV and TB tests contributes to the denominators of both the HIV and TB individual maps (i.e., it is "double counted").  This capacity is counted only once for the combined HIV and TB maps.  In this simple analysis, capacity for a region can only serve testing demand within that same region.  The only exception is the National-level table below, which pools tests and capacity across regions.""")
        add_vertical_space(vertical_space)

    ### Create and display summary tables
    HeaderDict = {
        'AnnualCapacity_HIV_Only' : 'Annual Capacity of MolDx Machines Only Usable for HIV Testing (# Tests)',
        'AnnualCapacity_TB_Only' : 'Annual Capacity of MolDx Machines Only Usable for TB Testing (# Tests)',
        'AnnualCapacity_HIV_TB' : 'Annual Capacity of MolDx Machines Usable for Both HIV and TB Testing (# Tests)',
        'AnnualTests_HIV' : 'Annual MolDx HIV Tests at Baseline',
        'AnnualTests_TB' : 'Annual MolDx TB Tests at Baseline',
        'TestingNeed_HIV' : 'Annual MolDx HIV Testing Need, Estimated',
        'TestingNeed_TB' : 'Annual MolDx TB Testing Need, Estimated',
        'AnnualTests_PctCapacity_HIV' : 'Annual MolDx HIV Tests as a Percentage of MolDx HIV Testing Capacity', 
        'AnnualTests_PctCapacity_TB' :'Annual MolDx TB Tests as a Percentage of MolDx TB Testing Capacity',
        'AnnualTests_PctCapacity_HIV_TB' : 'Annual MolDx HIV and TB Tests as a Percentage of MolDx HIV and TB Testing Capacity',
        'AnnualNeed_PctCapacity_HIV' : 'Annual MolDx HIV Testing Need as a Percentage of MolDx HIV Testing Capacity',
        'AnnualNeed_PctCapacity_TB' : 'Annual MolDx TB Testing Need as a Percentage of MolDx TB Testing Capacity',
        'AnnualNeed_PctCapacity_HIV_TB' : 'Annual MolDx HIV and TB Testing Need as a Percentage of MolDx HIV and TB Testing Capacity'
        }

    format_mapping = {
        "{:.0%}": ['AnnualTests_PctCapacity_HIV', 'AnnualTests_PctCapacity_TB', 'AnnualTests_PctCapacity_HIV_TB', 'AnnualNeed_PctCapacity_HIV', 'AnnualNeed_PctCapacity_TB', 'AnnualNeed_PctCapacity_HIV_TB'],
        "{:,.0f}": ['AnnualCapacity_HIV_Only', 'AnnualCapacity_TB_Only', 'AnnualCapacity_HIV_TB', 'AnnualTests_HIV', 'AnnualTests_TB', 'TestingNeed_HIV', 'TestingNeed_TB']
    }

    ## National table
    summary_national = pd.DataFrame(NationalStats, index=["National (Selected Regions)"])

    for format_string, columns in format_mapping.items():
        for column in columns:
            try:
                summary_national[column] = summary_national[column].apply(lambda x: format_string.format(x) if pd.notnull(x) else "")
            except:
                pass

    summary_national.rename(columns=HeaderDict, inplace=True)

    ## Subnational tables
    merge_cols = ['admin_name', 'AnnualCapacity_HIV_Only', 'AnnualCapacity_TB_Only', 'AnnualCapacity_HIV_TB', 'AnnualTests_HIV', 'AnnualTests_TB', 'TestingNeed_HIV', 'TestingNeed_TB']
    summary_table = geo_df.drop('geometry', axis=1)
    summary_table = summary_table.merge(CapacityNeed[merge_cols], on='admin_name', how='left')

    # Fill in values of testing need for subnational regions that were selected but do not have machine capacity.
    summary_table.set_index('admin_name', inplace=True)
    DiseaseBurden_Target.set_index('admin_name', inplace=True)
    summary_table['TestingNeed_HIV'] = summary_table['TestingNeed_HIV'].fillna(DiseaseBurden_Target['TestingNeed_HIV'])
    summary_table['TestingNeed_TB'] = summary_table['TestingNeed_TB'].fillna(DiseaseBurden_Target['TestingNeed_TB'])

    # Apply formatting
    for format_string, columns in format_mapping.items():
        for column in columns:
            summary_table[column] = summary_table[column].apply(lambda x: format_string.format(x) if pd.notnull(x) else "")

    # Split into two tables
    summary_table.rename(columns=HeaderDict, inplace=True)
    summary_table.index.rename('Region', inplace=True)
    summary_table = summary_table.sort_index()
    summary_table1 = summary_table.iloc[:, :6]
    summary_table2 = summary_table.iloc[:, 6:]

    def convert_df_to_csv(df):
        return df.to_csv(index=True).encode('utf-8')

    # Display all summary tables
    with st.container():
        # Display tables
        st.table(summary_national)
        st.download_button(
            label="Download National Summary as CSV",
            data=convert_df_to_csv(summary_national),
            file_name="summary_national.csv",
            mime="text/csv"
        )

        st.table(summary_table1)
        st.download_button(
            label="Download Table 1 as CSV",
            data=convert_df_to_csv(summary_table1),
            file_name="summary_table1.csv",
            mime="text/csv"
        )

        st.table(summary_table2)
        st.download_button(
            label="Download Table 2 as CSV",
            data=convert_df_to_csv(summary_table2),
            file_name="summary_table2.csv",
            mime="text/csv"
        )

        if TestingNeedAdjusted:
            st.caption(f'The number of tests at baseline is higher than estimated testing need for at least one region and one disease.  For the following region-disease combinations, the original testing need estimates were replaced with the number of tests performed at baseline.')
            for disease, regions in RegionsAdjusted.items():
                if regions:
                    st.caption(f'{disease}: {", ".join(regions)}')

    ### Display navigation buttons            
    with st.container():
        add_vertical_space(1)
        col1, _, _ = st.columns(st.session_state.NavButtonCols)

        with col1:
            if st.button('Back', use_container_width=True):
                switch_page('verify inputs')
