import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import itertools
from datetime import datetime

st.set_page_config(layout="wide")
st.markdown('<style>body{background-color: Black;}</style>',unsafe_allow_html=True)
#####plotting functions
def create_model_plot(dataset, x, y,title_name, yaxis_title, yaxis_tickformat, model_plot_scale_min, model_plot_scale_max,color_choice = 'skyblue'):
    point_space = np.linspace(model_plot_scale_min, model_plot_scale_max, 4)
    model_options = ['Random Forest', 'Logistic Regression', 'Convolutional Neural Network', 'Manual Labels (~30% of photos)']
    model_col_names = ['rf_pred', 'logit_pred', 'cnn_pred', 'Label_train']
    model_stagger = {key: value for key, value in zip(model_col_names, point_space)}

    for col in model_col_names:
        new_col_name = col + '_stagger'
        dataset[new_col_name] = dataset[col].map({0: np.nan, 1: model_stagger[col]})

    fig = px.line(
        dataset,
        x=x,
        y=y,
        title=title_name,
        labels={y: yaxis_title},
        color_discrete_sequence=[color_choice]
    )

    marker_settings = dict(size=10)
    for col, name, color in zip(model_col_names, model_options, ['purple', 'mediumvioletred', 'red', 'darkgreen']):
        scatter_symbol = 'circle' if col != 'Label_train' else 'triangle-up'
        fig.add_scatter(
            x=dataset[x],
            y=dataset[col + '_stagger'],
            mode='markers',
            name=name ,
            marker=dict(symbol=scatter_symbol, color=color, **marker_settings)
        )

    fig.update_layout(
        xaxis_title='Date (half hourly)',
        yaxis_title=yaxis_title,
        yaxis_tickformat=yaxis_tickformat,
        xaxis_tickangle=-45,
        legend_title_text='Foggy Classification Method',
        # legend_font_size=14,
        height=500,
        legend=dict(
            y=1,
            x=0.5,
            xanchor='center',
            yanchor='bottom',
            orientation='h'
        )
    )

    return fig
def create_summarized_heatmap(dataset, index, columns, values, title_name,colorscale='coolwarm'):
    # Pivot the dataset to create a comparison matrix
    compare_cooccure = dataset.pivot(index=index, columns=columns, values=values).reset_index()

    # Initialize the co-occurrence matrix with zeros
    matrix_df = pd.DataFrame(index=dataset[columns].unique(), columns=dataset[columns].unique())

    # Generate all unique pairs of columns and calculate the percentage of time both equal 1
    for pair in itertools.permutations(matrix_df.index, 2):
        if pair[0] != pair[1]:  # Exclude self-comparison to avoid a symmetrical heatmap
                matrix_df.at[pair[0], pair[1]]  = (compare_cooccure.loc[compare_cooccure[pair[0]] == 1, list(pair)].prod(axis=1) == 1).mean() 
    matrix_df.fillna(1, inplace=True)

    # Create the heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=matrix_df.values,
        x=matrix_df.columns,
        y=matrix_df.index,
        colorscale=colorscale,
        zmin=0,  # Set the scale minimum to 0
        zmax=1,  # Set the scale maximum to 1, adjust as needed
    ))
    
    # Update layout to set the title and make the plot square
    fig.update_layout(
        title_text= title_name,
        height=600,
        width=600,
        autosize=False
    )

    # Update axes to be equal (square)
    fig.update_xaxes(scaleanchor='y', scaleratio=1)

    return fig

###read in data and make the adjustments needed for plotting
fog_results = pd.read_csv('https://raw.githubusercontent.com/son-ra/Fog_Project_Code/main/streamlit/streamlit_dataset.csv')
fog_results = fog_results.loc[fog_results['site']!= 'Beverly']
asos = pd.read_csv('https://raw.githubusercontent.com/son-ra/Fog_Project_Code/main/streamlit/asos_oregon.csv')

###date formatting
fog_results['time_pst'] = pd.to_datetime(fog_results['time_pst'])
fog_results['date'] = pd.to_datetime(fog_results['date']).dt.date
###put in other units
fog_results['relh'] = fog_results['relh']/100
fog_results['vsby'] = fog_results['vsby']/1.60934 ##miles to kms
fog_results['sknt'] = fog_results['sknt'] * 0.51444 ##knots to m/s
fog_results['tmpf'] = (fog_results['tmpf'] -32)* 5/9 ##F to C

###function for plotting all the weather vars


### sidebar selection 
# Select date ranges
##select sites

model_options = ['Random Forest', 'Logistic Regression', 'Convolutional Neural Network']
selected_model = st.sidebar.selectbox(
     'Choose one model to view its results for image-based fog classifcation:'
     , model_options, placeholder='Random Forest')

model_col_names = {'Random Forest': 'rf_pred', 'Logistic Regression': 'logit_pred', 'Convolutional Neural Network': 'cnn_pred'}
selected_col_name = model_col_names[selected_model]
fog_results['model_show'] = fog_results[selected_col_name]

min_date, max_date = fog_results['time_pst'].min(), fog_results['time_pst'].max()

# Define the default start and end dates for the date_input
default_start_date_day = datetime(2023, 7, 1)  # Default start date
default_end_date_day = datetime(2023, 9, 30)   # Default end date

# Define the default start and end dates for the date_input
default_start_date_hourly = datetime(2023, 7, 1)  # Default start date
default_end_date_hourly = datetime(2023, 7, 31)   # Default end date

# Using the date_input Streamlit function to create a date range picker
start_date, end_date = st.sidebar.date_input(
    "Select date range in summer months to view camera location comparison plots:",
    [default_start_date_day, default_end_date_day],
    min_value=min_date,
    max_value=max_date
)

##select sites
unique_sites = fog_results['site'].unique()
selected_sites = st.sidebar.selectbox('Choose one camera location to compare fog classifcation to weather metrics from the closest ASOS station:'
                                      , unique_sites, placeholder='Agate Beach')

start_date_hourly, end_date_hourly = st.sidebar.date_input(
    "Select date range in summer months for half hourly plots:",
    [default_start_date_hourly, default_end_date_hourly],
    min_value=min_date,
    max_value=max_date
)


# # Select traces
# selected_traces = st.sidebar.multiselect(
#     'Select models to include for weather metric comparison:',
#      options=model_options,
#     default=model_options
# )


###create filtered datasets

# Filter data by the selected hourly date range to see timeseries of all sites, only one model
filtered_data_daily = fog_results.loc[
    (fog_results['time_pst'].dt.date >= start_date) & (fog_results['time_pst'].dt.date <= end_date)
]
##summarize by day
filtered_data_daily = filtered_data_daily.groupby(['site','date', 'year', 'month'])['model_show'].sum().reset_index()
filtered_data_daily['fog_day']=0
filtered_data_daily.loc[filtered_data_daily['model_show']>0, 'fog_day'] = 1


# Filter data by the selected daily date range to see comparison charts, only one model
filtered_data_hourly = fog_results.loc[
    (fog_results['time_pst'].dt.date >= start_date_hourly) & (fog_results['time_pst'].dt.date <= end_date_hourly)
]


##filter by hourly range and site for weather charts
filtered_data_hourly_weather = fog_results.loc[
    (fog_results['time_pst'].dt.date >= start_date_hourly) & (fog_results['time_pst'].dt.date <= end_date_hourly)
    # & (fog_results['relh'].notna()) 
    & (fog_results['site']==selected_sites)
]


colorscale = 'Darkmint'
hourly_cooccur_heatmap = create_summarized_heatmap(filtered_data_hourly
                                                   , 'time_pst'
                                                   , 'site'
                                                   , 'model_show'
                                                   ,'Percent of Fog Co-occurance by Camera Location - Half Hourly'
                                                   , colorscale)

colorscale = 'Purpor'
daily_cooccur_heatmap = create_summarized_heatmap(filtered_data_daily
                                                  , 'date'
                                                  , 'site'
                                                  , 'fog_day'
                                                  ,'Percent of Fog Co-occurance by Camera Location - Daily'
                                                  , colorscale)




lineplot_daily_by_site = px.line(filtered_data_daily,
                 title = 'Count of Half Hour Daytime Fog Identifications per Day by Camera Location',
                 x='date',
                 y='model_show',
                 color='site',
                 symbol='site'
                 )
lineplot_daily_by_site.update_layout(
        xaxis_title='Day'
        ,yaxis_title = 'Count of Half Hour Fog Classifications')


scatter_hourly_by_site = px.scatter(filtered_data_hourly,
                title = 'Half Hourly Fog Classification by Camera Location',
                 x='time_pst',
                 y='rf_pred_staggered_site',
                 color='site',
                 symbol='site',
                 size_max=200)
scatter_hourly_by_site.update_layout(
        xaxis_title='Date (half hourly)')
# Update layout to have the y-axis tick labels blank
scatter_hourly_by_site.update_yaxes(showticklabels=False)

# Remove y-axis label (if any)
scatter_hourly_by_site.update_yaxes(title=None)


# Filter the data where 'rf_pred' is 1
foggy_only = filtered_data_hourly.loc[filtered_data_hourly['rf_pred'] == 1]
label_only = fog_results.loc[fog_results['Label_train'] == 1] ##static for the whole timeperiod

# Create the facet grid using Plotly Express
hourly_hist_by_site = px.histogram(foggy_only, 
                   x='hour', 
                   color='site', 
                   facet_col='site', 
                   facet_col_wrap=2, 
                   category_orders={"site": foggy_only['site'].unique()},
                   histnorm='percent')

hourly_hist_by_site.update_layout(
    title_text='Distribution of Hours by Camera Location', # Set plot title
    xaxis_title_text='Hour', # Set x-axis title
    yaxis_title_text='Percent', # Set y-axis title
)

# Create the facet grid using Plotly Express
hourly_hist_by_site_labels = px.histogram(label_only, 
                   x='hour', 
                   color='site', 
                   facet_col='site', 
                   facet_col_wrap=2, 
                   category_orders={"site": label_only['site'].unique()},
                   histnorm='percent')

hourly_hist_by_site_labels.update_layout(
    title_text='Labeled Dataset - Distribution of Hours by Camera Location\r\n For Entire Timeperiod', # Set plot title
    xaxis_title_text='Hour', # Set x-axis title
    yaxis_title_text='Percent', # Set y-axis title
)

station = filtered_data_hourly_weather['station'].unique()
station = ", ".join(station)  # Join all stations into a single string
# Create the main line plot for Relative Humidity


relh_plot = create_model_plot(
    dataset=filtered_data_hourly_weather,
    x='time_pst',
    y='relh',
    title_name='Relative Humidity',
    yaxis_title='Relative Humidity (%)',
    yaxis_tickformat='.0%',
    model_plot_scale_min=1.0,
    model_plot_scale_max=1.1
)

visibility_plot = create_model_plot(
    dataset=filtered_data_hourly_weather,
    x='time_pst',
    y='vsby',
    title_name = 'Visbility', 
    yaxis_title='Visbility (kilometers)',
    yaxis_tickformat='0',
    model_plot_scale_min=0,
    model_plot_scale_max=0.7
)
visibility_plot.add_hline(y=1, line=dict(color='black', dash='dash'))

temperature_plot = create_model_plot(
    dataset=filtered_data_hourly_weather.loc[filtered_data_hourly_weather['tmpf']>0],
    x='time_pst',
    y='tmpf',
    title_name = 'Temperature', 
    yaxis_title='Temperature (deg C)',
    yaxis_tickformat='0',
    model_plot_scale_min=0,
    model_plot_scale_max=5
  ,color_choice = 'lightyellow'
)

wind_speed_plot = create_model_plot(
    dataset=filtered_data_hourly_weather,
    x='time_pst',
    y='sknt',
    title_name = 'Wind Speed', 
    yaxis_title='Wind Speed (m/s)',
    yaxis_tickformat='0',
    model_plot_scale_min=11,
    model_plot_scale_max=15
  ,color_choice = 'lightyellow'
)

wind_direction_plot = create_model_plot(
    dataset=filtered_data_hourly_weather,
    x='time_pst',
    y='drct',
    title_name = 'Wind Direction', 
    yaxis_title='Wind Direction (Degrees from N)',
    yaxis_tickformat='0',
    model_plot_scale_min=370,
    model_plot_scale_max=395
  ,color_choice = 'lightyellow'
)

asos_loc = asos[['station', 'lon', 'lat']].drop_duplicates()
asos_loc.columns = ['Site Name', 'lon', 'lat']
asos_loc['Site Name'] = 'ASOS Station ' + asos_loc['Site Name'] 
surfline_loc = fog_results[['site','lon_x','lat_x',]].drop_duplicates()
surfline_loc.columns = ['Site Name', 'lon', 'lat']
surfline_loc['Site Name'] = 'Surfline Camera at ' + surfline_loc['Site Name'] 
plot_data = pd.concat([asos_loc, surfline_loc])
# st.map(surfline)

# # Convert your 'Site Name' column to string if it's not already to avoid issues with PyDeck
# plot_data['Site Name'] = plot_data['Site Name'].astype(str)
# Assign unique colors to unique 'Site Name' values
site_names = plot_data['Site Name'].unique()
colors1 = sns.color_palette('winter', len(asos_loc)).as_hex()  # Using seaborn color palette
colors2 = sns.color_palette('autumn', len(surfline_loc)).as_hex()  # Using seaborn color palette
colors = colors1+colors2
color_map = dict(zip(site_names, colors))  # Create a mapping from 'Site Name' to color
# # Now apply this mapping to the 'Site Name' column to create a new column for colors
plot_data['color'] = plot_data['Site Name'].apply(lambda x: color_map[x])

# # Convert hex colors to RGB (as PyDeck expects RGB or RGBA values)
plot_data['color'] = plot_data['color'].apply(lambda hex_color: list(int(hex_color[i:i+2], 16) for i in (1, 3, 5)) + [255])

# # Define the layer for the scatterplot
# layer = pdk.Layer(
#     'ScatterplotLayer',     # Type of layer to use
#     plot_data,              # DataFrame containing your data
#     get_position='[lon, lat]',     # Define the position [longitude, latitude]
#     get_color='color', # RGBA color of the points
#    get_radius=2000,         # Radius of the points
# #    get_icon='icon_data',  # Expects a dictionary containing 'url', 'width', 'height', 'anchorY'
# #     size_scale=10,
#      pickable=True           # Make points clickable for more info
# )

# # Set the initial view for the map
# view_state = pdk.ViewState(
#     latitude=plot_data['lat'].mean(),
#     longitude=plot_data['lon'].mean(),
#     zoom=7,
#     pitch=0
# )
# # Create the deck.gl map using PyDeck
# r = pdk.Deck(
#     layers=[layer],
#     initial_view_state=view_state,
#     map_style='mapbox://styles/mapbox/light-v9',
#     tooltip={'text': '{Site Name}\n {Source}'}  # Customize the tooltip
# )
px.set_mapbox_access_token('pk.eyJ1Ijoic21tcnJyIiwiYSI6ImNsc2NvdnhzZzBycXoya3FsM3ZpeWYyZGEifQ.QRsn6gICJBD7xEAUIPKd-g')

map = px.scatter_mapbox(plot_data,
                        lat="lat",
                        lon="lon",
                        color=plot_data['Site Name'],
                        # size_max=50,
                        zoom=6,
                        size=[2] * len(plot_data),  # List of sizes, all same value to make dots larger                        mapbox_style="open-street-map",
                        title="Site Map",
                        hover_name='Site Name',
                        hover_data={'lon': True, 'lat': True},
                        color_discrete_map =color_map )

# Custom settings for the map
map.update_layout(
    margin={"r":0,"t":0,"l":0,"b":0},
    mapbox=dict(
        center=dict(lat=plot_data['lat'].mean(), lon=plot_data['lon'].mean()),
        zoom=6
    ),
    # Other customizations can go here
)
# Create the summary DataFrame using groupby
summary_df = fog_results.groupby(['site', 'year'])['model_show'].mean().reset_index().merge(
    fog_results.groupby(['site', 'year'])['date'].agg(['min', 'max']), on = ['site', 'year']
)
summary_df['model_show'] = (summary_df['model_show'] * 100).astype(int).astype(str) + '%'
summary_df.rename(columns={'model_show': '% Fog','min': 'Start', 'max': 'End'}, inplace=True)
summary_df['Start'] = pd.to_datetime(summary_df['Start']).dt.strftime('%b %d')
summary_df['End'] = pd.to_datetime(summary_df['End']).dt.strftime('%b %d')
summary_df['year'] = summary_df['year'].astype(str)
# Now format the DataFrame as required, for example:
# summary_df['Photo Collection Start Date'] = summary_df['Photo Collection Start Date'].dt.strftime('%Y-%m-%d')
# summary_df['Photo Collection End Date'] = summary_df['Photo Collection End Date'].dt.strftime('%Y-%m-%d')


###set up page configs and layout
st.title('Summertime Fog Presence Classified using Beach Cameras Along the Oregon Coast')
col1, col2 = st.columns([0.6, 0.4])

with col1:
     
    st.text("""Map of Camera Locations and Automated Surface Observing System \nWeather Stations at Airports""")
    st.plotly_chart(map, use_container_width=True)
with col2:
    st.text("""Table of the Start and End of Seasonal\nPhoto Collection by Camera Location\nand Seasonal % of Daytime Fog Hours""")
    st.dataframe(summary_df, hide_index = True, column_config=st.column_config.NumberColumn('year', format='$0'))

st.text("""We trained three different types of classification models on 8,000 manually labeled photos to detect the 
presence of coastal fog. For the five different camera locations, we collected an image every 30 minutes during the 
daytime local hours of 7am to 8pm. Each model correctly classified that an image contained fog ~80% of the time. The 
charts below shows the number of 30 minute daytime periods per day that were classified as foggy. In the sidebar panel, 
you can select which classification model to view as well as the time period. Please only select time periods which are 
within the ranges in the table above that shows the seasonal collection dates for 2022 and 2023.""")

st.plotly_chart(lineplot_daily_by_site, use_container_width=True)


st.text("""Below are some plots showing trends in fog presence at the five camera locations.

The fog co-occurence plots show the % of the time when there is fog at both camera locations. Please note that this co-occurence
metric is not the same when comparing, for example Pacific City vs. Otter Rock and Otter Rock vs. Pacific City because the 
denominator for Pacific City vs Otter Rock is the number of photos with fog present at Pacific City, whereas the denominator for 
Otter Rock and Pacific City is the number of photos with fog present at Otter Rock, which will be different. The chart on the left 
compares fog co-occurence for each half hour daytime period. The chart on the right shows the co-occurence by day, 
regardless of the time of day that the fog occured. 
        
Below the fog co-occurence charts are histograms of fog frequency by hour. The left shows frequency of the manually labeled 
photos and the right shows the fog frequency for the classified photos.        

Note that the classification model and time periods for these charts will be the same 
as selected for the chart above. 

Following the histograms is a chart of fog presence by site in half hour intervals. Please note that the date range for
this chart should be selected using the date range for half hourly plots, located on the sidebar to the left.""")

col3, col4 = st.columns([0.5, 0.5]) ##setting up two columns and their relative widths
### need to end these two columns for the weather data

with col3:

    st.plotly_chart(hourly_cooccur_heatmap, use_container_width=True)
    st.plotly_chart(hourly_hist_by_site_labels, use_container_width=True)

# Render the map

with col4:

# Display the plot hourly_cooccur_heatmap
    st.plotly_chart(daily_cooccur_heatmap, use_container_width=True)
    st.plotly_chart(hourly_hist_by_site, use_container_width=True)


st.plotly_chart(scatter_hourly_by_site, use_container_width=True)
st.subheader(f"Fog Classifications for {selected_sites} and Weather Observations from Closest Automated Surface Observing System {station} Station")
st.text("""The following charts compare the fog classification for all three models as well as the manual labels to weather observations from nearby airports. 
Around 30% of photos have a manual label. The charts of relative humidity and visibility can be used to validate the fog classification as relative humidity 
is directly related to fog, and visibility is a more precise measurement of fog presence. However, note that the distance from nearest weather station to 
camera location varies by site, and so the airport weather stations closer to the camera sites, like Agate Beach, will have a stronger relationship between 
weather observations and fog classificaiton. Also note that there is some missing data for weather observations, particularly for wind direction and speed.

Please select the camera location and time period for these half hourly plots in the sidebar to the left.""")
st.plotly_chart(relh_plot, use_container_width=True)
st.plotly_chart(visibility_plot, use_container_width=True)
st.plotly_chart(temperature_plot, use_container_width=True)
st.plotly_chart(wind_speed_plot, use_container_width=True)
st.plotly_chart(wind_direction_plot, use_container_width=True)

