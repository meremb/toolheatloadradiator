import pandas as pd
import streamlit as st
from plotly import express as px


def plot_pressure_loss(df: pd.DataFrame) -> None:
    """Plot the total pressure loss per collector using Plotly."""
    fig = px.bar(df, x='Radiator nr', y='Total Pressure Loss',
                 labels={'Radiator nr': 'Radiator nr', 'Total Pressure Loss': 'Total Pressure Loss'})
    fig.update_layout(title='Total Pressure Loss per radiator circuit',
                      xaxis_title='Radiator', yaxis_title='Total Pressure Loss')
    st.plotly_chart(fig)


def plot_thermostatic_valve_position(df: pd.DataFrame) -> None:
    """Plot the valve position for each radiator circuit with improved visualization."""
    # Convert valve positions to discrete values if needed
    df['Valve position'] = df['Valve position'].astype(int)

    # Create the bar plot
    fig = px.bar(
        df,
        x='Radiator nr',
        y='Valve position',
        color='Valve position',  # Color by valve position
        color_continuous_scale=px.colors.sequential.Greens,  # Use a sequential color scale
        labels={'Radiator nr': 'Radiator', 'Valve position': 'Valve Position'},
        title='Valve Position Required to Balance the Circuit',
        text='Valve position'  # Show the valve position on each bar
    )

    # Customize the layout
    fig.update_layout(
        xaxis_title='Radiator',
        yaxis_title='Valve Position',
        coloraxis_showscale=False  # Hide the color scale since it's redundant
    )

    # Update text display on the bars for better readability
    fig.update_traces(
        texttemplate='%{text}',
        textposition='outside',
        marker_line_color='black',  # Add black borders to the bars
        marker_line_width=1.5
    )

    # Display the plot
    st.plotly_chart(fig)


def plot_mass_flow_distribution(df: pd.DataFrame) -> None:
    """Plot a pie chart of mass flow rate distribution among radiators."""
    fig = px.pie(df, names='Radiator nr', values='Mass flow rate',
                 title='Mass Flow Rate Distribution Among Radiators',
                 labels={'Radiator nr': 'Radiator', 'Mass flow rate': 'Mass Flow Rate'})
    st.plotly_chart(fig)


def plot_temperature_heatmap(df: pd.DataFrame) -> None:
    """Plot a heatmap of supply and return temperatures across radiators."""
    # Prepare the data for the heatmap
    heatmap_data = df[['Radiator nr', 'Supply Temperature', 'Return Temperature']].set_index('Radiator nr')
    heatmap_data = heatmap_data.transpose()  # Transpose to have Temperature Type as rows

    # Create the heatmap using imshow
    fig = px.imshow(heatmap_data,
                    labels={'x': 'Radiator nr', 'y': 'Temperature Type', 'color': 'Temperature (°C)'},
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale='Bluered')

    fig.update_layout(title='Heatmap of Supply and Return Temperatures')
    st.plotly_chart(fig)
