import streamlit as st
import math
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# Define the Hall-Yarborough functions
def fy(y, alpha, Ppr, t):
    return (-alpha * Ppr +
            (y + y ** 2 + y ** 3 - y ** 4) / (1 - y) ** 3 -
            (14.76 * t - 9.76 * t ** 2 + 4.58 * t ** 3) * y ** 2 +
            (90.7 * t - 242.2 * t ** 2 + 42.4 * t ** 3) * y ** (2.18 + 2.82 * t))

def Zfac(Tpr, Ppr):
    t = 1 / Tpr
    alpha = 0.06125 * t * math.exp(-1.2 * (1 - t) ** 2)
    y_initial = 0.001  # Initial guess for y
    y = fsolve(fy, y_initial, args=(alpha, Ppr, t))[0]  # Solve for y using fsolve
    return alpha * Ppr / y

# Gas Expansion Factor calculation
def gas_expansion_factor(P, Z, T_rankine):
    return 35.37 * P / (Z * T_rankine)

# Gas Recovery Factor calculation
def grf(Ei, E):
    grf_value = ((Ei - E) / Ei) * 100
    if grf_value < 0:
        return 0
    return grf_value

# Gas Recovery Factor calculation - Properties tab
def p_z(P,Z):
    return P / Z

# Optional GIIP calculation - GIIP tab
# Function to calculate GIIP
def calculate_giip(area, thickness, porosity, connate_sw, Ei):
    return ((43560 * area * thickness * porosity * (1 - connate_sw) * Ei) / 10**9)

# Calculating Build-up Phase Gas Rate
def calculate_bug(PLG, BUY):
    # Initialize an empty list to store production values for each year
    production_years_bu = []
    
    # Calculate the production for the first year
    production_first_year = PLG / BUY
    production_years_bu.append(production_first_year)
    
    # Calculate the production for each subsequent year
    for year in range(1, BUY):
        production_current_year = production_years_bu[-1] + (PLG / BUY)
        # Exclude the year when PLG equals BUG
        if production_current_year != PLG:
            production_years_bu.append(production_current_year)
    
    # Calculate the summation of the BUG rates
    BUG_sum = sum(production_years_bu)
    
    # Calculate Gp1
    Gp1 = BUG_sum * 365.2 /1000
    
    return production_years_bu, Gp1

# Gas Recovery Factor calculation - Schedule tab
def grf_sch(df, column_name, constant, new_col_name):
    """  
    Divides a column in a DataFrame by a constant and adds the result as a new column.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the column.
    column_name (str): The name of the column to multiply.
    constant (float): The constant value to multiply the column by.
    new_col_name (str): The name of the new column to be added.
    
    Returns:
    pd.DataFrame: The DataFrame with the new column added.
    """
    # Ensure the column exists in the DataFrame
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in the DataFrame")

    # Perform the multiplication and add the new column so its in %
    df[new_col_name] = ( df[column_name] / constant ) * 100
    
    return df

# Recovery Factor Interpolation to get Pressures
def cubic_spline_interpolation(df_props, df_bu, x_col, y_col, interp_col):
    """
    Perform cubic spline interpolation on specified data points.
    
    Parameters:
    df_props (pd.DataFrame): DataFrame containing known data points.
    df_bu (pd.DataFrame): DataFrame containing points to interpolate.
    x_col (str): Column name for the x-axis values in df_props.
    y_col (str): Column name for the y-axis values in df_props.
    interp_col (str): Column name for the x-axis values in df_bu to interpolate.
    
    Returns:
    pd.DataFrame: df_bu with an added column of interpolated values.
    """
    #st.write("Initial df_props:")
    #st.write(df_props)
    
    #st.write("Initial df_bu:")
    #st.write(df_bu)
    
    # Sort the known data points based on the x_col in increasing order and remove duplicates
    df_props = df_props[df_props['Pressure (psi)'] <= Pi].sort_values(by=x_col)
    #st.write("Sorted and deduplicated df_props:")
    #st.write(df_props)
    
    # Extract the known data points
    x_known = df_props[x_col].to_numpy()
    y_known = df_props[y_col].to_numpy()
    #st.write(f"x_known ({x_col}):", x_known)
    #st.write(f"y_known ({y_col}):", y_known)
    
    # Create the cubic spline interpolator
    cs = CubicSpline(x_known, y_known)
    
    # Interpolate the values at specified points
    x_interp = df_bu[interp_col].to_numpy()
    y_interp = cs(x_interp)
    #st.write(f"x_interp ({interp_col}):", x_interp)
    #st.write(f"Interpolated {y_col}:", y_interp)
    
    # Add the interpolated values to df_bu
    df_bu[f'Interpolated Pressure (psi)'] = y_interp
    
    return df_bu

# Plateau Phase Calculations
def pl_gas_rate(df_bu, df_props, PLG, GIIP, PLP):
    """
    Adds new years to the DataFrame, calculates Gp, and updates interpolated pressure until a condition is met.
    
    Parameters:
    df_bu (pd.DataFrame): DataFrame containing the initial data.
    df_props (pd.DataFrame): DataFrame containing known data points for interpolation.
    PLG (float): Gas rate (MMscfd) to use for calculations.
    GIIP (float): Gas Initially In Place (Bcf).
    PLP (float): Pressure limit.
    
    Returns:
    pd.DataFrame: Updated DataFrame with additional years and interpolated pressures.
    """
    # Create a copy of df_bu to keep the original DataFrame unchanged
    df_pl = df_bu.copy()
    
    while True:
        # Add one year to the last year
        new_year = df_pl['Year'].iloc[-1] + 1
        
        # Calculate Bcf and add it to the last Gp (Bcf)
        pl_gp = df_pl['Gp (Bcf)'].iloc[-1] + (PLG * 365.2 / 1000)
        
        # Calculate the new Gas Recovery Factor (%)
        pl_grf = (pl_gp / GIIP) * 100
        
        # Create a new row
        new_row = {
            'Year': new_year,
            'Gas Rate (MMscfd)': PLG,
            'Gp (Bcf)': pl_gp,
            'Gas Recovery Factor (%)': pl_grf,
            'Interpolated Pressure (psi)': 0  # Placeholder, will be updated later
        }
        
        # Append the new row to df_pl
        df_pl = pd.concat([df_pl, pd.DataFrame([new_row])], ignore_index=True)
        
        # Update the interpolated pressure column
        df_pl = cubic_spline_interpolation(df_props, df_pl, 'Gas Recovery Factor (%)', 'Pressure (psi)', 'Gas Recovery Factor (%)')
        
        # Get the interpolated pressure for the new row
        interpolated_pressure = df_pl['Interpolated Pressure (psi)'].iloc[-1]
        
        # Check if the interpolated pressure is less than PLP (pressure limit)
        if interpolated_pressure < PLP:
            # Drop the last row and stop the calculation
            df_pl = df_pl[:-1]
            break
    
    return df_pl

# Decline Phase Calculations
def exp_decline(df_pl, df_props, DDR, DAP, GIIP, Pi):
    """
    Appends new years to the DataFrame, calculates the Gas Rate (MMscfd) using Exponential Decline,
    and updates interpolated pressure until a condition is met.
    
    Parameters:
    df_pl (pd.DataFrame): DataFrame containing the initial data.
    df_props (pd.DataFrame): DataFrame containing known data points for interpolation.
    DDR (float): Decline rate (constant).
    DAP (float): Pressure limit.
    GIIP (float): Gas Initially In Place (Bcf).
    Pi (float): Initial reservoir pressure.
    
    Returns:
    pd.DataFrame: New DataFrame (df_decline) with additional years and interpolated pressures.
    """
    # Create a copy of df_pl to keep the original DataFrame unchanged
    df_decline = df_pl.copy()
    
    while True:
        # Add one year to the last year
        new_year = df_decline['Year'].iloc[-1] + 1
        
        # Calculate the new Gas Rate (MMscfd) using Exponential Decline: Q = declining_gas_rate * e^(-bt)
        previous_year_gas_rate = df_decline['Gas Rate (MMscfd)'].iloc[-1]
        declining_gas_rate = previous_year_gas_rate * np.exp(-DDR * 1)  # t is 1 year
        
        # Calculate Bcf and add it to the last Gp (Bcf)
        declining_gp = df_decline['Gp (Bcf)'].iloc[-1] + (declining_gas_rate * 365.2 / 1000)
        
        # Calculate the new Gas Recovery Factor (%)
        declining_grf = (declining_gp / GIIP) * 100
        
        # Create a new row
        new_row = {
            'Year': new_year,
            'Gas Rate (MMscfd)': declining_gas_rate,
            'Gp (Bcf)': declining_gp,
            'Gas Recovery Factor (%)': declining_grf,
            'Interpolated Pressure (psi)': 0  # Placeholder, will be updated later
        }
        
        # Append the new row to df_decline
        df_decline = pd.concat([df_decline, pd.DataFrame([new_row])], ignore_index=True)
        
        # Update the interpolated pressure column
        df_decline = cubic_spline_interpolation(df_props, df_decline, 'Gas Recovery Factor (%)', 'Pressure (psi)', 'Gas Recovery Factor (%)')
        
        # Get the interpolated pressure for the new row
        interpolated_pressure = df_decline['Interpolated Pressure (psi)'].iloc[-1]
        
        # Check if the interpolated pressure is less than DAP (pressure limit)
        if interpolated_pressure < DAP:
            # Drop the last row and stop the calculation
            df_decline = df_decline[:-1]
            break
    
    return df_decline

# Adding Year 0 to the final result
def add_initial_row(df_decline):
    # Create a new row with all values set to 0
    new_row = pd.DataFrame({
        'Year': [0],
        'Gas Rate (MMscfd)': [0.0],
        'Gp (Bcf)': [0.0],
        'Gas Recovery Factor (%)': [0.0],
        'Interpolated Pressure (psi)': [Pi]
    })

    # Concatenate the new row with the existing DataFrame
    df_decline = pd.concat([new_row, df_decline]).reset_index(drop=True)
    
    
    
    return df_decline

# Define the function to plot the data
def plot_gas_rate_and_pressure(df_decline, PLG, Pi):
    # Function to determine the phase and set bar colors
    def determine_phase_and_highlight(df, PLG):
        plateau_reached = False

        def highlight_phases(row):
            nonlocal plateau_reached
            if row['Gas Rate (MMscfd)'] < PLG and not plateau_reached:
                return 'm'  # Changed build-up phase color to magenta
            elif row['Gas Rate (MMscfd)'] == PLG:
                plateau_reached = True
                return 'limegreen'  # Plateau phase color
            else:
                return 'red'  # Decline phase color remains the same

        df['Bar Color'] = df.apply(highlight_phases, axis=1)
        return df

    # Determine the phase and highlight
    df_decline = determine_phase_and_highlight(df_decline, PLG)

    # Set the style of seaborn
    sns.set_theme(style="ticks", palette="pastel")

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plotting the Gas Rate as bars on the y1 axis with transparency
    bars = sns.barplot(x='Year', y='Gas Rate (MMscfd)', data=df_decline, ax=ax1, alpha=0.6, dodge=False)

    # Set the colors for each bar
    for i, bar in enumerate(bars.patches):
        bar.set_color(df_decline['Bar Color'][i])

    # Create another y-axis to plot the Interpolated Pressure as points
    ax2 = ax1.twinx()
    sns.scatterplot(x='Year', y='Interpolated Pressure (psi)', data=df_decline, ax=ax2, color='r', s=100, label='Interpolated Pressure (psi)')

    # Set the labels for the axes
    ax1.set_xlabel('Year', color='black')
    ax1.set_ylabel('Gas Rate (MMscfd)', color='black')
    ax2.set_ylabel('Interpolated Pressure (psi)', color='black')

    # Set the title of the plot
    plt.title('Gas Rate and Interpolated Pressure Over Years')

    # Ensure all axes start at 0
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.set_xlim(left=0)

    # Set axis color to black
    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['bottom'].set_color('black')
    ax2.spines['top'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['bottom'].set_color('black')

    # Remove gridlines
    ax1.grid(False)
    ax2.grid(False)

    # Add labels to the bars
    for index, row in df_decline.iterrows():
        ax1.text(row['Year'], row['Gas Rate (MMscfd)'] - 10, f"{row['Gas Rate (MMscfd)']:.0f}", color='white', ha='center', va='top')

    # Add labels to the pressure points
    for index, row in df_decline.iterrows():
        ax2.annotate(f"{row['Interpolated Pressure (psi)']:.0f}", (row['Year'], row['Interpolated Pressure (psi)']),
                     textcoords="offset points", xytext=(5, 5), ha='left', color='black')

    # Adjust the legend to avoid overlap, placing them in the top right corner without frames
    pressure_legend = ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.95), frameon=False)

    # Show the plot in Streamlit
    st.pyplot(fig)

# Set the title and subtitle with custom HTML and CSS for styling
st.markdown("""
    <style>
    .title {
        font-family: 'Roboto', sans-serif;
        text-align: center;
        font-size: 36px;
        margin-top: 20px;
    }
    .subtitle {
        font-family: 'Roboto', sans-serif;
        text-align: center;
        font-size: 24px;
        margin-top: 10px;
    }
    .centered-text {
        font-family: 'Roboto', sans-serif;
        text-align: center;
        font-size: 18px;
        margin-top: 20px;
    }
    </style>
    <h1 class="title">Gas Production Scheduler</h1>
    """, unsafe_allow_html=True)

# Centered line using custom HTML and CSS
st.markdown("""
    <p class="centered-text">After LP. Dake (Chapter 1 - SOME BASIC CONCEPTS IN RESERVOIR ENGINEERING)</p>
    """, unsafe_allow_html=True)

# Add an image (update the path to your image)
st.image("images/screen.png", caption="Example of gas field development rate-time schedule", use_container_width=True)

# Sidebar selections
option = st.sidebar.radio(
    "Development Planning Inputs",
    ("Gas Properties", "Gas Originally In-place (GIIP)", "Production Schedule")
)

# Initialize session state for gas properties if not already done
session_variables = ['SG', 'Pi', 'T', 'y_H2S', 'y_CO2', 'y_N2', 'Ppc', 'Tpc', 'T_rankine', 'Tr', 'Zi', 'Z_PL', 'PLG', 'BUY', 'BUG', 'PLP', 'DPD', 'Ei']
for var in session_variables:
    if var not in st.session_state:
        st.session_state[var] = None

# Display content based on the selected option
if option == "Gas Properties":
    st.subheader("Gas Properties")
    st.write("Enter the properties of the gas here.")
    
    # Create columns for layout
    col1, col2 = st.columns([1, 3])
    
    with col2:
        col21, col22 = st.columns(2)
        
        with col21:
            # Adding input boxes in the first column
            st.session_state['SG'] = st.text_input("Gas Gravity (air) *", st.session_state['SG'])
            st.session_state['Pi'] = st.text_input("Reservoir Pressure (psi)", st.session_state['Pi'])
            st.session_state['T'] = st.text_input("Reservoir Temperature (deg F)", st.session_state['T'])
            st.write("* Acceptable Range for Gas Gravity is 0.55 - 1.0")
        
        with col22:
            # Adding input boxes in the second column
            st.session_state['y_H2S'] = st.text_input("H2S content (mole frac):", st.session_state['y_H2S'])
            st.session_state['y_CO2'] = st.text_input("CO2 content (mole frac):", st.session_state['y_CO2'])
            st.session_state['y_N2'] = st.text_input("N2 content (mole frac):", st.session_state['y_N2'])
            

    # Validate Gas Gravity input
    SG_valid = False
    if st.session_state['SG']:
        try:
            SG = float(st.session_state['SG'])
            if 0.55 <= SG <= 1.0:
                SG_valid = True
            else:
                st.error("Gas Gravity must be between 0.55 and 1.0")
        except ValueError:
            st.error("Please enter a valid number for Gas Gravity")

    # Validate Pressure input
    P_valid = False
    if st.session_state['Pi']:
        try:
            Pi = float(st.session_state['Pi'])
            if Pi >= 14.7:
                P_valid = True
            else:
                st.error("Pressure must be equal to or above pressure at standard conditions (14.7 psi).")
        except ValueError:
            st.error("Please enter a valid number for Pressure.")

    # Validate Temperature input
    T_valid = False
    if st.session_state['T']:
        try:
            T = float(st.session_state['T'])
            if T >= 60:
                T_valid = True
            else:
                st.error("Temperature must be equal to or above temperature at standard conditions (60 deg F).")
        except ValueError:
            st.error("Please enter a valid number for Temperature.")

    # Initialize variables to store the mole fractions
    y_H2S_val, y_CO2_val, y_N2_val = 0, 0, 0

    # Validate the inputs
    if st.session_state['y_H2S']:
        try:
            y_H2S_val = float(st.session_state['y_H2S'])
            if not (0 <= y_H2S_val <= 1):
                st.error("H2S content must be between 0 and 1")
                y_H2S_val = 0
        except ValueError:
            st.error("Please enter a valid number for H2S content")
            y_H2S_val = 0

    if st.session_state['y_CO2']:
        try:
            y_CO2_val = float(st.session_state['y_CO2'])
            if not (0 <= y_CO2_val <= 1):
                st.error("CO2 content must be between 0 and 1")
                y_CO2_val = 0
        except ValueError:
            st.error("Please enter a valid number for CO2 content")
            y_CO2_val = 0

    if st.session_state['y_N2']:
        try:
            y_N2_val = float(st.session_state['y_N2'])
            if not (0 <= y_N2_val <= 1):
                st.error("N2 content must be between 0 and 1")
                y_N2_val = 0
        except ValueError:
            st.error("Please enter a valid number for N2 content")
            y_N2_val = 0

    # Check if the sum of the mole fractions exceeds 1
    total_mole_frac = y_H2S_val + y_CO2_val + y_N2_val
    if total_mole_frac > 1:
        st.error("The sum of H2S, CO2, and N2 contents must not exceed 1. Currently, it is {:.2f}".format(total_mole_frac))
    else:
        st.write("The sum of H2S, CO2, and N2 contents is {:.2f}".format(total_mole_frac))

    # Ensure all required inputs are provided before performing calculations
    if SG_valid and P_valid and T_valid and total_mole_frac <= 1:
        # Sutton's correlation for pseudo-critical properties
        Tpc = 169.2 + 349.5 * SG - 74.0 * SG**2
        Ppc = 756.8 - 131.0 * SG - 3.6 * SG**2

        # Wichert-Aziz correction for non-hydrocarbon gases
        e = 120 * ((y_N2_val + y_CO2_val)**0.9 - (y_N2_val + y_CO2_val)**1.6) + 15 * (y_H2S_val**0.5 - y_H2S_val**4)
        Tpc_corr = Tpc - e
        Ppc_corr = Ppc * Tpc_corr / (Tpc + y_H2S_val * (1 - y_H2S_val) * (304.2 - Tpc_corr))

        # Store Ppc_corr, Tpc_corr, Pi, and Zi in session state
        st.session_state['Ppc'] = Ppc_corr
        st.session_state['Tpc'] = Tpc_corr
        st.session_state['Pi'] = Pi
        st.session_state['T_rankine'] = T + 459.67
        st.session_state['Tr'] = st.session_state['T_rankine'] / Tpc_corr
        Zi = Zfac(st.session_state['Tr'], st.session_state['Pi'] / Ppc_corr)
        st.session_state['Zi'] = Zi
        

        #st.write(f"Pseudo-critical properties: Ppc_corr = {Ppc_corr}, Tpc_corr = {Tpc_corr}")
        #st.write(f"Initial Z-factor: Zi = {Zi:.4f}")

        # Inject custom CSS to style the button
        st.markdown("""
            <style>
            .stButton>button {
                background-color: #007BFF;
                color: white;
                border-radius: 5px;
                padding: 10px 25px;
                font-size: 16px;
                border: none;
                cursor: pointer;
            }
            .stButton>button:hover {
                background-color: #0056b3;
            }
            </style>
            """, unsafe_allow_html=True)

        # Calculate Z-factor and Gas Expansion Factor when the button is clicked and inputs are valid
        if st.button("Calculate Z-factor and Gas Expansion Factor", key="calculate_z"):
            # Convert temperature from Fahrenheit to Rankine
            T_rankine = T + 459.67

            # Reduced properties
            Pr = Pi / Ppc_corr
            Tr = T_rankine / Tpc_corr

            Zi = Zfac(Tr, Pr)
            Ei = gas_expansion_factor(Pi, Zi, T_rankine)
            st.session_state['Ei'] = Ei

            # Use st.markdown with HTML/CSS to frame the text
            st.markdown("""
                <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
                    <p><b>At reservoir conditions:</b></p>
                    <p>The gas compressibility factor (Z-factor) is: <b>{Zi:.3f}</b></p>
                    <p>The gas expansion factor (E) is: <b>{Ei:.3f}</b> (scf/rcf)</p>
                </div>
                """.format(Zi=Zi, Ei=Ei), unsafe_allow_html=True)
            
            # Generate pressure values from 14.7 to (input pressure + 1000) with increments of 200
            pressures = [14.7] + [i for i in range(200, int(Pi + 1000) + 1, 200)]

            # Calculate Z-factors and Gas Expansion Factors for each pressure value
            z_factors = [Zfac(T_rankine / Tpc_corr, p / Ppc_corr) for p in pressures]
            gas_expansion_factors = [gas_expansion_factor(p, z, T_rankine) for p, z in zip(pressures, z_factors)]
            gas_recovery_factors = [grf(Ei, E) for E in gas_expansion_factors]
            p_over_z = [p_z(p, z) for p, z in zip(pressures, z_factors)]

            # Create a DataFrame for the table
            df_props = pd.DataFrame({'Pressure (psi)': pressures, 'Z-factor': z_factors, 'Gas Expansion Factor (scf/rcf)': gas_expansion_factors, 'Gas Recovery Factor (%)': gas_recovery_factors, 'P/z (psi)': p_over_z})
            st.session_state['df_props'] = df_props  # Store in session state
            
            # Display the DataFrame as a table
            st.write(df_props)

            # Plot Z-factor vs. Pressure using Plotly
            fig = px.line(df_props, x='Pressure (psi)', y='Z-factor', title='Z-factor vs. Pressure')
            fig.add_scatter(
                x=[Pi],
                y=[Zi],
                mode='markers+text',
                marker=dict(size=12, color='red'),
                text=[f"<b>{Zi:.4f}</b>"],
                textposition='top center',
                name='Reservoir Pressure'
            )
            st.plotly_chart(fig)

            # Plot Gas Expansion Factor vs. Pressure using Plotly
            fig2 = px.line(df_props, x='Pressure (psi)', y='Gas Expansion Factor (scf/rcf)', title='Gas Expansion Factor (E) vs. Pressure')
            fig2.add_scatter(
                x=[Pi],
                y=[Ei],
                mode='markers+text',
                marker=dict(size=12, color='blue'),
                text=[f"<b>{Ei:.4f}</b>"],
                textposition='top center',
                name='Reservoir Pressure',
                #labels={'Gas Expansion Factor (E)': 'Gas Expansion Factor (scf/rcf)'}  # Rename the y-axis
            )
            st.plotly_chart(fig2)
            
            # Plot P vs. Recovery Factor
            fig3 = px.line(df_props, x='Gas Recovery Factor (%)', y='Pressure (psi)', title='GRF (Ei - E) / Ei vs. Pressure', range_y=[0, Pi])
            fig3.add_scatter(
                #labels={'Gas Recovery Factor (%)': 'Recovery Factor (%)'}  # Rename the x-axis
                #x=[Pi],
                #y=[Zi],
                #mode='markers+text',
                #marker=dict(size=12, color='red'),
                #text=[f"<b>{Zi:.4f}</b>"],
                #textposition='top center',
                #name='Input Pressure'
            )
            st.plotly_chart(fig3)


if option == "Gas Originally In-place (GIIP)":
    st.subheader("Gas Originally In-place (GIIP)")

    # Initialize GIIP session state if not already set
    if 'GIIP' not in st.session_state:
        st.session_state['GIIP'] = ''

    if 'option_giip' not in st.session_state:
        st.session_state['option_giip'] = "Enter GIIP"

    option = st.radio("Is the GIIP known?", ("Enter GIIP", "Calculate GIIP"), key='option_giip')

    if option == "Enter GIIP":
        #st.subheader("Enter Known Gas Initially In-place (GIIP)")
        st.session_state['GIIP'] = st.text_input("Enter GIIP in billions of cubic feet (Bcf)", st.session_state['GIIP'])
    else:
        st.subheader("Calculate Gas Initially In-place (GIIP)")
        area = st.number_input("Area of the reservoir (acres)", min_value=0.0, format="%.1f")
        thickness = st.number_input("Thickness of the gas-bearing formation (feet)", min_value=0.0, format="%.2f")
        porosity = st.number_input("Porosity of the reservoir (fraction)", min_value=0.0, max_value=1.0, format="%.2f")
        connate_sw = st.number_input("Connate water saturation (fraction)", min_value=0.0, max_value=1.0, format="%.2f")

        # Ensure Ei is calculated before using it
        if st.session_state['Ei'] is not None:
            if st.button("Calculate GIIP"):
                GIIP = calculate_giip(area, thickness, porosity, connate_sw, st.session_state['Ei'])
                st.session_state['GIIP'] = f"{GIIP:.2f} "
                st.success(f"Calculated GIIP: {st.session_state['GIIP']}")
        else:
            st.warning("Please calculate the Z-factor and Gas Expansion Factor first.")

if option == "Production Schedule":
    st.subheader("Production Schedule")

    # Create a single column for layout
    col1, col2 = st.columns([1, 3])
    
    with col2:
        # Adding input boxes in a single column
        st.subheader("Build-up Phase")
        st.session_state['BUY'] = st.slider("Build-up Period Years (t1)", min_value=0, max_value=10, step=1)
        
        st.subheader("Plateau Phase")
        st.session_state['PLP'] = st.slider("Export Pressure (psi)", min_value=0, max_value=10000, step=50)
        st.session_state['PLG'] = st.slider("Export Gas Rate (Qg) (MMscfd)", min_value=0, max_value=1000, step=50)
                
        st.subheader("Decline Phase")
        st.session_state['DDR'] = st.slider("Anticipated Decline Rate (fracton/year)", min_value=0.001, max_value=0.90, step=0.05)
        st.session_state['DAP'] = st.slider("Abandonment Pressure (psi)", min_value=14.7, max_value=2000.0, step=10.0)

    # Inject custom CSS to style the button
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #007BFF;
            color: white;
            border-radius: 5px;
            padding: 10px 25px;
            font-size: 16px;
            border: none;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        </style>
        """, unsafe_allow_html=True)

    if st.button("Generate Gas Production Profile", key="generate_profile"):
        required_keys = ['SG', 'T', 'y_H2S', 'y_CO2', 'y_N2', 'GIIP', 'Ppc', 'Tpc', 'T_rankine', 'Tr', 'Pi', 'Zi', 'PLG', 'BUY', 'PLP', 'DPD']
        missing_keys = [key for key in required_keys if key not in st.session_state]

        # Display missing keys if any
        if missing_keys:
            st.write(f"Missing keys: {missing_keys}")
            st.error("Please fill in all required gas properties and GIIP.")
        else:
            SG = float(st.session_state['SG'])
            T = float(st.session_state['T'])
            y_H2S_val = float(st.session_state['y_H2S'])
            y_CO2_val = float(st.session_state['y_CO2'])
            y_N2_val = float(st.session_state['y_N2'])
            GIIP = float(st.session_state['GIIP'])
            Ppc_corr = st.session_state['Ppc']
            Tpc_corr = st.session_state['Tpc']
            T_rankine = st.session_state['T_rankine']
            Tr = st.session_state['Tr']
            Pi = st.session_state['Pi']
            Zi = st.session_state['Zi']
            PLG = float(st.session_state['PLG'])
            BUY = int(st.session_state['BUY'])
            PLP = float(st.session_state['PLP'])
            DPD = float(st.session_state['DPD'])
            DDR = float(st.session_state['DDR'])
            DAP = float(st.session_state['DAP'])
            
            
            # Display all the values being used
            #st.write("### Values being used for calculation:")
            #st.write(f"SG: {SG}")
            #st.write(f"T: {T}")
            #st.write(f"y_H2S: {y_H2S_val}")
            #st.write(f"y_CO2: {y_CO2_val}")
            #st.write(f"y_N2: {y_N2_val}")
            #st.write(f"GIIP: {GIIP}")
            #st.write(f"Ppc_corr: {Ppc_corr}")
            #st.write(f"Tpc_corr: {Tpc_corr}")
            #st.write(f"T_rankine: {T_rankine}")
            #st.write(f"Tr: {Tr}")
            #st.write(f"Pi: {Pi}")
            #st.write(f"Zi: {Zi}")
            #st.write(f"PLG: {PLG}")
            #st.write(f"BUY: {BUY}")
            #st.write(f"PLP: {PLP}")
            #st.write(f"DPD: {DPD}")

            # Reduced properties for PLP
            Pr_PL = PLP / Ppc_corr

            # Calculate Z factor at PLP
            Z_PL = Zfac(Tr, Pr_PL)
            st.session_state['Z_PL'] = Z_PL
            #st.write(f"Z factor at Plateau Export Pressure (PLP): {Z_PL:.4f}")
            
            # Calculate Gp2
            #Gp2 = GIIP * (1 - ((PLP / Z_PL) / (Pi / Zi)))
            #st.session_state['Gp2'] = Gp2
            #st.write(f"Gp2: {Gp2:.2f}")

            # Calculate production during the build-up period
            bug_production, Gp1 = calculate_bug(PLG, BUY)

            # Display the results
            #st.write("### Production during the Build-up Period (BUG):")
            #for year, production in enumerate(bug_production, start=1):
            #    st.write(f"Year {year}: Production = {production:.2f} MMscfd")
            #st.write(f"Build-up Gp (Gp1): {Gp1:.2f} Bcf")
            
            # Save to session state
            st.session_state['Gp1'] = Gp1
            st.session_state['bug_production'] = bug_production
            
            # Generate the DataFrame
            years = list(range(1, len(bug_production) + 1))
            bu_rates = bug_production
            bug_gp = pd.Series([rate * 365.2 / 1000 for rate in bu_rates]).cumsum()
            

            # Create a DataFrame
            df_bu = pd.DataFrame({
                'Year': years,
                'Gas Rate (MMscfd)': bu_rates,
                'Gp (Bcf)': bug_gp
            })
            
            # Adding Recovery Factor column
            df_bu = grf_sch(df_bu, 'Gp (Bcf)', GIIP, "Gas Recovery Factor (%)")
            st.session_state.df_bu = df_bu
            
            # Perform the interpolation
#            df_bu = cubic_spline_interpolation(df_props, df_bu, 'Pressure (psi)', 'Gas Recovery Factor (%)', 'Gas Recovery Factor (%)')
#            
            # Perform the interpolation
#            if 'df_props' in st.session_state:
#                df_props = st.session_state.df_props
#                df_bu = cubic_spline_interpolation(df_props, df_bu, 'Pressure (psi)', 'Gas Recovery Factor (%)', 'Gas Recovery Factor (%)')
#                st.write("DataFrame after interpolation:")
#                st.dataframe(df_bu)
#            else:
#                st.write("df_props not found in session state.")             
            
            # Check if df_props exists in session state
            if 'df_props' in st.session_state:
                df_props = st.session_state.df_props
                # Use df_props within functions like cubic_spline_interpolation
                df_bu = cubic_spline_interpolation(df_props, df_bu, 'Gas Recovery Factor (%)', 'Pressure (psi)', 'Gas Recovery Factor (%)')
                #st.write("Build-up Phase:")
                #st.dataframe(df_bu)
                
                #fig4 = px.line(df_props, x='Gas Recovery Factor (%)', y='Pressure (psi)', title='Gas Recovery Factor vs. Pressure')
                #st.plotly_chart(fig4)
                
            else:
                st.error("Gas properties data is not available. Please complete the Gas Properties section first.")
            
            df_pl = pl_gas_rate(df_bu, df_props, PLG, GIIP, PLP)
            st.session_state.df_pl = df_pl
            #st.write("Plateau Phase")
            #st.dataframe(df_pl)
            
            
            df_decline = exp_decline(df_pl, df_props, DDR, DAP, GIIP, Pi)
            
            df_decline = add_initial_row(df_decline)
            
            st.session_state.df_decline = df_decline
            st.write("Gas Production Profile")
            st.dataframe(df_decline)
            
            plot_gas_rate_and_pressure(df_decline, PLG, Pi)

            
            
            
            
            







































