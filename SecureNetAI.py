# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip

import streamlit as st
import numpy as np

# Define Streamlit app
st.title("Binary Classification Prediction for Network Intrusion")

# Create user input fields for each feature
st.subheader("Input the following network connection features:")

features = {
'Duration': st.number_input(
    'Duration (seconds)', min_value=0.0, step=1.0, 
    help='Duration of the connection in seconds'
),
'Protocol Type': st.selectbox(
    'Protocol Type', ['tcp', 'udp', 'icmp'], 
    help='Type of protocol used in the connection'
),
'Service': st.text_input(
    'Service (e.g., http, smtp, ftp-data)', 
    help='Service requested by the connection, e.g., HTTP or FTP'
),
'Flag': st.selectbox(
    'Flag', ['SF', 'S0', 'REJ', 'RSTO', 'RSTR', 'SH'], 
    help='State of the connection (e.g., successful, reset)'
),
'Source Bytes': st.number_input(
    'Source Bytes', min_value=0, step=1, 
    help='Number of data bytes sent from source to destination'
),
'Destination Bytes': st.number_input(
    'Destination Bytes', min_value=0, step=1, 
    help='Number of data bytes sent from destination to source'
),
'Land': st.selectbox(
    'Land (1 if equal, 0 otherwise)', [0, 1], 
    help='Indicates if source and destination addresses are the same'
),
'Wrong Fragment': st.number_input(
    'Wrong Fragment', min_value=0, step=1, 
    help='Number of wrong fragments in the connection'
),
'Urgent Packets': st.number_input(
    'Urgent Packets', min_value=0, step=1, 
    help='Number of urgent packets in the connection'
),
'Hot Indicators': st.number_input(
    'Hot Indicators', min_value=0, step=1, 
    help='Number of hot indicators (e.g., failed login attempts)'
),
'Number of Failed Logins': st.number_input(
    'Number of Failed Logins', min_value=0, step=1, 
    help='Number of failed login attempts during the connection'
),
'Logged In': st.selectbox(
    'Logged In (1 if yes, 0 otherwise)', [0, 1], 
    help='Indicates if the user logged in successfully'
),
'Number of Compromised Conditions': st.number_input(
    'Number of Compromised Conditions', min_value=0, step=1, 
    help='Number of compromised conditions during the connection'
),
'Root Shell': st.selectbox(
    'Root Shell (1 if obtained, 0 otherwise)', [0, 1], 
    help='Indicates if a root shell was obtained'
),
'SU Attempted': st.selectbox(
    'SU Attempted (1 if yes, 0 otherwise)', [0, 1], 
    help='Indicates if a superuser (SU) command was attempted'
),
'Number of Root Operations': st.number_input(
    'Number of Root Operations', min_value=0, step=1, 
    help='Number of root accesses or operations performed'
),
'Number of File Creations': st.number_input(
    'Number of File Creations', min_value=0, step=1, 
    help='Number of file creation operations during the connection'
),
'Number of Shells': st.number_input(
    'Number of Shells', min_value=0, step=1, 
    help='Number of shell prompts invoked'
),
'Number of Access Files': st.number_input(
    'Number of Access Files', min_value=0, step=1, 
    help='Number of times access to sensitive files was attempted'
),
'Number of Outbound Commands': st.number_input(
    'Number of Outbound Commands', min_value=0, step=1, 
    help='Number of outbound commands in an FTP session'
),
'Is Host Login': st.selectbox(
    'Is Host Login (1 if yes, 0 otherwise)', [0, 1], 
    help='Indicates if the login is for the host (1) or not (0)'
),
'Is Guest Login': st.selectbox(
    'Is Guest Login (1 if yes, 0 otherwise)', [0, 1], 
    help='Indicates if the login was performed as a guest'
),
'Count': st.number_input(
    'Count (Number of connections to the same host)', min_value=0, step=1, 
    help='Number of connections made to the same host in a given period'
),
'Srv Count': st.number_input(
    'Srv Count (Number of connections to the same service)', min_value=0, step=1, 
    help='Number of connections made to the same service in a given period'
),
'S Error Rate': st.number_input(
    'S Error Rate', min_value=0.0, step=0.01, 
    help='Percentage of connections that had SYN errors'
),
'Srv S Error Rate': st.number_input(
    'Srv S Error Rate', min_value=0.0, step=0.01, 
    help='Percentage of connections to the same service with SYN errors'
),
'R Error Rate': st.number_input(
    'R Error Rate', min_value=0.0, step=0.01, 
    help='Percentage of connections that had REJ errors'
),
'Srv R Error Rate': st.number_input(
    'Srv R Error Rate', min_value=0.0, step=0.01, 
    help='Percentage of connections to the same service with REJ errors'
),
'Same Srv Rate': st.number_input(
    'Same Srv Rate', min_value=0.0, step=0.01, 
    help='Percentage of connections to the same service'
),
'Diff Srv Rate': st.number_input(
    'Diff Srv Rate', min_value=0.0, step=0.01, 
    help='Percentage of connections to different services'
),
'Srv Diff Host Rate': st.number_input(
    'Srv Diff Host Rate', min_value=0.0, step=0.01, 
    help='Percentage of connections to different hosts'
),
'Dst Host Count': st.number_input(
    'Dst Host Count', min_value=0, step=1, 
    help='Number of connections made to the same destination host'
),
'Dst Host Srv Count': st.number_input(
    'Dst Host Srv Count', min_value=0, step=1, 
    help='Number of connections made to the same service at the destination host'
),
'Dst Host Same Srv Rate': st.number_input(
    'Dst Host Same Srv Rate', min_value=0.0, step=0.01, 
    help='Percentage of connections to the same service at the destination host'
),
'Dst Host Diff Srv Rate': st.number_input(
    'Dst Host Diff Srv Rate', min_value=0.0, step=0.01, 
    help='Percentage of connections to different services at the destination host'
),
'Dst Host Same Src Port Rate': st.number_input(
    'Dst Host Same Src Port Rate', min_value=0.0, step=0.01, 
    help='Percentage of connections from the same source port to the destination host'
),
'Dst Host Srv Diff Host Rate': st.number_input(
    'Dst Host Srv Diff Host Rate', min_value=0.0, step=0.01, 
    help='Percentage of connections to different hosts using the same service'
),
'Dst Host S Error Rate': st.number_input(
    'Dst Host S Error Rate', min_value=0.0, step=0.01, 
    help='Percentage of connections to the destination host with SYN errors'
),
'Dst Host Srv S Error Rate': st.number_input(
    'Dst Host Srv S Error Rate', min_value=0.0, step=0.01, 
    help='Percentage of connections to the same service at the destination host with SYN errors'
),
'Dst Host R Error Rate': st.number_input(
    'Dst Host R Error Rate', min_value=0.0, step=0.01, 
    help='Percentage of connections to the destination host with REJ errors'
),
'Dst Host Srv R Error Rate': st.number_input(
    'Dst Host Srv R Error Rate', min_value=0.0, step=0.01, 
    help='Percentage of connections to the same service at the destination host with REJ errors'
)
}

# Create a dictionary to map the input labels to required column names
column_map = {
    'Duration': 'duration',
    'Protocol Type': 'protocoltype',
    'Service': 'service',
    'Flag': 'flag',
    'Source Bytes': 'srcbytes',
    'Destination Bytes': 'dstbytes',
    'Land': 'land',
    'Wrong Fragment': 'wrongfragment',
    'Urgent Packets': 'urgent',
    'Hot Indicators': 'hot',
    'Number of Failed Logins': 'numfailedlogins',
    'Logged In': 'loggedin',
    'Number of Compromised Conditions': 'numcompromised',
    'Root Shell': 'rootshell',
    'SU Attempted': 'suattempted',
    'Number of Root Operations': 'numroot',
    'Number of File Creations': 'numfilecreations',
    'Number of Shells': 'numshells',
    'Number of Access Files': 'numaccessfiles',
    'Number of Outbound Commands': 'numoutboundcmds',
    'Is Host Login': 'ishostlogin',
    'Is Guest Login': 'isguestlogin',
    'Count': 'count',
    'Srv Count': 'srvcount',
    'S Error Rate': 'serrorrate',
    'Srv S Error Rate': 'srvserrorrate',
    'R Error Rate': 'rerrorrate',
    'Srv R Error Rate': 'srvrerrorrate',
    'Same Srv Rate': 'samesrvrate',
    'Diff Srv Rate': 'diffsrvrate',
    'Srv Diff Host Rate': 'srvdiffhostrate',
    'Dst Host Count': 'dsthostcount',
    'Dst Host Srv Count': 'dsthostsrvcount',
    'Dst Host Same Srv Rate': 'dsthostsamesrvrate',
    'Dst Host Diff Srv Rate': 'dsthostdiffsrvrate',
    'Dst Host Same Src Port Rate': 'dsthostsamesrcportrate',
    'Dst Host Srv Diff Host Rate': 'dsthostsrvdiffhostrate',
    'Dst Host S Error Rate': 'dsthostserrorrate',
    'Dst Host Srv S Error Rate': 'dsthostsrvserrorrate',
    'Dst Host R Error Rate': 'dsthostrerrorrate',
    'Dst Host Srv R Error Rate': 'dsthostsrvrerrorrate'
}

# Convert user input into a DataFrame
input_data = pd.DataFrame([features])

# Rename the columns
input_data.rename(columns=column_map, inplace=True)

# Display input data if needed
st.write("User Input:", input_data)

# Prediction button
if st.button("Predict Attack or Normal"):
    # Call the prediction function
    result = binary_classification_prediction(input_data)
    st.subheader(f"Prediction: {result}")

# Prediction button
if st.button("Predict Attack Category"):
    # Call the prediction function
    result = multi_classification_prediction(input_data)
    st.subheader(f"Attack Category: {result}")