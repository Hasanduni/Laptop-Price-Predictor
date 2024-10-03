import streamlit as st
import pickle
import numpy as np
import os

# Check if files exist and load them
if not os.path.exists('rd_model.pkl') or not os.path.exists('lapdf.pkl') or not os.path.exists('lapbeforeencodedf.pkl'):
    st.error("One or more required files are missing. Please check the file paths.")
else:
    try:
        # Load the model and data
        pipe = pickle.load(open('rd_model.pkl', 'rb'))
        df_encoded = pickle.load(open('lapdf.pkl', 'rb'))
        df = pickle.load(open('lapbeforeencodedf.pkl', 'rb'))
    except Exception as e:
        st.error(f"Error loading files: {e}")

# Streamlit app title
st.title("Laptop Price Predictor")

# Ensure required columns are in the data
required_columns = ['Company', 'TypeName', 'OpSys', 'Gpu Brand', 'CPU_Brand']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"The following columns are missing in the data: {', '.join(missing_columns)}")
else:
    # Collect user inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        company_input = st.selectbox('Company', df['Company'].unique())
        typename_input = st.selectbox('Type Name', df['TypeName'].unique())
        inches = st.number_input('Inches', min_value=10.0, max_value=25.0, value=15.0, step=0.1)
        ram = st.number_input('RAM (in GB)', min_value=2, max_value=64, value=8, step=1)
        opsys_input = st.selectbox('Operating System', df['OpSys'].unique())

    with col2:
        weight = st.number_input('Weight (in kg)', min_value=0.5, max_value=5.0, value=1.5, step=0.1)
        gpu_input = st.selectbox('GPU', df['Gpu Brand'].unique())
        screen_width = st.number_input('Screen Width (in pixels)', min_value=800, max_value=4000, value=1920, step=1)
        screen_height = st.number_input('Screen Height (in pixels)', min_value=600, max_value=3000, value=1080, step=1)
        cpu_input = st.selectbox('CPU Brand', df['CPU_Brand'].unique())

    with col3:
        cpu_freq = st.number_input('CPU Frequency (in GHz)', min_value=1.0, max_value=5.0, value=2.5, step=0.1)
        ssd = st.number_input('SSD (in GB)', min_value=0, max_value=2048, value=256, step=128)
        flash = st.number_input('Flash (in GB)', min_value=0, max_value=512, value=0, step=128)
        hdd = st.number_input('HDD (in GB)', min_value=0, max_value=2048, value=0, step=128)
        hybrid = st.number_input('Hybrid (in GB)', min_value=0, max_value=2048, value=0, step=128)

    # Function to map non-encoded input to encoded value
    def get_encoded_value(column_name, input_value):
        try:
            return df_encoded[df_encoded[column_name] == input_value][column_name].values[0]
        except IndexError:
            st.error(f"Encoded value for {input_value} not found in column {column_name}")
            return None

    # Debugging: print selected inputs
    st.write(f"Selected company: {company_input}")
    st.write(f"Selected typename: {typename_input}")
    st.write(f"Selected OS: {opsys_input}")
    st.write(f"Selected GPU: {gpu_input}")
    st.write(f"Selected CPU: {cpu_input}")

    # Predict button
    if st.button('Predict Price'):
        with st.spinner('Predicting...'):
            try:
                # Get encoded values
                company = get_encoded_value('Company', company_input)
                typename = get_encoded_value('TypeName', typename_input)
                opsys = get_encoded_value('OpSys', opsys_input)
                gpu = get_encoded_value('Gpu Brand', gpu_input)
                cpu_brand = get_encoded_value('CPU_Brand', cpu_input)

                # Ensure all encoded values are retrieved before proceeding
                if None in [company, typename, opsys, gpu, cpu_brand]:
                    st.error("Error with one or more selected values. Cannot proceed with prediction.")
                else:
                    # Prepare query for prediction
                    query = np.array([company, typename, inches, ram, opsys, weight, gpu,
                                      screen_width, screen_height, cpu_brand, cpu_freq, ssd, flash, hdd, hybrid])
                    query = query.reshape(1, -1)

                    # Predict price
                    predicted_price = pipe.predict(query)[0]
                    st.title(f"The predicted price is ${predicted_price:.2f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
