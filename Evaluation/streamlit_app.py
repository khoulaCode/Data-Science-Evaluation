import warnings
warnings.filterwarnings("ignore")

import streamlit as st


# Import the contents of the first app (lr3.py)
import lr3  # Assuming lr3.py has a main() function

# Import the contents of the second app (ts1.py)
import ts1  # Assuming ts1.py has a main() function

# Main application title
st.title("Evaluation Application: Linear Regression & Time Series")

# Create tabs for each of the applications
tab1, tab2 = st.tabs(["Linear Regression", "Time Series"])

# Linear Regression Tab
with tab1:
    st.header("Linear Regression Analysis")
    lr3.main()  # Call the main function from lr3.py

# Time Series Tab
with tab2:
    st.header("Time Series Analysis")
    ts1.main()  # Call the main function from ts1.py

# Run the combined app with the following command in your terminal:
# streamlit run combined_app.py
