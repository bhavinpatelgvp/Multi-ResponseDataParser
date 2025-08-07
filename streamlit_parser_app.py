import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re

# Configure the Streamlit page
st.set_page_config(
    page_title="Multi-Response Data Parser",
    page_icon="📊",
    layout="wide"
)

def load_data(uploaded_file):
    """Load data from uploaded CSV or Excel file."""
    try:
        if uploaded_file.name.endswith('.csv'):
            # Try different encodings for CSV files
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
                except UnicodeDecodeError:
                    df = pd.read_csv(uploaded_file, encoding='cp1252')
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def detect_separator(sample_values, potential_separators=[',', ';', '|', '/', '\n', '\t']):
    """Detect the most likely separator used in multi-response data."""
    separator_counts = {}
    
    for sep in potential_separators:
        count = 0
        for value in sample_values:
            if pd.notna(value) and isinstance(value, str):
                count += value.count(sep)
        separator_counts[sep] = count
    
    # Return the separator with the highest count, or comma as default
    if max(separator_counts.values()) > 0:
        return max(separator_counts, key=separator_counts.get)
    return ','

def normalize_responses(df, response_column, separator, preserve_columns=None):
    """
    Normalize multi-response data by splitting responses into individual rows.
    
    Args:
        df: DataFrame containing the data
        response_column: Column name containing multi-response data
        separator: Character used to separate multiple responses
        preserve_columns: List of columns to preserve in the normalized data
    """
    if preserve_columns is None:
        preserve_columns = [col for col in df.columns if col != response_column]
    
    normalized_rows = []
    
    for idx, row in df.iterrows():
        responses = str(row[response_column]) if pd.notna(row[response_column]) else ''
        
        if responses and responses.lower() != 'nan':
            # Split the responses and clean them
            individual_responses = [resp.strip() for resp in responses.split(separator) if resp.strip()]
            
            if individual_responses:
                for response in individual_responses:
                    new_row = {}
                    # Copy preserved columns
                    for col in preserve_columns:
                        new_row[col] = row[col]
                    # Add the individual response
                    new_row[response_column] = response
                    normalized_rows.append(new_row)
            else:
                # Handle empty responses
                new_row = {}
                for col in preserve_columns:
                    new_row[col] = row[col]
                new_row[response_column] = ''
                normalized_rows.append(new_row)
        else:
            # Handle NaN or empty responses
            new_row = {}
            for col in preserve_columns:
                new_row[col] = row[col]
            new_row[response_column] = ''
            normalized_rows.append(new_row)
    
    return pd.DataFrame(normalized_rows)

def create_download_link(df, filename="normalized_data.csv"):
    """Create a download link for the DataFrame."""
    csv = df.to_csv(index=False)
    b64 = pd.io.common.urlsafe_base64_encode(csv.encode())
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
    return href

# Main app
def main():
    st.title("📊 Multi-Response Data Parser")
    st.markdown("Upload a CSV or Excel file to normalize multi-response data into individual rows.")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file containing multi-response data (e.g., comma-separated tags)"
    )
    
    if uploaded_file is not None:
        # Load the data
        with st.spinner("Loading data..."):
            df = load_data(uploaded_file)
        
        if df is not None:
            st.success(f"File loaded successfully! Shape: {df.shape}")
            
            # Display original data preview
            st.subheader("📋 Original Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column selection
            st.subheader("🎯 Select Multi-Response Column")
            response_column = st.selectbox(
                "Choose the column containing multiple responses:",
                options=df.columns.tolist(),
                help="Select the column that contains comma-separated or multi-response data"
            )
            
            if response_column:
                # Show sample values from selected column
                sample_values = df[response_column].dropna().head(10).tolist()
                st.write("**Sample values from selected column:**")
                for i, val in enumerate(sample_values[:5]):
                    st.write(f"{i+1}. {val}")
                
                # Separator detection and selection
                st.subheader("🔧 Separator Configuration")
                detected_sep = detect_separator(sample_values)
                
                col1, col2 = st.columns(2)
                with col1:
                    separator_option = st.radio(
                        "Choose separator:",
                        options=["Auto-detect", "Custom"],
                        help="Auto-detect will try to find the best separator"
                    )
                
                with col2:
                    if separator_option == "Auto-detect":
                        separator = detected_sep
                        st.info(f"Detected separator: '{separator}'")
                    else:
                        separator = st.text_input(
                            "Enter custom separator:",
                            value=",",
                            max_chars=3,
                            help="Enter the character that separates multiple responses"
                        )
                
                # Column preservation options
                st.subheader("📝 Column Selection")
                preserve_columns = st.multiselect(
                    "Select columns to preserve in normalized data:",
                    options=[col for col in df.columns if col != response_column],
                    default=[col for col in df.columns if col != response_column],
                    help="Choose which columns to keep alongside the normalized responses"
                )
                
                # Process button
                if st.button("🚀 Normalize Data", type="primary"):
                    with st.spinner("Processing data..."):
                        try:
                            # Normalize the data
                            normalized_df = normalize_responses(
                                df, 
                                response_column, 
                                separator, 
                                preserve_columns
                            )
                            
                            # Store in session state
                            st.session_state['normalized_df'] = normalized_df
                            st.session_state['original_shape'] = df.shape
                            st.session_state['normalized_shape'] = normalized_df.shape
                            
                        except Exception as e:
                            st.error(f"Error processing data: {str(e)}")
    
    # Display results if available
    if 'normalized_df' in st.session_state:
        st.subheader("✅ Normalized Data Results")
        
        # Show transformation summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Rows", st.session_state['original_shape'][0])
        with col2:
            st.metric("Normalized Rows", st.session_state['normalized_shape'][0])
        with col3:
            expansion_ratio = st.session_state['normalized_shape'][0] / st.session_state['original_shape'][0]
            st.metric("Expansion Ratio", f"{expansion_ratio:.2f}x")
        
        # Display normalized data
        st.dataframe(st.session_state['normalized_df'], use_container_width=True)
        
        # Download options
        st.subheader("💾 Download Options")
        
        col1, col2 = st.columns(2)
        with col1:
            filename = st.text_input(
                "Filename for download:",
                value="normalized_data.csv",
                help="Enter the desired filename for the CSV download"
            )
        
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            
            # Create download button
            csv_data = st.session_state['normalized_df'].to_csv(index=False)
            st.download_button(
                label="📥 Download Normalized CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                type="secondary"
            )
        
        # Data quality summary
        with st.expander("📊 Data Quality Summary"):
            st.write("**Summary Statistics:**")
            st.write(f"- Total unique responses: {st.session_state['normalized_df'][response_column].nunique()}")
            st.write(f"- Empty responses: {st.session_state['normalized_df'][response_column].eq('').sum()}")
            st.write(f"- Most common response: {st.session_state['normalized_df'][response_column].mode().iloc[0] if not st.session_state['normalized_df'][response_column].empty else 'N/A'}")
            
            # Show response frequency
            if not st.session_state['normalized_df'].empty:
                st.write("**Top 10 Most Frequent Responses:**")
                response_counts = st.session_state['normalized_df'][response_column].value_counts().head(10)
                st.dataframe(response_counts.reset_index(), use_container_width=True)

    # Instructions and examples
    with st.expander("ℹ️ How to Use This App"):
        st.markdown("""
        ### Instructions:
        1. **Upload your file**: Choose a CSV or Excel file containing multi-response data
        2. **Select column**: Choose the column that contains multiple responses (e.g., "red, blue, green")
        3. **Configure separator**: The app will auto-detect the separator, or you can specify a custom one
        4. **Choose columns**: Select which other columns to preserve in the normalized output
        5. **Normalize**: Click the "Normalize Data" button to process your data
        6. **Download**: Download the cleaned, normalized data as a CSV file
        
        ### Example:
        **Original data:**
        | ID | Name | Colors |
        |----|------|---------|
        | 1  | John | red, blue, green |
        | 2  | Jane | yellow, red |
        
        **Normalized data:**
        | ID | Name | Colors |
        |----|------|---------|
        | 1  | John | red |
        | 1  | John | blue |
        | 1  | John | green |
        | 2  | Jane | yellow |
        | 2  | Jane | red |
        
        ### Supported Separators:
        - Comma (,)
        - Semicolon (;)
        - Pipe (|)
        - Forward slash (/)
        - Tab (\\t)
        - New line (\\n)
        """)

if __name__ == "__main__":
    main()