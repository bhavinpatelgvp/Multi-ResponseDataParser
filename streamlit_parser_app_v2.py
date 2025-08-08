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

def detect_separator(sample_values, potential_separators=[',', ';', '|', '/', '\n', '\t', ' - ', ' / ']):
    """Detect the most likely separator used in multi-response data."""
    separator_counts = {}
    total_non_empty = 0
    
    # Count occurrences of each separator
    for sep in potential_separators:
        count = 0
        rows_with_sep = 0
        for value in sample_values:
            if pd.notna(value) and isinstance(value, str) and value.strip():
                total_non_empty += 1
                sep_count = value.count(sep)
                if sep_count > 0:
                    count += sep_count
                    rows_with_sep += 1
        
        # Weight by both frequency and prevalence across rows
        separator_counts[sep] = {
            'total_count': count, 
            'rows_with_sep': rows_with_sep,
            'score': count * (rows_with_sep / max(len(sample_values), 1))
        }
    
    # Return the separator with the highest weighted score
    if separator_counts and max(sep['score'] for sep in separator_counts.values()) > 0:
        best_sep = max(separator_counts, key=lambda x: separator_counts[x]['score'])
        return best_sep
    return ','

def normalize_responses(df, response_column, separator, preserve_columns=None):
    """
    Normalize multi-response data by splitting responses into individual rows using pandas explode.
    
    Args:
        df: DataFrame containing the data
        response_column: Column name containing multi-response data
        separator: Character used to separate multiple responses
        preserve_columns: List of columns to preserve in the normalized data
    """
    if preserve_columns is None:
        preserve_columns = [col for col in df.columns if col != response_column]
    
    # Create a copy of the dataframe with only required columns
    columns_to_keep = preserve_columns + [response_column]
    df_copy = df[columns_to_keep].copy()
    
    # Convert the multi-response column to lists
    def split_responses(value):
        if pd.isna(value) or value == '' or str(value).lower() == 'nan':
            return [np.nan]  # Keep as single NaN for explode to work properly
        
        # Convert to string and split
        responses = str(value).split(separator)
        # Clean and filter empty responses
        cleaned_responses = [resp.strip() for resp in responses if resp.strip()]
        
        # Return original if no valid responses found after cleaning
        return cleaned_responses if cleaned_responses else [np.nan]
    
    # Apply the splitting function
    df_copy[response_column] = df_copy[response_column].apply(split_responses)
    
    # Use pandas explode method to normalize the data
    normalized_df = df_copy.explode(response_column, ignore_index=True)
    
    # Clean up the response column - remove any remaining NaN values if desired
    # (Comment out the next line if you want to keep NaN values)
    # normalized_df = normalized_df.dropna(subset=[response_column])
    
    # Replace NaN values with empty strings for better display
    normalized_df[response_column] = normalized_df[response_column].fillna('')
    
    return normalized_df

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
                            # Show some debugging info
                            st.write("**Processing Details:**")
                            sample_for_debug = df[response_column].dropna().head(3)
                            st.write(f"- Selected column: `{response_column}`")
                            st.write(f"- Using separator: `'{separator}'`")
                            st.write(f"- Sample values to be split:")
                            for i, val in enumerate(sample_for_debug):
                                split_preview = str(val).split(separator)
                                st.write(f"  {i+1}. `{val}` → `{split_preview}`")
                            
                            # Normalize the data
                            normalized_df = normalize_responses(
                                df, 
                                response_column, 
                                separator, 
                                preserve_columns
                            )
                            
                            # Validate results
                            if normalized_df.empty:
                                st.error("Normalization resulted in empty dataset. Please check your separator and data.")
                                return
                            
                            if len(normalized_df) == len(df):
                                st.warning("No expansion occurred. This might indicate the separator is not working correctly.")
                            
                            # Store in session state
                            st.session_state['normalized_df'] = normalized_df
                            st.session_state['original_shape'] = df.shape
                            st.session_state['normalized_shape'] = normalized_df.shape
                            st.session_state['response_column'] = response_column
                            
                            st.success("✅ Data normalized successfully!")
                            
                        except Exception as e:
                            st.error(f"Error processing data: {str(e)}")
                            st.write("**Debug Information:**")
                            st.write(f"- Error type: {type(e).__name__}")
                            st.write(f"- DataFrame shape: {df.shape}")
                            st.write(f"- Response column data type: {df[response_column].dtype}")
                            st.write(f"- Sample non-null values: {df[response_column].dropna().head(3).tolist()}")
    
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
            response_col = st.session_state.get('response_column', 'Response')
            normalized_data = st.session_state['normalized_df']
            
            st.write("**Summary Statistics:**")
            st.write(f"- Total unique responses: {normalized_data[response_col].nunique()}")
            st.write(f"- Empty responses: {normalized_data[response_col].eq('').sum()}")
            
            # Handle mode calculation safely
            non_empty_responses = normalized_data[response_col][normalized_data[response_col] != '']
            if not non_empty_responses.empty:
                most_common = non_empty_responses.mode()
                if not most_common.empty:
                    st.write(f"- Most common response: `{most_common.iloc[0]}`")
                else:
                    st.write("- Most common response: No clear mode found")
            else:
                st.write("- Most common response: All responses are empty")
            
            # Show response frequency
            if not normalized_data.empty and response_col in normalized_data.columns:
                st.write("**Top 10 Most Frequent Responses:**")
                response_counts = normalized_data[response_col].value_counts().head(10)
                if not response_counts.empty:
                    freq_df = pd.DataFrame({
                        'Response': response_counts.index,
                        'Count': response_counts.values
                    })
                    st.dataframe(freq_df, use_container_width=True)
                else:
                    st.write("No response data to display")

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