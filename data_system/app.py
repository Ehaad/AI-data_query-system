import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# Helper function to process uploaded files
def process_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
        return data
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Helper function to query the model
def query_model(prompt):
    try:
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        outputs = model.generate(**inputs, max_length=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        st.error(f"Error querying the model: {e}")
        return "An error occurred"

# Streamlit UI
st.title("AI-Powered Data Interaction App")

# File Upload Section
st.subheader("Upload a File")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

# If file is uploaded, process it
if uploaded_file:
    data = process_file(uploaded_file)
    
    if data is not None:
        st.write("File uploaded successfully. Here's a preview:")
        st.write(data.head())

        # Prompt Section
        st.subheader("Ask a Question About Your Data")
        prompt = st.text_input("Enter your prompt here")

        # Handle Prompt Submission
        if st.button("Submit Prompt"):
            if prompt:
                with st.spinner("Processing..."):
                    response = query_model(prompt)
                    st.write("Response from LLM:")
                    st.write(response)
            else:
                st.error("Please enter a valid prompt.")
