import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.decomposition import PCA
import time

# Set page config
st.set_page_config(page_title="ICD-10 Embedding Demo", layout="wide")

# App title and description
st.title("ICD-10 Code Embedding for Healthcare Cost Prediction")
st.markdown("""
This application demonstrates how embedding ICD-10 diagnosis codes can improve healthcare cost prediction models.
Instead of treating each code as a separate variable, we convert them into meaningful numerical vectors that capture their semantic relationships.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Preparation", "Create Embeddings", "Dimensionality Reduction", "XGBoost Prediction", "SHAP Explanation"])

# Initialize session state variables if they don't exist
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'icd_descriptions' not in st.session_state:
    st.session_state.icd_descriptions = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'reduced_embeddings' not in st.session_state:
    st.session_state.reduced_embeddings = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None

# Sample data generator
def generate_sample_data(num_members=1000):
    # Create member IDs
    member_ids = [f"MEM{i:05d}" for i in range(1, num_members + 1)]
    
    # Common ICD-10 codes with descriptions
    icd10_codes = {
        "E11.9": "Type 2 diabetes mellitus without complications",
        "I10": "Essential (primary) hypertension",
        "J45.909": "Unspecified asthma, uncomplicated",
        "M54.5": "Low back pain",
        "F41.1": "Generalized anxiety disorder",
        "K21.9": "Gastro-esophageal reflux disease without esophagitis",
        "G89.29": "Other chronic pain",
        "E78.5": "Hyperlipidemia, unspecified",
        "N39.0": "Urinary tract infection, site not specified",
        "M17.9": "Osteoarthritis of knee, unspecified",
        "H40.9": "Unspecified glaucoma",
        "E03.9": "Hypothyroidism, unspecified",
        "J30.1": "Allergic rhinitis due to pollen",
        "K57.30": "Diverticulosis of large intestine without perforation or abscess without bleeding",
        "D64.9": "Anemia, unspecified"
    }
    
    # Create a dataset
    data = []
    for member_id in member_ids:
        # Random age between 18 and 90
        age = np.random.randint(18, 91)
        
        # Gender (0 for male, 1 for female)
        gender = np.random.choice([0, 1])
        
        # Sample 1-3 ICD-10 codes for each member
        num_codes = np.random.randint(1, 4)
        codes = np.random.choice(list(icd10_codes.keys()), size=num_codes, replace=False)
        
        # Base allowed amount with some randomness based on conditions
        base_amount = 500
        
        # Certain conditions cost more
        if "E11.9" in codes:  # Diabetes adds cost
            base_amount += 300
        if "I10" in codes:    # Hypertension adds cost
            base_amount += 200
        if "M17.9" in codes:  # Knee osteoarthritis adds cost
            base_amount += 400
        if "G89.29" in codes: # Chronic pain adds cost
            base_amount += 250
        
        # Age factor
        age_factor = 1.0 + (age - 18) / 72 * 0.5  # Older members cost more
        
        # Final allowed amount with some random noise
        allowed_amount = base_amount * age_factor * (0.9 + 0.2 * np.random.random())
        
        # Add to dataset
        for code in codes:
            data.append({
                "member_id": member_id,
                "age": age,
                "gender": gender,
                "icd10_code": code,
                "allowed_amount": allowed_amount
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create a dictionary for ICD-10 code descriptions
    icd_descriptions = icd10_codes
    
    return df, icd_descriptions

# Function to load pretrained model
@st.cache_resource
def load_embedding_model():
    # Load a sentence transformer model for embeddings
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except:
        st.error("Failed to load the embedding model. Please install sentence-transformers package.")
        return None

# Function to create embeddings
def create_embeddings(descriptions, model):
    # Create embeddings from the ICD-10 code descriptions
    texts = list(descriptions.values())
    embeddings = model.encode(texts)
    
    # Create a dictionary mapping ICD-10 codes to their embeddings
    embedding_dict = {code: embeddings[i] for i, code in enumerate(descriptions.keys())}
    return embedding_dict

# Function to reduce dimensionality
def reduce_dimensions(embeddings_dict, method='pca', n_components=10):
    # Extract the embeddings and codes
    codes = list(embeddings_dict.keys())
    emb_array = np.array([embeddings_dict[code] for code in codes])
    
    # Perform dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    else:  # UMAP
        reducer = UMAP(n_components=n_components, random_state=42)
    
    reduced_embeddings = reducer.fit_transform(emb_array)
    
    # Create a new dictionary mapping codes to reduced embeddings
    reduced_dict = {codes[i]: reduced_embeddings[i] for i in range(len(codes))}
    
    return reduced_dict, reducer

# Function to prepare data for modeling
def prepare_modeling_data(df, embedding_dict, embedding_dim=10):
    # Group by member_id to get all codes for each member
    member_groups = df.groupby('member_id')
    
    # Prepare the feature matrix
    X_data = []
    y_data = []
    
    for member_id, group in member_groups:
        # Get demographic features (same for all rows of this member)
        age = group['age'].iloc[0]
        gender = group['gender'].iloc[0]
        
        # Get all ICD-10 codes for this member
        codes = group['icd10_code'].tolist()
        
        # Average the embeddings for all codes
        if codes:
            code_embeddings = [embedding_dict.get(code, np.zeros(embedding_dim)) for code in codes]
            avg_embedding = np.mean(code_embeddings, axis=0)
        else:
            avg_embedding = np.zeros(embedding_dim)
        
        # Combine demographics and embeddings
        features = np.concatenate([[age, gender], avg_embedding])
        
        # Target is the allowed amount
        target = group['allowed_amount'].iloc[0]
        
        X_data.append(features)
        y_data.append(target)
    
    return np.array(X_data), np.array(y_data)

# Function to train XGBoost model
def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return predictions, mae, rmse, r2

# Function to calculate SHAP values
def calculate_shap(model, X_test):
    # Create a SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    return shap_values

# Data Preparation Page
if page == "Data Preparation":
    st.header("Step 1: Data Preparation")
    
    # Option to upload data or use sample data
    data_option = st.radio(
        "Choose data source:",
        ("Generate sample data", "Upload CSV file")
    )
    
    if data_option == "Generate sample data":
        num_members = st.slider("Number of members:", 100, 2000, 500)
        
        if st.button("Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                df, icd_descriptions = generate_sample_data(num_members)
                st.session_state.raw_data = df
                st.session_state.icd_descriptions = icd_descriptions
                st.success(f"Generated sample data with {len(df)} records for {num_members} members.")
    
    else:
        uploaded_file = st.file_uploader("Upload your CSV file:", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.raw_data = df
            
            # Upload ICD-10 descriptions
            st.write("Please upload a CSV file with ICD-10 code descriptions (columns: code, description):")
            desc_file = st.file_uploader("Upload ICD-10 descriptions:", type=["csv"])
            
            if desc_file is not None:
                desc_df = pd.read_csv(desc_file)
                icd_descriptions = dict(zip(desc_df['code'], desc_df['description']))
                st.session_state.icd_descriptions = icd_descriptions
    
    # Display the data if available
    if st.session_state.raw_data is not None:
        st.write("Preview of the data:")
        st.dataframe(st.session_state.raw_data.head())
        
        st.write("Summary statistics:")
        st.dataframe(st.session_state.raw_data.describe())
        
        st.write("ICD-10 code distribution:")
        code_counts = st.session_state.raw_data['icd10_code'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=code_counts.index, y=code_counts.values, ax=ax)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

# Create Embeddings Page
elif page == "Create Embeddings":
    st.header("Step 2: Create ICD-10 Code Embeddings")
    
    if st.session_state.icd_descriptions is None:
        st.warning("Please prepare your data first.")
    else:
        st.write("We'll use a pretrained language model to create embeddings for each ICD-10 code description.")
        
        # Display ICD codes and descriptions
        st.write("ICD-10 Codes and Descriptions:")
        desc_df = pd.DataFrame({
            'Code': list(st.session_state.icd_descriptions.keys()),
            'Description': list(st.session_state.icd_descriptions.values())
        })
        st.dataframe(desc_df)
        
        # Load the embedding model
        if st.button("Create Embeddings"):
            with st.spinner("Loading embedding model and creating embeddings..."):
                model = load_embedding_model()
                if model:
                    embeddings = create_embeddings(st.session_state.icd_descriptions, model)
                    st.session_state.embeddings = embeddings
                    
                    # Show embedding dimensions
                    first_code = list(embeddings.keys())[0]
                    first_embedding = embeddings[first_code]
                    st.success(f"Created embeddings for {len(embeddings)} ICD-10 codes. Each embedding has {len(first_embedding)} dimensions.")
                    
                    # Show a sample embedding
                    st.write(f"Sample embedding for code '{first_code}':")
                    st.write(first_embedding[:10])  # Show first 10 dimensions
                    
                    # Visualize embeddings in 2D for a quick check
                    st.write("2D visualization of embeddings (using PCA):")
                    temp_reduced, _ = reduce_dimensions(embeddings, method='pca', n_components=2)
                    
                    # Create a dataframe for plotting
                    plot_df = pd.DataFrame({
                        'Code': list(temp_reduced.keys()),
                        'Dim1': [emb[0] for emb in temp_reduced.values()],
                        'Dim2': [emb[1] for emb in temp_reduced.values()]
                    })
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.scatterplot(data=plot_df, x='Dim1', y='Dim2', ax=ax)
                    
                    # Add labels to points
                    for _, row in plot_df.iterrows():
                        ax.text(row['Dim1'], row['Dim2'], row['Code'], fontsize=9)
                    
                    plt.title('2D PCA Visualization of ICD-10 Code Embeddings')
                    st.pyplot(fig)

# Dimensionality Reduction Page
elif page == "Dimensionality Reduction":
    st.header("Step 3: Reduce Embedding Dimensions")
    
    if st.session_state.embeddings is None:
        st.warning("Please create embeddings first.")
    else:
        st.write("Now we'll reduce the high-dimensional embeddings to a more manageable size.")
        
        # Options for dimensionality reduction
        reduction_method = st.selectbox("Reduction Method:", ["PCA", "UMAP"])
        n_components = st.slider("Number of dimensions:", 2, 20, 10)
        
        if st.button("Reduce Dimensions"):
            with st.spinner(f"Reducing embeddings to {n_components} dimensions using {reduction_method}..."):
                method = 'pca' if reduction_method == 'PCA' else 'umap'
                reduced_embeddings, reducer = reduce_dimensions(st.session_state.embeddings, method=method, n_components=n_components)
                st.session_state.reduced_embeddings = reduced_embeddings
                
                st.success(f"Reduced embeddings to {n_components} dimensions.")
                
                # Show a sample reduced embedding
                first_code = list(reduced_embeddings.keys())[0]
                st.write(f"Sample reduced embedding for code '{first_code}':")
                st.write(reduced_embeddings[first_code])
                
                # Visualize reduced embeddings in 2D
                if n_components > 2:
                    st.write("2D visualization of reduced embeddings (first two dimensions):")
                    
                    # Create a dataframe for plotting
                    plot_df = pd.DataFrame({
                        'Code': list(reduced_embeddings.keys()),
                        'Dim1': [emb[0] for emb in reduced_embeddings.values()],
                        'Dim2': [emb[1] for emb in reduced_embeddings.values()]
                    })
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.scatterplot(data=plot_df, x='Dim1', y='Dim2', ax=ax)
                    
                    # Add labels to points
                    for _, row in plot_df.iterrows():
                        ax.text(row['Dim1'], row['Dim2'], row['Code'], fontsize=9)
                    
                    plt.title(f'First 2 Dimensions of {n_components}-D Reduced Embeddings')
                    st.pyplot(fig)

# XGBoost Prediction Page
elif page == "XGBoost Prediction":
    st.header("Step 4: Predict Allowed Amount with XGBoost")
    
    if st.session_state.reduced_embeddings is None:
        st.warning("Please reduce embedding dimensions first.")
    elif st.session_state.raw_data is None:
        st.warning("Please prepare your data first.")
    else:
        st.write("We'll use XGBoost to predict allowed amounts using the reduced embeddings.")
        
        # Prepare data for modeling
        if st.button("Train XGBoost Model"):
            with st.spinner("Preparing data and training model..."):
                # Get embedding dimension
                first_code = list(st.session_state.reduced_embeddings.keys())[0]
                embedding_dim = len(st.session_state.reduced_embeddings[first_code])
                
                # Prepare features and target
                X, y = prepare_modeling_data(
                    st.session_state.raw_data, 
                    st.session_state.reduced_embeddings,
                    embedding_dim
                )
                
                # Feature names (for SHAP later)
                feature_names = ['age', 'gender'] + [f'emb_dim_{i+1}' for i in range(embedding_dim)]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                model = train_xgboost(X_train, y_train)
                st.session_state.model = model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.feature_names = feature_names
                
                # Evaluate model
                predictions, mae, rmse, r2 = evaluate_model(model, X_test, y_test)
                
                st.success("Model training complete!")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Absolute Error", f"${mae:.2f}")
                col2.metric("Root Mean Squared Error", f"${rmse:.2f}")
                col3.metric("R² Score", f"{r2:.4f}")
                
                # Plot actual vs predicted
                st.write("Actual vs Predicted Allowed Amounts:")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_test, predictions, alpha=0.5)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_xlabel('Actual Amount ($)')
                ax.set_ylabel('Predicted Amount ($)')
                ax.set_title('Actual vs Predicted Allowed Amounts')
                st.pyplot(fig)
                
                # Feature importance plot
                st.write("XGBoost Feature Importance:")
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                ax.set_title('XGBoost Feature Importance')
                st.pyplot(fig)

# SHAP Explanation Page
elif page == "SHAP Explanation":
    st.header("Step 5: Explain Predictions with SHAP")
    
    if st.session_state.model is None:
        st.warning("Please train an XGBoost model first.")
    else:
        st.write("We'll use SHAP (SHapley Additive exPlanations) to explain the predictions.")
        
        if st.button("Calculate SHAP Values"):
            with st.spinner("Calculating SHAP values..."):
                # Calculate SHAP values
                shap_values = calculate_shap(st.session_state.model, st.session_state.X_test)
                st.session_state.shap_values = shap_values
                
                st.success("SHAP values calculated!")
                
                # Summary plot
                st.write("SHAP Summary Plot:")
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, st.session_state.X_test, 
                                  feature_names=st.session_state.feature_names,
                                  show=False)
                st.pyplot(fig)
                
                # Bar plot
                st.write("SHAP Mean Absolute Impact:")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, st.session_state.X_test,
                                  feature_names=st.session_state.feature_names,
                                  plot_type="bar", show=False)
                st.pyplot(fig)
                
                # Sample explanation for a specific prediction
                st.write("Sample Individual Prediction Explanation:")
                sample_idx = 0
                
                # Show prediction details
                actual = st.session_state.y_test[sample_idx]
                predicted = st.session_state.model.predict(st.session_state.X_test[sample_idx].reshape(1, -1))[0]
                
                st.write(f"Sample Member:")
                st.write(f"- Actual Allowed Amount: ${actual:.2f}")
                st.write(f"- Predicted Allowed Amount: ${predicted:.2f}")
                
                # Waterfall plot
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.waterfall_plot(shap_values[sample_idx], show=False)
                st.pyplot(fig)
                
                # Force plot (convert to HTML)
                st.write("SHAP Force Plot:")
                force_plot = shap.plots.force(shap_values[sample_idx], matplotlib=True)
                st.pyplot(force_plot)
                
                # Analysis of the embedding dimensions
                st.write("Analysis of Embedding Dimensions:")
                emb_dims = [f for f in st.session_state.feature_names if f.startswith('emb_dim_')]
                emb_indices = [i for i, f in enumerate(st.session_state.feature_names) if f.startswith('emb_dim_')]
                
                if emb_indices:
                    emb_importance = pd.DataFrame({
                        'Dimension': emb_dims,
                        'SHAP Impact': [np.abs(shap_values.values[:, i]).mean() for i in emb_indices]
                    }).sort_values(by='SHAP Impact', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='SHAP Impact', y='Dimension', data=emb_importance, ax=ax)
                    ax.set_title('Impact of Embedding Dimensions')
                    st.pyplot(fig)
                    
                    st.write("This shows which dimensions of our ICD-10 embeddings are most influential in cost prediction.")

st.sidebar.markdown("---")
st.sidebar.info("""
This application demonstrates how to:
1. Create embeddings from ICD-10 code descriptions
2. Reduce dimensions to a manageable size
3. Use the embeddings to predict healthcare costs
4. Explain predictions with SHAP values

The embeddings capture semantic relationships between diagnoses, allowing the model to learn that similar conditions have similar effects on costs.
""")
