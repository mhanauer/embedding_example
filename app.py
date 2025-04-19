import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import string
import random

# Set page config
st.set_page_config(page_title="ICD-10 Embedding Demo", layout="wide")

# App title and introduction
st.title("ICD-10 Code Embedding for Healthcare Cost Prediction")

# Introduction with detailed explanation
st.markdown("""
## Direct ICD-10 Code Embeddings with Controlled Dimensionality

### The Challenge of ICD-10 Codes in Analytics

In healthcare analytics, ICD-10 codes present a significant challenge due to their high dimensionality. With thousands of possible codes, traditional approaches like one-hot encoding create an impractically large number of sparse features that are difficult for models to process effectively.

### The Direct Embedding Solution

This application demonstrates how to create and maintain direct one-to-one mappings between ICD-10 codes and their vector embeddings:

1. Each ICD-10 code is first represented by a high-dimensional vector using a pre-trained language model
2. We then use PCA to reduce these vectors to exactly 10 dimensions while preserving their semantic relationships
3. The one-to-one mapping between codes and embeddings is maintained throughout
4. These fixed-size vectors can be used as features in predictive models
5. We can always translate back from any vector to its corresponding ICD-10 code

This approach allows us to:
- Dramatically reduce dimensionality while preserving information
- Capture semantic relationships between similar conditions
- Maintain the ability to interpret which specific codes influence predictions
""")

# Sample data generator
def generate_sample_data(num_members=500):
    # Create member IDs
    member_ids = [f"MEM{i:05d}" for i in range(1, num_members + 1)]
    
    # Common ICD-10 codes with descriptions
    icd10_codes = {
        "E11.9": "Type 2 diabetes mellitus without complications",
        "E11.51": "Type 2 diabetes mellitus with diabetic peripheral angiopathy",
        "E10.9": "Type 1 diabetes mellitus without complications",
        "I10": "Essential primary hypertension",
        "I11.9": "Hypertensive heart disease without heart failure",
        "J45.909": "Unspecified asthma uncomplicated",
        "J45.20": "Mild intermittent asthma uncomplicated",
        "M54.5": "Low back pain",
        "M54.16": "Radiculopathy lumbar region",
        "F41.1": "Generalized anxiety disorder",
        "F32.9": "Major depressive disorder single episode unspecified",
        "K21.9": "Gastro-esophageal reflux disease without esophagitis",
        "K58.9": "Irritable bowel syndrome without diarrhea",
        "G89.29": "Other chronic pain",
        "G43.909": "Migraine unspecified not intractable without status migrainosus"
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
        if "E11.9" in codes or "E11.51" in codes or "E10.9" in codes:  # Diabetes adds cost
            base_amount += 300
        if "I10" in codes or "I11.9" in codes:  # Hypertension adds cost
            base_amount += 200
        if "M54.5" in codes or "M54.16" in codes:  # Back pain adds cost
            base_amount += 150
        if "G89.29" in codes:  # Chronic pain adds cost
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

# Function to create embeddings using a pre-trained model and reduce dimensions with PCA
def create_pca_embeddings(descriptions, n_components=10):
    """
    Create embeddings for ICD-10 codes using a pre-trained model
    and reduce dimensions using PCA
    """
    try:
        # Try to load the model - if this fails, we'll show an error message
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate the embeddings for each code description
        codes = []
        original_embeddings = []
        
        for code, description in descriptions.items():
            # Generate embedding for this description
            embedding = model.encode(description)
            codes.append(code)
            original_embeddings.append(embedding)
        
        # Convert to numpy array
        original_embeddings = np.array(original_embeddings)
        
        # Reduce dimensions with PCA
        reducer = PCA(n_components=n_components)
        reduced_embeddings = reducer.fit_transform(original_embeddings)
        
        # Create dictionary mapping codes to reduced embeddings
        embeddings_dict = {codes[i]: reduced_embeddings[i] for i in range(len(codes))}
        
        # Create lookup dataframe
        embedding_lookup = pd.DataFrame({
            'ICD10_Code': codes,
            'Description': [descriptions[code] for code in codes]
        })
        
        # Add embedding columns
        for i in range(n_components):
            embedding_lookup[f'dim_{i}'] = [reduced_embeddings[i][j] for i in range(len(codes)) for j in range(n_components) if j == i][0:len(codes)]
        
        return embeddings_dict, reducer, embedding_lookup, True
        
    except Exception as e:
        # If there's an error, return None and the error message
        return None, None, None, False

# Function to compute cosine similarity
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    return dot_product / (norm_v1 * norm_v2)

# Function to find the exact ICD-10 code for a given embedding
def get_icd10_code(embedding_vector, embedding_lookup):
    """Find the exact matching ICD-10 code for a given embedding vector"""
    # Convert embedding columns to numpy array for comparison
    dim_cols = [col for col in embedding_lookup.columns if col.startswith('dim_')]
    saved_embeddings = embedding_lookup[dim_cols].values
    
    # Find exact match (or closest match if using approximate values)
    distances = np.linalg.norm(saved_embeddings - embedding_vector, axis=1)
    closest_idx = np.argmin(distances)
    
    # Return the corresponding ICD-10 code
    return embedding_lookup.iloc[closest_idx]['ICD10_Code']

# Function to prepare data for modeling
def prepare_modeling_data(df, embedding_dict, embedding_dim=10):
    # Group by member_id to get all codes for each member
    member_groups = df.groupby('member_id')
    
    # Prepare the feature matrix
    X_data = []
    y_data = []
    member_ids = []
    member_codes = []
    
    for member_id, group in member_groups:
        # Get demographic features (same for all rows of this member)
        age = group['age'].iloc[0]
        gender = group['gender'].iloc[0]
        
        # Get all ICD-10 codes for this member
        codes = group['icd10_code'].tolist()
        member_codes.append(codes)
        
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
        member_ids.append(member_id)
    
    return np.array(X_data), np.array(y_data), member_ids, member_codes

# Calculate feature impact 
def calculate_feature_impact(model, X, feature_names):
    # Initialize impact
    impact = np.zeros(X.shape[1])
    
    # Get baseline prediction
    baseline_pred = model.predict(X).mean()
    
    # For each feature, calculate its impact
    for i in range(X.shape[1]):
        # Make a copy of the data
        X_permuted = X.copy()
        
        # Permute the feature
        X_permuted[:, i] = np.random.permutation(X[:, i])
        
        # Get new predictions
        new_preds = model.predict(X_permuted)
        
        # Calculate impact as the difference in predictions
        impact[i] = baseline_pred - new_preds.mean()
    
    # Return as a dict
    return {feature_names[i]: impact[i] for i in range(len(feature_names))}

# Function to identify important ICD-10 codes based on feature importance
def identify_important_codes(model, feature_names, embeddings_dict, embedding_lookup, icd_descriptions, top_n=10):
    """Identify the most important ICD-10 codes based on feature importance"""
    # Get feature importances from the model
    importances = model.feature_importances_
    
    # Find indices of embedding dimensions
    embedding_indices = [i for i, name in enumerate(feature_names) if name.startswith('emb_dim_')]
    
    # Calculate the importance of each code based on its embedding and feature importance
    code_importances = {}
    
    for code, embedding in embeddings_dict.items():
        # Compute the weighted importance of this code
        importance = 0
        
        # For each embedding dimension
        for i, idx in enumerate(embedding_indices):
            # Add the contribution of this dimension (importance * embedding value)
            importance += importances[idx] * abs(embedding[i])
        
        # Store the code importance
        code_importances[code] = importance
    
    # Sort codes by importance
    sorted_codes = sorted(code_importances.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top codes
    top_codes = sorted_codes[:top_n]
    
    # Create a dataframe for display
    important_codes = []
    
    for code, importance in top_codes:
        important_codes.append({
            'ICD-10 Code': code,
            'Description': icd_descriptions.get(code, 'Unknown'),
            'Importance Score': importance,
            'Embedding Vector': str(embeddings_dict[code].round(3).tolist())
        })
    
    return pd.DataFrame(important_codes)

# Main workflow
st.header("Step 1: Generate Sample Data")

# Explanation for data generation
st.markdown("""
### Generating Healthcare Data with ICD-10 Codes

In real healthcare analytics, data would come from claims or EHR systems. Here, we're generating synthetic data that represents:

- **Members with multiple diagnoses**: Each member can have 1-3 ICD-10 diagnosis codes
- **Demographic information**: Age and gender, which affect healthcare costs
- **Allowed amounts**: The total healthcare cost for each member, affected by their diagnoses

This simulates a payer dataset where we need to predict costs based on members' health conditions.
""")

num_members = st.slider("Number of members:", 100, 1000, 500)

if st.button("Generate Sample Data and Run Full Pipeline"):
    # Step 1: Generate Data
    with st.spinner("Generating sample data..."):
        df, icd_descriptions = generate_sample_data(num_members)
        st.success(f"Generated sample data with {len(df)} records for {num_members} members.")
        
        # Display sample data
        st.subheader("Sample Data:")
        st.dataframe(df.head())
        
        # Display ICD-10 codes used
        st.subheader("ICD-10 Codes and Descriptions:")
        codes_df = pd.DataFrame({
            "Code": list(icd_descriptions.keys()),
            "Description": list(icd_descriptions.values())
        })
        st.dataframe(codes_df)
    
    # Step 2: Create Embeddings
    st.header("Step 2: Create Direct ICD-10 Code Embeddings with PCA")
    
    # Explanation for embedding creation
    st.markdown("""
    ### Creating Direct One-to-One ICD-10 Embeddings with PCA
    
    In this step, we:
    
    1. Use a pre-trained language model to create initial embeddings for each ICD-10 code description
    2. Apply PCA to reduce these embeddings to exactly 10 dimensions
    3. Maintain a direct one-to-one mapping between each code and its embedding
    
    This approach ensures that:
    - Each ICD-10 code has exactly one corresponding vector
    - Similar medical conditions have similar vectors
    - Every vector has exactly 10 dimensions
    """)
    
    with st.spinner("Creating embeddings from ICD-10 descriptions..."):
        # Create embeddings using pre-trained model and PCA
        embedding_dim = 10
        embeddings_dict, reducer, embedding_lookup, success = create_pca_embeddings(icd_descriptions, n_components=embedding_dim)
        
        if not success:
            st.error("Failed to create embeddings. Make sure the sentence-transformers package is installed.")
            st.stop()
        
        # Display embedding information
        st.success(f"Created {embedding_dim}-dimensional embeddings for {len(embeddings_dict)} ICD-10 codes.")
        
        # Show the exact embeddings for each code
        st.subheader("Direct ICD-10 Code to Embedding Mapping:")
        st.dataframe(embedding_lookup)
        
        # Display sample cosine similarities
        st.subheader("Sample Embedding Similarities:")
        
        # Let's check if diabetes codes are similar
        if "E11.9" in embeddings_dict and "E10.9" in embeddings_dict:
            diabetes_sim = cosine_similarity(embeddings_dict["E11.9"], embeddings_dict["E10.9"])
            st.write(f"Similarity between 'E11.9' (Type 2 diabetes) and 'E10.9' (Type 1 diabetes): {diabetes_sim:.4f}")
        
        # Check if asthma codes are similar
        if "J45.909" in embeddings_dict and "J45.20" in embeddings_dict:
            asthma_sim = cosine_similarity(embeddings_dict["J45.909"], embeddings_dict["J45.20"])
            st.write(f"Similarity between 'J45.909' (Unspecified asthma) and 'J45.20' (Mild asthma): {asthma_sim:.4f}")
        
        # Check if unrelated codes are less similar
        if "E11.9" in embeddings_dict and "J45.909" in embeddings_dict:
            unrelated_sim = cosine_similarity(embeddings_dict["E11.9"], embeddings_dict["J45.909"])
            st.write(f"Similarity between 'E11.9' (Type 2 diabetes) and 'J45.909' (Asthma): {unrelated_sim:.4f}")
        
        # Demonstrate the ability to find the most similar code to a given embedding
        st.subheader("Embedding to ICD-10 Code Reverse Lookup:")
        
        # Pick a random code to demonstrate
        sample_code = random.choice(list(embeddings_dict.keys()))
        sample_embedding = embeddings_dict[sample_code]
        
        st.write(f"Starting with the embedding for '{sample_code}' ({icd_descriptions.get(sample_code, '')}):")
        st.write(f"Embedding vector: {sample_embedding.round(3).tolist()}")
        
        # Find the most similar code using our lookup function
        most_similar_code = get_icd10_code(sample_embedding, embedding_lookup)
        
        st.write(f"Most similar code: '{most_similar_code}' ({icd_descriptions.get(most_similar_code, '')})")
        
        # Explain the significance
        st.markdown("""
        This demonstrates that we can always map back and forth between embeddings and ICD-10 codes:
        - From code to embedding: A direct lookup in our embeddings dictionary
        - From embedding to code: Finding the most similar code in embedding space
        
        This bidirectional mapping is essential for maintaining interpretability throughout the modeling process.
        """)
    
    # Step 3: Prepare Data for XGBoost
    st.header("Step 3: Prepare Data for Modeling")
    
    # Explanation for data preparation
    st.markdown("""
    ### Preparing Data with Embeddings
    
    Now we transform our raw data with embeddings into a format suitable for machine learning:
    
    1. **Patient-level aggregation**: Moving from diagnosis-level to patient-level data
    2. **Multiple diagnosis handling**: Combining embeddings for patients with multiple conditions
    3. **Feature matrix creation**: Creating a consistent set of features for each patient
    
    The key is maintaining the direct relationship between embeddings and ICD-10 codes throughout this process.
    """)
    
    with st.spinner("Preparing data for XGBoost model..."):
        # Prepare features and target
        X, y, member_ids, member_codes = prepare_modeling_data(df, embeddings_dict, embedding_dim=embedding_dim)
        
        # Feature names
        feature_names = ['age', 'gender'] + [f'emb_dim_{i+1}' for i in range(embedding_dim)]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.success(f"Prepared feature matrix with shape {X.shape} and target vector with shape {y.shape}")
        
        # Display feature matrix example
        st.subheader("Example Feature Matrix:")
        example_df = pd.DataFrame(X[:5], columns=feature_names)
        st.dataframe(example_df)
    
    # Step 4: Train XGBoost Model
    st.header("Step 4: Train XGBoost Model for Cost Prediction")
    
    # Explanation for model training
    st.markdown("""
    ### Training a Predictive Model with Embeddings
    
    We now train an XGBoost regression model to predict healthcare costs using:
    - Demographic features (age, gender)
    - The 10-dimensional embeddings that represent each member's diagnoses
    
    XGBoost can identify which embedding dimensions most strongly influence healthcare costs.
    """)
    
    with st.spinner("Training XGBoost model..."):
        # Train model
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predictions
        predictions = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        st.success("Model training complete!")
        
        # Display metrics
        st.subheader("Model Performance:")
        metrics_df = pd.DataFrame({
            "Metric": ["Mean Absolute Error", "Root Mean Squared Error", "R² Score"],
            "Value": [f"${mae:.2f}", f"${rmse:.2f}", f"{r2:.4f}"]
        })
        st.dataframe(metrics_df)
    
    # Step 5: Analyze Feature Importance and ICD-10 Code Impact
    st.header("Step 5: Identify Most Important ICD-10 Codes")
    
    # Explanation for ICD-10 code analysis
    st.markdown("""
    ### Translating Model Insights to Specific ICD-10 Codes
    
    Now we can identify exactly which ICD-10 codes most strongly influence healthcare costs:
    
    1. We calculate the importance of each embedding dimension from the XGBoost model
    2. For each ICD-10 code, we compute its overall importance based on its specific embedding values
    3. This gives us a direct ranking of which codes are most predictive of costs
    
    Because we maintained the direct mapping between codes and embeddings, we can precisely identify which specific medical conditions drive costs.
    """)
    
    with st.spinner("Identifying most important ICD-10 codes..."):
        # Display feature importance
        st.subheader("Embedding Dimension Importance:")
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        st.dataframe(importance_df)
        
        # Identify the most important ICD-10 codes
        st.subheader("Most Important ICD-10 Codes for Cost Prediction:")
        
        important_codes_df = identify_important_codes(model, feature_names, embeddings_dict, embedding_lookup, icd_descriptions)
        
        st.dataframe(important_codes_df)
        
        # Explanation of results
        st.markdown("""
        ### Interpreting the Results
        
        The table above shows which specific ICD-10 codes have the greatest impact on healthcare costs, ranked by importance. This analysis:
        
        1. Precisely identifies which medical conditions are most predictive of costs
        2. Maintains direct traceability from model insights to specific diagnoses
        3. Provides the exact embedding vector for each influential code
        
        These insights enable targeted interventions focused on the specific conditions that drive healthcare expenditures.
        """)
        
        st.success("Analysis complete! We've maintained a direct one-to-one mapping between ICD-10 codes and their embeddings throughout the process.")
