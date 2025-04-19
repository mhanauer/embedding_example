import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.decomposition import PCA
import string
import random

# Set page config
st.set_page_config(page_title="ICD-10 Embedding Demo", layout="wide")

# App title and comprehensive introduction
st.title("ICD-10 Code Embedding for Healthcare Cost Prediction")

# Introduction with detailed explanation
st.markdown("""
## Introduction: Why Embeddings Matter for Healthcare Data

### The Challenge of ICD-10 Codes in Machine Learning

In healthcare analytics, ICD-10 diagnosis codes present a significant challenge. With over 70,000 possible codes, traditional one-hot encoding approaches create an impractically large number of sparse features. For example:

- One-hot encoding turns each ICD-10 code into a separate binary column
- This creates thousands of sparse columns where most values are zero
- Models struggle with this high dimensionality and sparsity
- Similar medical conditions (like Type 1 and Type 2 diabetes) are treated as completely unrelated entities

### The Embedding Solution

Embeddings solve this problem by representing ICD-10 codes as dense vectors in a continuous space where:

- Each code is represented by a small vector (e.g., 10 numbers)
- Similar medical conditions have similar vector representations
- The semantic relationships between conditions are preserved
- Thousands of possible codes are compressed into just a few numeric columns
- Machine learning models can generalize across related conditions

This application demonstrates how to create and use these embeddings to improve healthcare cost prediction models.
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

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # Tokenize
    return text.split()

# Create manually designed embeddings based on medical concepts
def create_manual_embeddings(descriptions, dim=30):
    # We'll create embeddings manually by mapping terms to specific dimensions
    
    # Map of medical terms to dimension indices
    term_to_dim = {
        "diabetes": 0,
        "type 1": 1,
        "type 2": 2,
        "mellitus": 3,
        "hypertension": 4,
        "heart": 5,
        "asthma": 6,
        "pain": 7,
        "back": 8,
        "anxiety": 9,
        "depression": 10,
        "gastro": 11,
        "reflux": 12,
        "chronic": 13,
        "esophageal": 14,
        "peripheral": 15,
        "angiopathy": 16,
        "hypertensive": 17,
        "radiculopathy": 18,
        "lumbar": 19,
        "disorder": 20,
        "without": 21,
        "with": 22,
        "uncomplicated": 23,
        "failure": 24,
        "complication": 25,
        "migraine": 26,
        "unspecified": 27,
        "intermittent": 28,
        "irritable": 29
    }
    
    # Create embeddings dictionary
    embeddings = {}
    
    # For each code and description
    for code, desc in descriptions.items():
        # Initialize embedding vector
        embedding = np.zeros(dim)
        
        # Process the description
        tokens = preprocess_text(desc)
        
        # For each token in the description
        for token in tokens:
            # If token is a known medical term, activate its dimension
            if token in term_to_dim:
                dim_idx = term_to_dim[token]
                embedding[dim_idx] = 1.0
        
        # Add some relatedness between similar conditions
        
        # Diabetes related - dims 0,1,2,3
        if "diabetes" in desc.lower():
            if "type 1" in desc.lower():
                embedding[0] = 1.0
                embedding[1] = 1.0
                embedding[3] = 1.0
            elif "type 2" in desc.lower():
                embedding[0] = 1.0
                embedding[2] = 1.0
                embedding[3] = 1.0
        
        # Hypertension related - dims 4,5,17
        if "hypertension" in desc.lower() or "hypertensive" in desc.lower():
            embedding[4] = 1.0
            if "heart" in desc.lower():
                embedding[5] = 1.0
                embedding[17] = 1.0
        
        # Asthma related - dim 6
        if "asthma" in desc.lower():
            embedding[6] = 1.0
            if "mild" in desc.lower() and "intermittent" in desc.lower():
                embedding[28] = 1.0
        
        # Pain related - dims 7,8,13,18,19
        if "pain" in desc.lower():
            embedding[7] = 1.0
            if "back" in desc.lower():
                embedding[8] = 1.0
            if "chronic" in desc.lower():
                embedding[13] = 1.0
            if "radiculopathy" in desc.lower():
                embedding[18] = 1.0
            if "lumbar" in desc.lower():
                embedding[19] = 1.0
        
        # Scale by ICD-10 code prefix
        prefix = code.split('.')[0]
        
        # E codes: endocrine disorders like diabetes
        if prefix.startswith('E'):
            embedding[0:5] *= 1.5
        
        # I codes: circulatory system disorders like hypertension
        if prefix.startswith('I'):
            embedding[4:7] *= 1.5
        
        # J codes: respiratory system disorders like asthma
        if prefix.startswith('J'):
            embedding[6:7] *= 1.5
        
        # M codes: musculoskeletal disorders like back pain
        if prefix.startswith('M'):
            embedding[7:10] *= 1.5
        
        # F codes: mental disorders like anxiety
        if prefix.startswith('F'):
            embedding[9:11] *= 1.5
        
        # G codes: nervous system disorders
        if prefix.startswith('G'):
            embedding[13:14] *= 1.5
            embedding[26:27] *= 1.5
        
        # K codes: digestive system disorders
        if prefix.startswith('K'):
            embedding[11:13] *= 1.5
            embedding[29:30] *= 1.5
        
        # Store the embedding
        embeddings[code] = embedding
    
    # For reference, create reverse mapping of dimensions to medical concepts
    dim_to_concept = {}
    for term, dim_idx in term_to_dim.items():
        dim_to_concept[dim_idx] = term
    
    return embeddings, dim_to_concept

# Function to compute cosine similarity
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    return dot_product / (norm_v1 * norm_v2)

# Function to reduce dimensionality
def reduce_dimensions(embeddings_dict, n_components=10):
    # Extract the embeddings and codes
    codes = list(embeddings_dict.keys())
    emb_array = np.array([embeddings_dict[code] for code in codes])
    
    # Perform dimensionality reduction
    reducer = PCA(n_components=n_components)
    reduced_embeddings = reducer.fit_transform(emb_array)
    
    # Create a new dictionary mapping codes to reduced embeddings
    reduced_dict = {codes[i]: reduced_embeddings[i] for i in range(len(codes))}
    
    return reduced_dict, reducer, codes

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

# Calculate feature impact (simplified SHAP alternative)
def calculate_feature_impact(model, X, feature_names, n_samples=100):
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

# NEW: Function to interpret the embedding dimensions in terms of ICD-10 codes
def interpret_embedding_dimensions(reducer, original_embeddings, codes, dim_to_concept, top_n=5):
    # Get the PCA components
    components = reducer.components_
    
    # Dictionary to store interpretations
    interpretations = {}
    
    # For each reduced dimension
    for dim_idx in range(components.shape[0]):
        # Get the weights of original dimensions for this reduced dimension
        weights = components[dim_idx]
        
        # Get the top original dimensions for this reduced dimension
        top_indices = np.argsort(np.abs(weights))[-top_n:]
        
        # Get the concepts associated with these original dimensions
        top_concepts = [dim_to_concept.get(idx, f"Unknown-{idx}") for idx in top_indices]
        top_weights = [weights[idx] for idx in top_indices]
        
        # Create interpretation for this dimension
        concept_str = ", ".join([f"{concept} ({weight:.3f})" for concept, weight in zip(top_concepts, top_weights)])
        interpretations[dim_idx] = concept_str
        
        # Also find the ICD-10 codes that have high values in this dimension
        code_values = []
        for code_idx, code in enumerate(codes):
            reduced_vec = reducer.transform([original_embeddings[code]])[0]
            code_values.append((code, reduced_vec[dim_idx]))
        
        # Sort by absolute value and get top codes
        code_values.sort(key=lambda x: abs(x[1]), reverse=True)
        top_codes = code_values[:top_n]
        
        # Add to interpretation
        code_str = ", ".join([f"{code} ({value:.3f})" for code, value in top_codes])
        interpretations[dim_idx] = f"{concept_str}\nTop ICD-10 codes: {code_str}"
    
    return interpretations

# NEW: Function to find most important ICD-10 codes based on feature importance
def identify_important_codes(model, feature_names, reduced_embeddings, codes, icd_descriptions, top_n=5):
    # Get feature importances from the model
    importances = model.feature_importances_
    
    # Find indices of embedding dimensions
    embedding_indices = [i for i, name in enumerate(feature_names) if name.startswith('emb_dim_')]
    
    # Get importances of embedding dimensions
    embedding_importances = [(i-2, importances[i]) for i in embedding_indices]  # -2 to account for age,gender
    
    # Sort by importance
    embedding_importances.sort(key=lambda x: x[1], reverse=True)
    
    # Get top important dimensions
    top_dimensions = embedding_importances[:top_n]
    
    # For each important dimension, find codes with highest values
    important_codes = []
    
    for dim_idx, importance in top_dimensions:
        # Calculate influence of each code
        code_influences = []
        
        for code in codes:
            # Get the value of this code in this dimension
            dim_value = reduced_embeddings[code][dim_idx]
            
            # Calculate influence as value * importance
            influence = abs(dim_value * importance)
            
            code_influences.append((code, influence, dim_value, icd_descriptions.get(code, '')))
        
        # Sort by influence
        code_influences.sort(key=lambda x: x[1], reverse=True)
        
        # Add top codes for this dimension
        dimension_name = f"emb_dim_{dim_idx+1}"
        for code, influence, dim_value, desc in code_influences[:3]:
            important_codes.append({
                'Dimension': dimension_name,
                'Dimension Importance': importance,
                'ICD-10 Code': code,
                'Description': desc,
                'Dimension Value': dim_value,
                'Overall Influence': influence
            })
    
    return pd.DataFrame(important_codes)

# Main workflow
st.header("Step 1: Generate Sample Data")

# Enhanced explanation for data generation
st.markdown("""
### Why This Step Matters

In real healthcare analytics, data would come from claims or EHR systems. Here, we're generating synthetic data that represents:

- **Members with multiple diagnoses**: Each member can have 1-3 ICD-10 diagnosis codes
- **Demographic information**: Age and gender, which affect healthcare costs
- **Allowed amounts**: The total healthcare cost for each member, affected by their diagnoses

This simulates a payer dataset where we need to predict costs based on members' health conditions. The relationship between diagnoses and costs is critical - certain conditions like diabetes or chronic pain significantly increase healthcare expenditures.

**Key Point**: In real-world healthcare data, we often have hundreds or thousands of unique ICD-10 codes. Using traditional one-hot encoding would create an impractically large feature matrix.
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
    st.header("Step 2: Create ICD-10 Code Embeddings")
    
    # Enhanced explanation for embedding creation
    st.markdown("""
    ### Why Embeddings Are Critical
    
    In this step, we transform each ICD-10 code from a categorical value into a meaningful numerical vector. This is the core innovation that allows us to:
    
    1. **Represent semantic relationships**: Similar medical conditions get similar vectors
    2. **Capture hierarchical structure**: The ICD-10 system has inherent hierarchy (e.g., all E11.x codes are Type 2 diabetes variants)
    3. **Enable numerical processing**: Convert text-based medical concepts into numbers that ML models can process
    4. **Reduce dimensionality**: Instead of thousands of sparse columns, we get a small number of dense features
    
    **Traditional Approach (One-Hot Encoding):**
    - E11.9 → [0,0,1,0,0,0,0,0,0,0,0,0,0,...thousands more zeros...]
    - E11.51 → [0,0,0,1,0,0,0,0,0,0,0,0,0,...thousands more zeros...]
    
    **Embedding Approach:**
    - E11.9 → [0.235, -0.412, 0.178, 0.051, -0.133, 0.273, -0.084, 0.318, -0.221, 0.192]
    - E11.51 → [0.267, -0.395, 0.149, 0.062, -0.155, 0.301, -0.079, 0.302, -0.209, 0.201]
    
    Notice how the vectors for related conditions (both Type 2 diabetes codes) are similar. This lets the model generalize across related conditions.
    """)
    
    with st.spinner("Creating embeddings from ICD-10 descriptions..."):
        # Create manual embeddings
        embeddings, dim_to_concept = create_manual_embeddings(icd_descriptions)
        
        # Display embedding information
        first_code = list(embeddings.keys())[0]
        first_embedding = embeddings[first_code]
        st.success(f"Created embeddings for {len(embeddings)} ICD-10 codes. Original dimension: {len(first_embedding)}")
        
        # Display sample cosine similarities
        st.subheader("Sample Embedding Similarities:")
        
        # Let's check if diabetes codes are similar
        if "E11.9" in embeddings and "E10.9" in embeddings:
            diabetes_sim = cosine_similarity(embeddings["E11.9"], embeddings["E10.9"])
            st.write(f"Similarity between 'E11.9' (Type 2 diabetes) and 'E10.9' (Type 1 diabetes): {diabetes_sim:.4f}")
        
        # Check if asthma codes are similar
        if "J45.909" in embeddings and "J45.20" in embeddings:
            asthma_sim = cosine_similarity(embeddings["J45.909"], embeddings["J45.20"])
            st.write(f"Similarity between 'J45.909' (Unspecified asthma) and 'J45.20' (Mild asthma): {asthma_sim:.4f}")
        
        # Check if unrelated codes are less similar
        if "E11.9" in embeddings and "J45.909" in embeddings:
            unrelated_sim = cosine_similarity(embeddings["E11.9"], embeddings["J45.909"])
            st.write(f"Similarity between 'E11.9' (Type 2 diabetes) and 'J45.909' (Asthma): {unrelated_sim:.4f}")
    
        # Enhanced explanation of similarity results
        st.markdown("""
        ### Understanding the Similarities
        
        These similarity scores demonstrate a key benefit of embeddings:
        
        - **High similarity between related conditions**: Type 1 and Type 2 diabetes have similar vectors despite being different codes
        - **High similarity between variants of the same condition**: Different asthma types have similar vectors
        - **Low similarity between unrelated conditions**: Diabetes and asthma have very different vectors
        
        This pattern recognition is impossible with one-hot encoding, where every code is equally different from every other code.
        
        In production systems, we would use more sophisticated embedding approaches like Med2Vec, GRAM, or transformer-based models trained on medical literature. These create even more nuanced embeddings that capture complex medical relationships.
        """)
    
    # Step 3: Reduce Dimensions
    st.header("Step 3: Reduce Embedding Dimensions")
    
    # Enhanced explanation for dimensionality reduction
    st.markdown("""
    ### Why Dimension Reduction Matters
    
    Our initial embeddings have 30 dimensions, but we can compress them further while preserving their important relationships. This step:
    
    1. **Improves computational efficiency**: Fewer dimensions mean faster model training and inference
    2. **Reduces overfitting risk**: Fewer parameters help the model generalize better
    3. **Removes noise**: Lower dimensions focus on the most important patterns in the data
    4. **Makes visualization possible**: Lower dimensions are easier to visualize and understand
    5. **Standardizes embedding size**: Ensures all ICD-10 codes are represented by exactly 10 values
    
    We use Principal Component Analysis (PCA) to find the dimensions that capture the most variance in our embeddings. This is like finding the "essence" of the medical conditions while discarding redundant information.
    """)
    
    with st.spinner("Reducing embedding dimensions to 10..."):
        # Reduce to 10 dimensions
        reduced_embeddings, reducer, all_codes = reduce_dimensions(embeddings, n_components=10)
        
        # Display reduced embedding information
        st.success(f"Reduced embeddings to 10 dimensions.")
        
        # Show example of reduced embedding
        first_reduced = reduced_embeddings[first_code]
        st.write(f"Example reduced embedding for '{first_code}':")
        st.write(pd.DataFrame([first_reduced], columns=[f"Dim {i+1}" for i in range(10)]))
        
        # Show explained variance
        explained_variance = reducer.explained_variance_ratio_
        cum_explained_variance = np.cumsum(explained_variance)
        
        st.write("Explained variance by dimension:")
        variance_df = pd.DataFrame({
            "Dimension": [f"Dim {i+1}" for i in range(10)],
            "Explained Variance": explained_variance,
            "Cumulative Explained Variance": cum_explained_variance
        })
        st.dataframe(variance_df)
        
        st.write(f"Total variance explained by 10 dimensions: {cum_explained_variance[-1]:.4f} ({cum_explained_variance[-1]*100:.2f}%)")
        
        # NEW: Interpret what each dimension means
        st.subheader("What Each Dimension Represents:")
        
        interpretations = interpret_embedding_dimensions(reducer, embeddings, all_codes, dim_to_concept)
        
        for dim_idx, interpretation in interpretations.items():
            st.write(f"**Dimension {dim_idx + 1}**: {interpretation}")
        
        # Enhanced explanation of PCA results
        st.markdown("""
        ### Understanding the Explained Variance and Dimensions
        
        The table above shows how much information each dimension captures, while the interpretations show what medical concepts each dimension represents.
        
        - Each dimension is a combination of the original medical concept dimensions
        - The ICD-10 codes listed for each dimension are those that have the strongest values in that dimension
        - This helps us understand what each numerical dimension actually means in medical terms
        
        This interpretation is critical for making embeddings actionable in healthcare settings, as it connects the abstract numerical representations back to medical concepts that clinicians and administrators understand.
        """)
    
    # Step 4: Prepare Data for XGBoost
    st.header("Step 4: Prepare Data for Modeling")
    
    # Enhanced explanation for data preparation
    st.markdown("""
    ### Why Data Preparation Is Crucial
    
    This step transforms our raw data and embeddings into a format suitable for machine learning:
    
    1. **Patient-level aggregation**: Moves from diagnosis-level to patient-level data
    2. **Multiple diagnosis handling**: Combines embeddings for patients with multiple conditions
    3. **Feature matrix creation**: Creates a consistent set of features for each patient
    4. **Demographic integration**: Combines embeddings with age and gender features
    
    The key innovation here is how we handle patients with multiple diagnoses. By averaging the embeddings of all a patient's conditions, we create a single vector that represents their overall health status. This approach:
    
    - Accounts for comorbidities (multiple conditions)
    - Maintains consistent feature dimensions regardless of diagnosis count
    - Preserves the semantic relationships between conditions
    - Creates a holistic representation of patient health
    
    Without embeddings, handling multiple diagnoses would be much more complex and less effective.
    """)
    
    with st.spinner("Preparing data for XGBoost model..."):
        # Prepare features and target
        X, y, member_ids, member_codes = prepare_modeling_data(df, reduced_embeddings, embedding_dim=10)
        
        # Feature names
        feature_names = ['age', 'gender'] + [f'emb_dim_{i+1}' for i in range(10)]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.success(f"Prepared feature matrix with shape {X.shape} and target vector with shape {y.shape}")
        
        # Display feature matrix example
        st.subheader("Example Feature Matrix:")
        example_df = pd.DataFrame(X[:5], columns=feature_names)
        st.dataframe(example_df)
        
        # Enhanced explanation of feature matrix
        st.markdown("""
        ### Understanding the Feature Matrix
        
        Each row in this matrix represents one member, with:
        
        - **Demographics**: Age and gender (first two columns)
        - **Embedding dimensions**: The 10 embedding dimensions that represent their diagnoses
        
        This compact representation replaces what would otherwise be thousands of sparse binary columns in a one-hot encoding approach. The embedding dimensions capture complex patterns in the ICD-10 codes that relate to healthcare costs.
        """)
    
    # Step 5: Train XGBoost Model
    st.header("Step 5: Train XGBoost Model for Cost Prediction")
    
    # Enhanced explanation for XGBoost training
    st.markdown("""
    ### Why XGBoost Is Well-Suited for Embeddings
    
    XGBoost is a powerful gradient boosting algorithm that excels at finding complex patterns in data:
    
    1. **Handles non-linear relationships**: Can capture complex interactions between embeddings and costs
    2. **Feature importance**: Identifies which embedding dimensions most impact costs
    3. **Robust to irrelevant features**: Automatically identifies and focuses on important dimensions
    4. **High performance**: Works well with the dense numerical features created by our embeddings
    
    The combination of embeddings and XGBoost is particularly powerful because:
    
    - Embeddings provide a rich, semantic representation of diagnoses
    - XGBoost can identify complex patterns in these representations
    - Together they can uncover nuanced relationships between medical conditions and costs
    
    This approach outperforms traditional one-hot encoding with linear models, which struggle with the high dimensionality and can't capture complex interactions.
    """)
    
    with st.spinner("Training XGBoost model..."):
        # Train model
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,colsample_bytree=0.8,
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
        
        # Enhanced explanation of metrics
        st.markdown("""
        ### Understanding Model Performance
        
        These metrics tell us how well our embedding-based approach predicts healthcare costs:
        
        - **Mean Absolute Error (MAE)**: Average dollar amount our predictions are off by
        - **Root Mean Squared Error (RMSE)**: Similar to MAE but penalizes large errors more
        - **R² Score**: Proportion of variance explained (higher is better, 1.0 is perfect)
        
        In real-world healthcare settings, even small improvements in these metrics can translate to millions of dollars in better financial planning and risk adjustment.
        """)
        
        # Sample predictions
        st.subheader("Sample Predictions:")
        sample_indices = random.sample(range(len(X_test)), min(5, len(X_test)))
        samples_df = pd.DataFrame({
            "Actual": [f"${y_test[i]:.2f}" for i in sample_indices],
            "Predicted": [f"${predictions[i]:.2f}" for i in sample_indices],
            "Difference": [f"${(predictions[i] - y_test[i]):.2f}" for i in sample_indices]
        })
        st.dataframe(samples_df)
    
    # Step 6: Feature Importance and Impact
    st.header("Step 6: Analyze Feature Importance and ICD-10 Code Significance")
    
    # Enhanced explanation for feature importance
    st.markdown("""
    ### Why Feature Analysis Is Essential
    
    Beyond prediction accuracy, we need to understand which factors drive healthcare costs. This analysis:
    
    1. **Identifies key cost drivers**: Shows which embedding dimensions most influence costs
    2. **Provides interpretability**: Makes the "black box" of embeddings more transparent
    3. **Informs interventions**: Helps payers focus on managing the most impactful conditions
    4. **Validates embedding quality**: Confirms that our embeddings capture meaningful patterns
    
    Feature importance tells us which features were most useful for prediction, while feature impact shows the direction and magnitude of each feature's effect on costs.
    
    This level of insight is impossible with traditional one-hot encoding approaches, which would give us thousands of binary features with small individual impacts. With embeddings, we can identify meaningful patterns across related diagnoses.
    """)
    
    with st.spinner("Analyzing feature importance and ICD-10 code significance..."):
        # Display feature importance
        st.subheader("Feature Importance:")
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        st.dataframe(importance_df)
        
        # Calculate feature impact
        st.subheader("Feature Impact Analysis:")
        impacts = calculate_feature_impact(model, X_test, feature_names)
        impact_df = pd.DataFrame({
            'Feature': list(impacts.keys()),
            'Impact': list(impacts.values())
        }).sort_values(by='Impact', ascending=False)
        
        st.dataframe(impact_df)
        
        # NEW: Identify the most important ICD-10 codes based on the feature importance
        st.subheader("Most Influential ICD-10 Codes for Cost Prediction:")
        
        important_codes_df = identify_important_codes(model, feature_names, reduced_embeddings, 
                                                   all_codes, icd_descriptions)
        
        st.dataframe(important_codes_df)
        
        # Enhanced explanation of feature importance results
        st.markdown("""
        ### Interpreting the Results and Key ICD-10 Codes
        
        The feature importance analysis reveals:
        
        - **Which embedding dimensions matter most**: Some dimensions capture patterns that strongly predict costs
        - **Relative importance of demographics vs. diagnoses**: How much age and gender matter compared to medical conditions
        
        The ICD-10 code analysis translates these abstract dimensions back into actionable medical information:
        
        - **Most influential diagnoses**: Specific ICD-10 codes that have the strongest impact on cost predictions
        - **Direction of influence**: Which conditions tend to increase or decrease costs
        - **Medical patterns**: Groups of related conditions that drive healthcare expenditures
        
        This translation from embedding dimensions back to specific diagnoses is critical for healthcare organizations to take action based on the model's insights. For example:
        
        - Care management teams can focus on members with high-impact conditions
        - Risk adjustment teams can ensure proper documentation of cost-driving diagnoses
        - Medical management can develop programs targeted at the most impactful conditions
        """)
        
        st.success("Analysis complete! Now we can see not just which embedding dimensions matter, but which specific ICD-10 codes have the most influence on healthcare costs.")
        
        # Final explanation
        st.header("Conclusion: From Embeddings Back to Actionable ICD-10 Insights")
        st.markdown("""
        This demonstration completes the full cycle of ICD-10 analysis:
        
        1. **From codes to embeddings**: We transformed categorical ICD-10 codes into numerical vectors
        2. **From raw embeddings to reduced dimensions**: We compressed the information to 10 key dimensions
        3. **From dimensions to predictions**: We used these dimensions to predict healthcare costs
        4. **From abstract dimensions back to specific codes**: We translated the model's insights back into actionable ICD-10 codes
        
        This last step is crucial for real-world implementation. While embeddings are powerful for machine learning models, healthcare decision-makers need interpretable insights expressed in the language of medical conditions, not abstract mathematical dimensions.
        
        By connecting the abstract embedding dimensions back to specific ICD-10 codes, we've made the model's insights actionable for:
        
        - **Clinicians**: Who think in terms of medical conditions, not embedding dimensions
        - **Care managers**: Who need to know which specific conditions to focus on
        - **Financial planners**: Who need to understand which diagnoses drive costs
        - **Risk adjusters**: Who need to ensure proper documentation of high-impact conditions
        
        This bidirectional translation - from codes to embeddings and back again - preserves the mathematical power of embeddings while maintaining the interpretability needed in healthcare.
        """)
