import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import gensim
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.decomposition import PCA
import string
import random

# Set page config
st.set_page_config(page_title="ICD-10 Embedding Demo", layout="wide")

# App title and description
st.title("ICD-10 Code Embedding for Healthcare Cost Prediction")
st.markdown("""
This application demonstrates how embedding ICD-10 diagnosis codes can improve healthcare cost prediction models.
Instead of treating each code as a separate variable, we convert them into meaningful numerical vectors.
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

# Function to create embeddings using Word2Vec
def create_word2vec_embeddings(descriptions, vector_size=100, window=5, min_count=1, epochs=100):
    # Preprocess descriptions
    processed_descriptions = {code: preprocess_text(desc) for code, desc in descriptions.items()}
    
    # Prepare data for Word2Vec
    sentences = list(processed_descriptions.values())
    
    # Train Word2Vec model
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, 
                     min_count=min_count, workers=4, epochs=epochs)
    
    # Create embeddings for each ICD-10 code
    embeddings = {}
    for code, tokens in processed_descriptions.items():
        # Get word vectors for each token in the description
        word_vectors = [model.wv[word] for word in tokens if word in model.wv]
        
        if word_vectors:
            # Average the word vectors to get a single vector for the code
            embedding = np.mean(word_vectors, axis=0)
        else:
            embedding = np.zeros(vector_size)
        
        embeddings[code] = embedding
    
    return embeddings

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
    
    return reduced_dict, reducer

# Function to prepare data for modeling
def prepare_modeling_data(df, embedding_dict, embedding_dim=10):
    # Group by member_id to get all codes for each member
    member_groups = df.groupby('member_id')
    
    # Prepare the feature matrix
    X_data = []
    y_data = []
    member_ids = []
    
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
        member_ids.append(member_id)
    
    return np.array(X_data), np.array(y_data), member_ids

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

# Main workflow
st.header("Step 1: Generate Sample Data")
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
    with st.spinner("Creating embeddings from ICD-10 descriptions..."):
        # Create Word2Vec embeddings
        vector_size = 32  # Smaller vector size for faster computation
        embeddings = create_word2vec_embeddings(icd_descriptions, vector_size=vector_size, epochs=50)
        
        # Display embedding information
        first_code = list(embeddings.keys())[0]
        first_embedding = embeddings[first_code]
        st.success(f"Created embeddings for {len(embeddings)} ICD-10 codes. Original dimension: {len(first_embedding)}")
        
        # Display sample cosine similarities
        st.subheader("Sample Embedding Similarities:")
        
        # Let's check if diabetes codes are similar
        if "E11.9" in embeddings and "E10.9" in embeddings:
            diabetes_sim = np.dot(embeddings["E11.9"], embeddings["E10.9"]) / (
                np.linalg.norm(embeddings["E11.9"]) * np.linalg.norm(embeddings["E10.9"])
            )
            st.write(f"Similarity between 'E11.9' (Type 2 diabetes) and 'E10.9' (Type 1 diabetes): {diabetes_sim:.4f}")
        
        # Check if asthma codes are similar
        if "J45.909" in embeddings and "J45.20" in embeddings:
            asthma_sim = np.dot(embeddings["J45.909"], embeddings["J45.20"]) / (
                np.linalg.norm(embeddings["J45.909"]) * np.linalg.norm(embeddings["J45.20"])
            )
            st.write(f"Similarity between 'J45.909' (Unspecified asthma) and 'J45.20' (Mild asthma): {asthma_sim:.4f}")
        
        # Check if unrelated codes are less similar
        if "E11.9" in embeddings and "J45.909" in embeddings:
            unrelated_sim = np.dot(embeddings["E11.9"], embeddings["J45.909"]) / (
                np.linalg.norm(embeddings["E11.9"]) * np.linalg.norm(embeddings["J45.909"])
            )
            st.write(f"Similarity between 'E11.9' (Type 2 diabetes) and 'J45.909' (Asthma): {unrelated_sim:.4f}")
    
    # Step 3: Reduce Dimensions
    with st.spinner("Reducing embedding dimensions to 10..."):
        # Reduce to 10 dimensions
        reduced_embeddings, reducer = reduce_dimensions(embeddings, n_components=10)
        
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
    
    # Step 4: Prepare Data for XGBoost
    with st.spinner("Preparing data for XGBoost model..."):
        # Prepare features and target
        X, y, member_ids = prepare_modeling_data(df, reduced_embeddings, embedding_dim=10)
        
        # Feature names
        feature_names = ['age', 'gender'] + [f'emb_dim_{i+1}' for i in range(10)]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.success(f"Prepared feature matrix with shape {X.shape} and target vector with shape {y.shape}")
    
    # Step 5: Train XGBoost Model
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
    with st.spinner("Analyzing feature importance..."):
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
        
        st.success("Analysis complete! Notice how some embedding dimensions have high importance - these capture meaningful patterns in the ICD-10 codes that predict costs.")
        
        # Final explanation
        st.subheader("Summary:")
        st.markdown("""
        This demonstration shows how embeddings can effectively represent ICD-10 codes in a predictive model:
        
        1. We started with raw ICD-10 codes and their text descriptions
        2. We used Word2Vec to create embeddings that place similar medical conditions near each other in vector space
        3. We reduced the embeddings to exactly 10 dimensions using PCA
        4. We used these embeddings along with demographic features to predict healthcare costs
        5. We analyzed which embedding dimensions were most important for prediction
        
        The key advantage is that similar medical conditions have similar embeddings, allowing the model to generalize across related diagnoses.
        """)
