import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION VARIABLES - can be modified by users
# ============================================================================
INPUT_CSV_FILE = 'medically_consistent_trauma_reg_200.csv'  # input data
OUTPUT_FOLDER_PREFIX = 'text_emb_rf_xgb_mdl_res'  # output folder prefix
# ============================================================================

# Advanced text embedding imports - only using Gensim for Word2Vec
try:
    import gensim
    from gensim.models import Word2Vec
    from gensim.parsing.preprocessing import preprocess_string
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Gensim not available. Install with: pip install gensim")

class AdvancedTextEmbeddingISSPredictor:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.word2vec_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters, keep letters and numbers
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_medical_keywords(self, text):
        """Extract medical keywords with enhanced vocabulary"""
        medical_keywords = [
            # Basic trauma terms
            'fracture', 'laceration', 'hemorrhage', 'contusion', 'pneumothorax',
            'hemothorax', 'cardiac', 'trauma', 'injury', 'bleeding', 'shock',
            'consciousness', 'unconscious', 'conscious', 'vital', 'signs',
            'blood', 'pressure', 'heart', 'rate', 'respiratory', 'breathing',
            'oxygen', 'saturation', 'temperature', 'pain', 'swelling',
            'bruising', 'abrasion', 'penetrating', 'blunt', 'burn', 'crush',
            'amputation', 'dislocation', 'sprain', 'strain', 'concussion',
            'brain', 'injury', 'spinal', 'cord', 'chest', 'abdomen', 'pelvis',
            'extremity', 'limb', 'head', 'neck', 'face', 'eye', 'ear', 'nose',
            'throat', 'airway', 'breathing', 'circulation', 'disability',
            'exposure', 'mvc', 'motor', 'vehicle', 'accident', 'fall', 'assault',
            'gunshot', 'wound', 'stab', 'knife', 'blade', 'bullet', 'explosion',
            'fire', 'smoke', 'inhalation', 'drowning', 'electrical', 'chemical',
            'aortic', 'dissection', 'hematoma', 'subdural', 'epidural',
            'peritonitis', 'rupture', 'spleen', 'liver', 'kidney', 'bladder',
            'urethral', 'vascular', 'ischemia', 'compartment', 'syndrome',
            
            # Enhanced medical terms
            'intracranial', 'subarachnoid', 'subdural', 'epidural', 'hematoma',
            'cerebral', 'edema', 'herniation', 'infarction', 'embolism',
            'thrombosis', 'aneurysm', 'stenosis', 'occlusion', 'perforation',
            'rupture', 'avulsion', 'amputation', 'degloving', 'crush',
            'compartment', 'syndrome', 'necrosis', 'gangrene', 'infection',
            'sepsis', 'bacteremia', 'pneumonia', 'empyema', 'abscess',
            'peritonitis', 'pancreatitis', 'cholecystitis', 'appendicitis',
            'diverticulitis', 'colitis', 'enteritis', 'gastritis', 'ulcer',
            'varices', 'cirrhosis', 'hepatitis', 'nephritis', 'pyelonephritis',
            'cystitis', 'urethritis', 'prostatitis', 'orchitis', 'epididymitis',
            'endometritis', 'salpingitis', 'oophoritis', 'mastitis',
            'osteomyelitis', 'arthritis', 'bursitis', 'tendonitis', 'myositis',
            'dermatitis', 'cellulitis', 'erysipelas', 'impetigo', 'folliculitis'
        ]
        
        text_lower = str(text).lower()
        found_keywords = []
        
        for keyword in medical_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return ' '.join(found_keywords)
    
    def create_word2vec_embeddings(self, texts, vector_size=100):
        """Create Word2Vec embeddings"""
        if not GENSIM_AVAILABLE:
            print("Gensim not available, skipping Word2Vec")
            return None
            
        print("=== Creating Word2Vec Embeddings ===")
        
        # Tokenize texts
        tokenized_texts = []
        for text in texts:
            if pd.notna(text) and text != '':
                tokens = preprocess_string(text)
                tokenized_texts.append(tokens)
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=5,
            min_count=2,
            workers=4,
            sg=1,  # Skip-gram
            epochs=10
        )
        
        # Create document embeddings
        doc_embeddings = []
        for text in texts:
            if pd.notna(text) and text != '':
                tokens = preprocess_string(text)
                word_vectors = []
                for token in tokens:
                    if token in self.word2vec_model.wv:
                        word_vectors.append(self.word2vec_model.wv[token])
                
                if word_vectors:
                    # Average word vectors for document embedding
                    doc_embedding = np.mean(word_vectors, axis=0)
                else:
                    doc_embedding = np.zeros(vector_size)
            else:
                doc_embedding = np.zeros(vector_size)
            
            doc_embeddings.append(doc_embedding)
        
        # Create DataFrame
        w2v_df = pd.DataFrame(
            doc_embeddings,
            columns=[f'w2v_{i}' for i in range(vector_size)]
        )
        
        print(f"Word2Vec embeddings created: {w2v_df.shape[1]} features")
        return w2v_df
    
    def create_text_features(self, df):
        """Create comprehensive text features"""
        print("=== Advanced Text Feature Engineering ===")
        
        # Process main text columns
        text_columns = ['Injury Text', 'INJ_CAU_MEMO', 'llm_extracted_data_columns']
        
        # Combine all text data
        combined_text = []
        for idx, row in df.iterrows():
            texts = []
            for col in text_columns:
                if col in df.columns and pd.notna(row[col]):
                    texts.append(str(row[col]))
            combined_text.append(' '.join(texts))
        
        df['combined_text'] = combined_text
        
        # Preprocess text
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        # Extract medical keywords
        df['medical_keywords'] = df['combined_text'].apply(self.extract_medical_keywords)
        
        # Create basic text features
        df['text_length'] = df['processed_text'].apply(len)
        df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()) if x else 0)
        df['keyword_count'] = df['medical_keywords'].apply(lambda x: len(x.split()) if x else 0)
        df['keyword_density'] = df['keyword_count'] / (df['word_count'] + 1)
        
        # Create advanced text features
        df['sentence_count'] = df['combined_text'].apply(lambda x: len(str(x).split('.')) if x else 0)
        df['avg_word_length'] = df['processed_text'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x and len(x.split()) > 0 else 0
        )
        df['unique_word_ratio'] = df['processed_text'].apply(
            lambda x: len(set(x.split())) / len(x.split()) if x and len(x.split()) > 0 else 0
        )
        
        # Create n-gram features
        df['bigram_count'] = df['processed_text'].apply(
            lambda x: len([x[i:i+2] for i in range(len(x)-1)]) if x else 0
        )
        df['trigram_count'] = df['processed_text'].apply(
            lambda x: len([x[i:i+3] for i in range(len(x)-2)]) if x else 0
        )
        
        print(f"Advanced text features created:")
        print(f"- Average text length: {df['text_length'].mean():.1f}")
        print(f"- Average word count: {df['word_count'].mean():.1f}")
        print(f"- Average keyword count: {df['keyword_count'].mean():.1f}")
        print(f"- Average sentence count: {df['sentence_count'].mean():.1f}")
        
        return df
    
    def create_word_embeddings(self, df, max_features=100):
        """Create comprehensive word embedding features"""
        print("=== Creating Advanced Word Embedding Features ===")
        
        # TF-IDF vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        tfidf_features = self.tfidf_vectorizer.fit_transform(df['processed_text'])
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        # Count vectorization
        self.count_vectorizer = CountVectorizer(
            max_features=50,
            ngram_range=(1, 1),
            stop_words='english',
            min_df=1
        )
        
        keyword_features = self.count_vectorizer.fit_transform(df['medical_keywords'])
        keyword_df = pd.DataFrame(
            keyword_features.toarray(),
            columns=[f'keyword_{i}' for i in range(keyword_features.shape[1])]
        )
        
        # Word2Vec embeddings
        w2v_df = self.create_word2vec_embeddings(df['processed_text'].tolist())
        
        print(f"Advanced word embedding features created:")
        print(f"- TF-IDF features: {tfidf_df.shape[1]}")
        print(f"- Keyword features: {keyword_df.shape[1]}")
        if w2v_df is not None:
            print(f"- Word2Vec features: {w2v_df.shape[1]}")
        
        return tfidf_df, keyword_df, w2v_df
    
    def create_enhanced_features(self, df):
        """Create enhanced features with medical logic"""
        print("=== Creating Enhanced Features ===")
        
        df_enhanced = df.copy()
        
        # 1. Create composite features
        df_enhanced['Age_Group'] = pd.cut(df_enhanced['Age'], 
                                         bins=[0, 30, 50, 70, 100], 
                                         labels=['Young', 'Middle', 'Senior', 'Elderly'])
        
        df_enhanced['BP_Status'] = pd.cut(df_enhanced['SBP on Admission'],
                                         bins=[0, 90, 120, 160, 300],
                                         labels=['Hypotensive', 'Normal', 'Elevated', 'High'])
        
        df_enhanced['GCS_Status'] = pd.cut(df_enhanced['GCS on Admission'],
                                          bins=[0, 8, 13, 15],
                                          labels=['Severe', 'Moderate', 'Mild'])
        
        # 2. Create advanced risk score
        df_enhanced['Advanced_Risk_Score'] = (
            (15 - df_enhanced['GCS on Admission']) * 3 +
            (200 - df_enhanced['SBP on Admission']) / 8 +
            (df_enhanced['Unassisted Resp Rate on Admission'] - 12) * 1.5 +
            (df_enhanced['Age'] - 40) / 20 +
            df_enhanced['keyword_count'] * 2
        )
        
        # 3. Create age-related features
        df_enhanced['Age_Squared'] = df_enhanced['Age'] ** 2
        df_enhanced['Age_Log'] = np.log(df_enhanced['Age'] + 1)
        df_enhanced['Age_Cubic'] = df_enhanced['Age'] ** 3
        
        # 4. Create physiological ratio features
        df_enhanced['BP_HR_Ratio'] = df_enhanced['SBP on Admission'] / (df_enhanced['Unassisted Resp Rate on Admission'] + 1)
        df_enhanced['GCS_Age_Ratio'] = df_enhanced['GCS on Admission'] / (df_enhanced['Age'] + 1)
        df_enhanced['RTS_TRISS_Ratio'] = df_enhanced['RTS on Admission'] / (df_enhanced['TRISS'] + 0.1)
        
        # 5. Create text-related composite features
        df_enhanced['text_risk_score'] = (
            df_enhanced['keyword_count'] * 3 +
            df_enhanced['text_length'] / 50 +
            df_enhanced['keyword_density'] * 15 +
            df_enhanced['sentence_count'] * 2
        )
        
        # 6. Create medical logic features
        df_enhanced['severe_injury_flag'] = df_enhanced['Injury Text'].apply(
            lambda x: 1 if any(severe in str(x).lower() for severe in [
                'brain injury', 'spinal cord', 'aortic dissection', 'liver laceration',
                'pelvic fracture', 'pneumothorax', 'intracranial', 'hemorrhage'
            ]) else 0
        )
        
        df_enhanced['penetrating_injury_flag'] = (df_enhanced['Injury Type'] == 'Penetrating').astype(int)
        df_enhanced['elderly_flag'] = (df_enhanced['Age'] > 65).astype(int)
        df_enhanced['critical_gcs'] = (df_enhanced['GCS on Admission'] <= 8).astype(int)
        df_enhanced['hypotensive'] = (df_enhanced['SBP on Admission'] < 90).astype(int)
        df_enhanced['tachypneic'] = (df_enhanced['Unassisted Resp Rate on Admission'] > 30).astype(int)
        df_enhanced['bradypneic'] = (df_enhanced['Unassisted Resp Rate on Admission'] < 10).astype(int)
        
        return df_enhanced
    
    def prepare_features(self, df_enhanced, tfidf_df, keyword_df, w2v_df=None):
        """Prepare all features with advanced embeddings"""
        # Numerical features
        numeric_features = [
            'Age', 'SBP on Admission', 'Unassisted Resp Rate on Admission',
            'GCS on Admission', 'RTS on Admission', 'TRISS',
            'Advanced_Risk_Score', 'Age_Squared', 'Age_Log', 'Age_Cubic',
            'BP_HR_Ratio', 'GCS_Age_Ratio', 'RTS_TRISS_Ratio',
            'text_length', 'word_count', 'keyword_count', 'keyword_density',
            'sentence_count', 'avg_word_length', 'unique_word_ratio',
            'bigram_count', 'trigram_count', 'text_risk_score', 
            'severe_injury_flag', 'penetrating_injury_flag',
            'elderly_flag', 'critical_gcs', 'hypotensive', 'tachypneic', 'bradypneic'
        ]
        
        # Categorical features
        categorical_features = [
            'Gender', 'Injury Type', 'Transfer In', 'Discharge Status',
            'Age_Group', 'BP_Status', 'GCS_Status'
        ]
        
        # Process categorical variables
        X_categorical = pd.DataFrame()
        for feature in categorical_features:
            if feature in df_enhanced.columns:
                le = LabelEncoder()
                X_categorical[feature] = le.fit_transform(df_enhanced[feature].astype(str))
                self.label_encoders[feature] = le
        
        # Process numerical variables
        X_numeric = df_enhanced[numeric_features].copy()
        
        # Combine all features
        feature_dfs = [X_numeric, X_categorical, tfidf_df, keyword_df]
        
        if w2v_df is not None:
            feature_dfs.append(w2v_df)
        
        X_combined = pd.concat(feature_dfs, axis=1)
        y = df_enhanced['ISS']
        
        # Remove rows with NaN
        valid_indices = ~(X_combined.isna().any(axis=1) | y.isna())
        X_combined = X_combined[valid_indices]
        y = y[valid_indices]
        
        self.feature_names = X_combined.columns.tolist()
        
        print(f"Final advanced feature set: {len(X_combined)} samples, {len(X_combined.columns)} features")
        print(f"Feature type distribution:")
        print(f"- Numerical features: {len(numeric_features)}")
        print(f"- Categorical features: {len(categorical_features)}")
        print(f"- TF-IDF features: {tfidf_df.shape[1]}")
        print(f"- Keyword features: {keyword_df.shape[1]}")
        if w2v_df is not None:
            print(f"- Word2Vec features: {w2v_df.shape[1]}")
        
        return X_combined, y
    
    def train_models(self, X, y):
        """Train multiple models with advanced features"""
        print("\n=== Training Models (Advanced Text Embedding Features) ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models with optimized parameters
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=0.5),
            'Lasso Regression': Lasso(alpha=0.05),
            'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_test': y_test,
                'X_test': X_test
            }
            
            print(f"{name} Results:")
            print(f"  MSE: {mse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.3f}")
            print(f"  CV R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        return results, X_test, y_test
    
    def save_results_to_csv(self, results, output_folder):
        """Save model results to CSV files"""
        print(f"\n=== Saving Results to CSV Files ===")
        
        # 1. Save overall model performance comparison
        performance_data = []
        for name, result in results.items():
            performance_data.append({
                'Model': name,
                'R2_Score': result['r2'],
                'MAE': result['mae'],
                'MSE': result['mse'],
                'CV_R2_Mean': result['cv_mean'],
                'CV_R2_Std': result['cv_std']
            })
        
        performance_df = pd.DataFrame(performance_data)
        performance_csv_path = os.path.join(output_folder, 'advanced_model_performance_comparison.csv')
        performance_df.to_csv(performance_csv_path, index=False)
        print(f"✓ Advanced model performance comparison saved to: {performance_csv_path}")
        
        # 2. Save predictions for each model
        for name, result in results.items():
            predictions_df = pd.DataFrame({
                'Actual_ISS': result['y_test'],
                'Predicted_ISS': result['y_pred'],
                'Prediction_Error': result['y_test'] - result['y_pred'],
                'Absolute_Error': np.abs(result['y_test'] - result['y_pred'])
            })
            
            # Add feature values for the test set
            for i, feature_name in enumerate(self.feature_names):
                if i < len(result['X_test'].columns):
                    predictions_df[f'Feature_{feature_name}'] = result['X_test'].iloc[:, i]
            
            predictions_csv_path = os.path.join(output_folder, f'Advanced_{name.replace(" ", "_")}_predictions.csv')
            predictions_df.to_csv(predictions_csv_path, index=False)
            print(f"✓ Advanced {name} predictions saved to: {predictions_csv_path}")
        
        # 3. Save feature importance for tree-based models
        feature_importance_data = []
        for name, result in results.items():
            if hasattr(result['model'], 'feature_importances_'):
                importances = result['model'].feature_importances_
                for i, (feature_name, importance) in enumerate(zip(self.feature_names, importances)):
                    feature_importance_data.append({
                        'Model': name,
                        'Feature_Name': feature_name,
                        'Importance': importance,
                        'Rank': len([x for x in importances if x > importance]) + 1
                    })
        
        if feature_importance_data:
            feature_importance_df = pd.DataFrame(feature_importance_data)
            feature_importance_csv_path = os.path.join(output_folder, 'advanced_feature_importance_analysis.csv')
            feature_importance_df.to_csv(feature_importance_csv_path, index=False)
            print(f"✓ Advanced feature importance analysis saved to: {feature_importance_csv_path}")
        
        # 4. Save model configuration summary
        config_data = {
            'Input_File': INPUT_CSV_FILE,
            'Total_Samples': len(results[list(results.keys())[0]]['y_test']) * 5,  # Approximate
            'Test_Samples': len(results[list(results.keys())[0]]['y_test']),
            'Total_Features': len(self.feature_names),
            'Numerical_Features': 29,
            'Categorical_Features': 7,
            'TFIDF_Features': 100,
            'Keyword_Features': 50,
            'Word2Vec_Features': 100,
            'Training_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Best_Model': max(results.keys(), key=lambda x: results[x]['r2']),
            'Best_R2_Score': max(results[x]['r2'] for x in results.keys()),
            'Model_Type': 'Advanced_Text_Embedding'
        }
        
        config_df = pd.DataFrame([config_data])
        config_csv_path = os.path.join(output_folder, 'advanced_model_configuration_summary.csv')
        config_df.to_csv(config_csv_path, index=False)
        print(f"✓ Advanced model configuration summary saved to: {config_csv_path}")
        
        # 5. Save text embedding analysis
        text_analysis_data = {
            'Text_Processing': 'Advanced',
            'Medical_Keywords_Count': 170,
            'TFIDF_Max_Features': 100,
            'Keyword_Max_Features': 50,
            'Word2Vec_Vector_Size': 100,
            'Word2Vec_Window': 5,
            'Word2Vec_Min_Count': 2,
            'Word2Vec_Epochs': 10,
            'N_Gram_Range_TFIDF': '(1, 2)',
            'N_Gram_Range_Keywords': '(1, 1)',
            'Advanced_Text_Features': 'sentence_count, avg_word_length, unique_word_ratio, bigram_count, trigram_count'
        }
        
        text_analysis_df = pd.DataFrame([text_analysis_data])
        text_analysis_csv_path = os.path.join(output_folder, 'advanced_text_embedding_analysis.csv')
        text_analysis_df.to_csv(text_analysis_csv_path, index=False)
        print(f"✓ Advanced text embedding analysis saved to: {text_analysis_csv_path}")
    
    def plot_results(self, results, X_test, y_test, output_folder):
        """Plot results with advanced features and save to output folder"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model performance comparison
        model_names = list(results.keys())
        r2_scores = [results[name]['r2'] for name in model_names]
        mae_scores = [results[name]['mae'] for name in model_names]
        
        # R² score comparison
        colors = ['green' if x > 0.5 else 'orange' if x > 0.3 else 'red' for x in r2_scores]
        axes[0, 0].bar(model_names, r2_scores, color=colors, alpha=0.7)
        axes[0, 0].set_title('Model R² Score Comparison (Advanced Text Embeddings)')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Good (0.5)')
        axes[0, 0].axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Acceptable (0.3)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE comparison
        axes[0, 1].bar(model_names, mae_scores, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Model MAE Comparison')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Best model prediction vs actual
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        best_y_pred = results[best_model_name]['y_pred']
        
        axes[1, 0].scatter(y_test, best_y_pred, alpha=0.6, color='green')
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual ISS Score')
        axes[1, 0].set_ylabel('Predicted ISS Score')
        axes[1, 0].set_title(f'Best Model ({best_model_name}): Predicted vs Actual')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature importance
        if hasattr(results[best_model_name]['model'], 'feature_importances_'):
            importances = results[best_model_name]['model'].feature_importances_
            
            # Get top 15 most important features
            top_indices = np.argsort(importances)[-15:]
            top_importances = importances[top_indices]
            top_features = [self.feature_names[i] for i in top_indices]
            
            axes[1, 1].barh(range(len(top_importances)), top_importances)
            axes[1, 1].set_yticks(range(len(top_importances)))
            axes[1, 1].set_yticklabels(top_features)
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title(f'{best_model_name} Feature Importance')
        
        plt.tight_layout()
        
        # Save plot to output folder
        plot_path = os.path.join(output_folder, 'advanced_text_embedding_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Advanced text embedding results visualization saved to: {plot_path}")
        plt.show()
        
        return best_model_name
    
    def save_model(self, results, best_model_name, output_folder):
        """Save advanced model to output folder"""
        import joblib
        
        best_model = results[best_model_name]['model']
        
        model_data = {
            'best_model': best_model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'count_vectorizer': self.count_vectorizer,
            'word2vec_model': self.word2vec_model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'results': results,
            'best_model_name': best_model_name
        }
        
        model_path = os.path.join(output_folder, 'advanced_iss_prediction_model.pkl')
        joblib.dump(model_data, model_path)
        print(f"\n✓ Advanced model saved to: {model_path}")
        print(f"Best model: {best_model_name}")

def create_output_folder():
    """Create timestamped output folder"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{OUTPUT_FOLDER_PREFIX}_{timestamp}"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"✓ Created output folder: {folder_name}")
    else:
        print(f"⚠ Output folder already exists: {folder_name}")
    
    return folder_name

def main():
    # Create output folder
    output_folder = create_output_folder()
    
    # Load data
    print(f"Loading data from: {INPUT_CSV_FILE}")
    try:
        df = pd.read_csv(INPUT_CSV_FILE)
        print(f"✓ Successfully loaded {len(df)} records from {INPUT_CSV_FILE}")
    except FileNotFoundError:
        print(f"❌ Error: File '{INPUT_CSV_FILE}' not found!")
        print(f"Please make sure the file exists in the current directory.")
        return
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Initialize advanced predictor
    predictor = AdvancedTextEmbeddingISSPredictor()
    
    # Create advanced text features
    df = predictor.create_text_features(df)
    
    # Create advanced word embeddings
    tfidf_df, keyword_df, w2v_df = predictor.create_word_embeddings(df)
    
    # Create enhanced features
    df_enhanced = predictor.create_enhanced_features(df)
    
    # Prepare all features
    X, y = predictor.prepare_features(df_enhanced, tfidf_df, keyword_df, w2v_df)
    
    # Train models
    results, X_test, y_test = predictor.train_models(X, y)
    
    # Save results to CSV files
    predictor.save_results_to_csv(results, output_folder)
    
    # Plot results and save to output folder
    best_model_name = predictor.plot_results(results, X_test, y_test, output_folder)
    
    # Save model to output folder
    predictor.save_model(results, best_model_name, output_folder)
    
    # Display results
    print(f"\n=== Advanced Model Results ===")
    best_result = results[best_model_name]
    print(f"Best Model: {best_model_name}")
    print(f"R²: {best_result['r2']:.3f}")
    print(f"MAE: {best_result['mae']:.2f}")
    print(f"MSE: {best_result['mse']:.2f}")
    print(f"CV R²: {best_result['cv_mean']:.3f} (±{best_result['cv_std']:.3f})")
    
    # Compare with previous model
    print(f"\n=== Comparison with Previous Model ===")
    print("Previous model (basic text features):")
    print("- R²: ~0.906")
    print("- MAE: ~4.66")
    
    print(f"\nAdvanced model (with Word2Vec embeddings):")
    print(f"- R²: {best_result['r2']:.3f}")
    print(f"- MAE: {best_result['mae']:.2f}")
    
    improvement = best_result['r2'] - 0.906
    if improvement > 0:
        print(f"Improvement: +{improvement:.3f} R² points")
    else:
        print(f"Change: {improvement:.3f} R² points")
    
    print(f"\n=== Output Summary ===")
    print(f"All results saved to folder: {output_folder}")
    print(f"Files generated:")
    print(f"- advanced_model_performance_comparison.csv")
    print(f"- Advanced_[Model]_predictions.csv (for each model)")
    print(f"- advanced_feature_importance_analysis.csv")
    print(f"- advanced_model_configuration_summary.csv")
    print(f"- advanced_text_embedding_analysis.csv")
    print(f"- advanced_text_embedding_results.png")
    print(f"- advanced_iss_prediction_model.pkl")

if __name__ == "__main__":
    main()
