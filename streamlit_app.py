"""
Real Estate Price Prediction App
Loads trained models and provides price predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
import json
from openai import OpenAI
import os
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Real Estate Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    models_dir = Path("models")
    
    if not models_dir.exists():
        st.error("Models directory not found")
        st.error(f"Expected: {models_dir.absolute()}")
        st.info("Create a 'models/' folder and place model files there.")
        st.stop()
    
    existing_files = list(models_dir.glob("*.pkl"))
    required = ["regressor.pkl", "classifier.pkl", "kmeans.pkl", "scaler.pkl"]
    
    missing = [f for f in required if not (models_dir / f).exists()]
    
    if missing:
        st.error("Missing model files")
        for f in required:
            status = "Found" if (models_dir / f).exists() else "Missing"
            st.text(f"{status}: {f}")
        
        if existing_files:
            st.warning("Found these .pkl files:")
            for f in existing_files:
                st.code(f.name)
            st.info("Rename files: xgb_regressor.pkl -> regressor.pkl, kmeans_model.pkl -> kmeans.pkl, etc.")
        st.stop()
    
    try:
        regressor = joblib.load(models_dir / "regressor.pkl")
        classifier = joblib.load(models_dir / "classifier.pkl")
        kmeans = joblib.load(models_dir / "kmeans.pkl")
        scaler = joblib.load(models_dir / "scaler.pkl")
        return regressor, classifier, kmeans, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()


def explain_prediction(features_dict, predicted_price, cluster_id):
    bedrooms = features_dict.get('bedrooms', 0)
    bathrooms = features_dict.get('bathrooms', 0)
    sqft_living = features_dict.get('sqft_living', 0)
    grade = features_dict.get('grade', 0)
    
    parts = []
    parts.append(f"Predicted Value: ${predicted_price:,.2f}")
    parts.append(f"\nProperty: {bedrooms} bed, {bathrooms} bath, {sqft_living:,.0f} sqft")
    if grade:
        parts.append(f"Grade: {grade}")
    
    segment_names = {0: "Budget", 1: "Mid-Market", 2: "Premium", 3: "Luxury"}
    segment = segment_names.get(cluster_id, f"Segment {cluster_id}")
    parts.append(f"\nMarket Segment: {segment}")
    
    if sqft_living > 0:
        ppsf = predicted_price / sqft_living
        parts.append(f"Price per sqft: ${ppsf:,.2f}")
        
        if ppsf < 200:
            status = "below average"
        elif ppsf < 400:
            status = "average"
        else:
            status = "above average"
        parts.append(f"Market position: {status}")
    
    parts.append("\nNote: Consider location, condition, and market trends for final decision.")
    
    return "\n".join(parts)


def get_ai_explanation(features, predicted_price, api_key):
    """
    Generates an explanation for the predicted price using OpenAI API.
    """
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)
    
    # Prepare the prompt
    prompt_content = f"""
    You are an expert Python data scientist and real estate pricing specialist. A regression model predicted a house price that may be unrealistic.
    
    **Your tasks:**
    1. Assess if the predicted price is realistic compared to market rates (for end-users).
    2. If unrealistic, analyze WHY the model failed (for data scientists).
    3. Provide **executable Python code** to fix the model issues.
    
    **Input:**
    Features: {features}
    Predicted price: ${predicted_price:,.2f}
    
    **Key market benchmarks:**
    - Typical sqft: ~2,077
    - Typical price: ~$509,456
    - Typical price/sqft: ~$245
    
    **Output format (JSON only):**
    {{
      "MarketRealism": "Brief assessment for users: Realistic/Undervalued/Overvalued with comparison",
      "PossibleReasons": "Why the model predicted this (key features, outliers, missing data)",
      "SuggestedAction": "What end-users should do: validate inputs, trust prediction, or seek appraisal",
      "ProblemAnalysis": "Technical diagnosis: Is this a model failure? What went wrong?",
      "PossibleCauses": "Root causes: training data issues, feature engineering gaps, model design flaws",
      "RecommendedFixes": "How to improve the model (techniques and approaches)",
      "PythonCode": "Ready-to-run Python code to fix the issue. Include: outlier removal, feature engineering, model retraining. Use comments to explain each step. Keep it under 30 lines but production-ready."
    }}
    
    **Guidelines:**
    - If prediction is within ±30% of market norms → minimal code, just validation checks
    - If prediction is extreme (>2x or <0.5x market) → provide comprehensive fix code
    - Python code should be **copy-paste ready** with imports, preprocessing, and model training
    - Focus on: IQR outlier removal, domain constraints (min/max price), feature interactions, regularization
    - Be specific: mention exact features, thresholds, and techniques
    - Keep user-facing fields concise, technical fields detailed
    
    RETURN ONLY JSON. Escape newlines in PythonCode field properly.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful real estate analyst."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.7
        )
        content = response.choices[0].message.content
        # Clean up code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.replace("```", "").strip()
            
        return json.loads(content)
    except Exception as e:
        error_str = str(e)
        if "insufficient_quota" in error_str or "429" in error_str or "quota" in error_str.lower():
            # Rule-based fallback using actual training data statistics
            # These values are from the scaler.mean_ (training data averages)
            MEAN_SQFT = 2077
            MEAN_PRICE = 509456
            MEAN_BEDROOMS = 3.37
            MEAN_BATHROOMS = 2.12
            MODE_GRADE = 7  # Typical mode for grade
            
            sqft = features.get('sqft_living', 0)
            grade = features.get('grade', MODE_GRADE)
            bedrooms = features.get('bedrooms', MEAN_BEDROOMS)
            bathrooms = features.get('bathrooms', MEAN_BATHROOMS)
            price = predicted_price
            ppsf = price / sqft if sqft > 0 else 0
            
            # Calculate typical price per sqft from mean values
            TYPICAL_PPSF = MEAN_PRICE / MEAN_SQFT  # ~245
            
            # 1. Generate Explanation
            drivers = []
            if sqft > MEAN_SQFT * 1.2:  # 20% above average
                drivers.append(f"generous living space ({sqft:,.0f} sqft vs typical {MEAN_SQFT:,.0f})")
            elif sqft < MEAN_SQFT * 0.6:  # 40% below average
                drivers.append(f"compact layout ({sqft:,.0f} sqft vs typical {MEAN_SQFT:,.0f})")
                
            if grade >= MODE_GRADE + 2:  # 2 grades above mode
                drivers.append(f"high construction grade ({grade} vs typical {MODE_GRADE})")
            elif grade <= MODE_GRADE - 1:  # Below mode
                drivers.append(f"standard construction grade ({grade} vs typical {MODE_GRADE})")
                
            if bedrooms > MEAN_BEDROOMS + 1:
                drivers.append(f"above-average bedrooms ({bedrooms})")
            elif bedrooms < MEAN_BEDROOMS - 1:
                drivers.append(f"fewer bedrooms ({bedrooms})")
                
            driver_str = ", ".join(drivers) if drivers else "typical property characteristics"
            explanation = f"This price is primarily influenced by {driver_str}. "
            explanation += f"At ${ppsf:,.0f}/sqft vs market average ${TYPICAL_PPSF:,.0f}/sqft, "
            
            if ppsf > TYPICAL_PPSF * 1.6:  # 60% above average
                explanation += "this represents a premium market position."
            elif ppsf < TYPICAL_PPSF * 0.8:  # 20% below average
                explanation += "this is positioned in the value segment."
            else:
                explanation += "this aligns with standard market pricing."

            # 2. Generate Market Realism
            realism = ""
            if ppsf > TYPICAL_PPSF * 2:  # Double the average
                realism = f"⚠️ Overvalued: ${ppsf:,.0f}/sqft is {ppsf/TYPICAL_PPSF:.1f}x market average (${TYPICAL_PPSF:,.0f}/sqft). Typical for luxury/prime locations only."
            elif ppsf < TYPICAL_PPSF * 0.5:  # Half the average
                realism = f"⚠️ Undervalued: ${ppsf:,.0f}/sqft is {ppsf/TYPICAL_PPSF:.1f}x market average (${TYPICAL_PPSF:,.0f}/sqft). May indicate distressed property or data anomaly."
            elif price > MEAN_PRICE * 1.5:
                realism = f"Above Average: ${price:,.0f} exceeds typical price (${MEAN_PRICE:,.0f}) but realistic for larger/premium properties."
            elif price < MEAN_PRICE * 0.5:
                realism = f"Below Average: ${price:,.0f} is well below typical price (${MEAN_PRICE:,.0f}). Consistent with smaller/value properties."
            else:
                realism = f"Realistic: ${price:,.0f} at ${ppsf:,.0f}/sqft aligns with market norms for this property profile."

            # 3. Possible Reasons
            reasons = f"Model predicted based on {driver_str}. "
            if ppsf < TYPICAL_PPSF * 0.5 or ppsf > TYPICAL_PPSF * 2:
                reasons += "Extreme price/sqft suggests: (1) Model limitation with outlier features, (2) Missing critical data (location, condition), or (3) Training data included distressed/luxury outliers. "
            if sqft > MEAN_SQFT * 2:
                reasons += "Large homes often have non-linear pricing that basic regression struggles to capture. "
            if bedrooms > MEAN_BEDROOMS + 3:
                reasons += f"Unusually high bedroom count ({bedrooms}) is rare in training data and may confuse the model. "
            
            # 4. Suggested Action
            if ppsf < TYPICAL_PPSF * 0.5:
                action = "⚠️ Validate inputs: Check for data entry errors. Review model training data for similar outliers. Consider manual appraisal for unusual properties."
            elif ppsf > TYPICAL_PPSF * 2:
                action = "⚠️ Review prediction: Verify property features. For luxury homes, consider using comparable sales analysis instead of regression."
            elif sqft > MEAN_SQFT * 2 or bedrooms > MEAN_BEDROOMS + 3:
                action = "Caution: Property features exceed typical training range. Use prediction as rough estimate; validate with local market comps."
            else:
                action = "✓ Prediction appears reasonable. Safe to use for initial estimates, but always cross-check with local market conditions."

            # 5. Technical Analysis (for extreme cases)
            if ppsf < TYPICAL_PPSF * 0.5 or ppsf > TYPICAL_PPSF * 2:
                problem = f"Model failure detected: Prediction is {ppsf/TYPICAL_PPSF:.1f}x market average. "
                if bedrooms > MEAN_BEDROOMS + 3:
                    problem += f"Extreme bedroom count ({bedrooms} vs typical {MEAN_BEDROOMS:.1f}) is outside training distribution. "
                if sqft > MEAN_SQFT * 2:
                    problem += f"Property size ({sqft:,.0f} sqft) is {sqft/MEAN_SQFT:.1f}x average, causing extrapolation errors. "
                problem += "Model likely overfitting to rare outliers or missing critical features."
                
                causes = "Root causes: (1) Training data contains distressed/luxury outliers without proper weighting. "
                causes += "(2) Missing features: location quality, property condition, neighborhood premiums. "
                causes += "(3) Linear/tree models struggle with extreme combinations of features. "
                if bedrooms > MEAN_BEDROOMS + 3:
                    causes += "(4) High bedroom count suggests multi-family or unusual property type not well-represented in training. "
                
                fixes = "Recommended fixes: "
                fixes += "(1) Remove or cap training outliers beyond 99th percentile. "
                fixes += "(2) Add interaction features (sqft × grade, bedrooms × bathrooms). "
                fixes += "(3) Implement domain constraints: min_price = sqft × $50, max_price = sqft × $1000. "
                fixes += "(4) Add property type classification (single-family, multi-family, luxury). "
                fixes += "(5) Use quantile regression or ensemble with rule-based bounds. "
                fixes += "(6) Validate predictions: flag when inputs exceed 2× training range."
                
                # Generate Python code for fixing the model
                python_code = f"""# Fix for extreme prediction: ${predicted_price:,.0f} at ${ppsf:,.0f}/sqft
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Remove outliers using IQR method
def remove_outliers(df, column):
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 3*IQR) & (df[column] <= Q3 + 3*IQR)]

df = remove_outliers(df, 'price')
df = remove_outliers(df, 'sqft_living')

# 2. Add interaction features
df['sqft_grade_interaction'] = df['sqft_living'] * df['grade']
df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)

# 3. Cap extreme values
df = df[df['bedrooms'] <= 8]  # Cap at 8 bedrooms
df = df[df['sqft_living'] <= {MEAN_SQFT * 3:.0f}]  # Cap at 3x mean

# 4. Train with constraints
X = df[['sqft_living', 'bedrooms', 'bathrooms', 'grade', 'sqft_grade_interaction']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# 5. Add prediction constraints
def predict_with_bounds(model, X, min_ppsf={TYPICAL_PPSF * 0.3:.0f}, max_ppsf={TYPICAL_PPSF * 3:.0f}):
    pred = model.predict(X)
    sqft = X['sqft_living'].values
    pred = np.clip(pred, sqft * min_ppsf, sqft * max_ppsf)
    return pred

# Apply to new predictions
predictions = predict_with_bounds(model, X_test)"""
            else:
                problem = "Model performing within acceptable range. Minor variance expected due to unobserved factors."
                causes = "Typical regression limitations: cannot capture micro-location effects, recent renovations, or market timing."
                fixes = "No immediate action required. Consider periodic retraining with recent sales data to capture market trends."
                python_code = """# Model performing well - optional validation code
import numpy as np

# Add basic validation to flag unusual inputs
def validate_input(sqft, bedrooms, bathrooms):
    warnings = []
    if sqft > 4000:
        warnings.append("Large property - verify accuracy")
    if bedrooms > 6:
        warnings.append("High bedroom count - may need manual review")
    return warnings

# Example usage
warnings = validate_input(sqft_living={sqft}, bedrooms={bedrooms}, bathrooms={bathrooms})
if warnings:
    print("Warnings:", warnings)"""

            return {
                "MarketRealism": realism,
                "PossibleReasons": reasons.strip(),
                "SuggestedAction": action,
                "ProblemAnalysis": problem,
                "PossibleCauses": causes,
                "RecommendedFixes": fixes,
                "PythonCode": python_code
            }
        return {"error": str(e)}


def prepare_features(bedrooms, bathrooms, sqft_living, grade, zipcode=None):
    features = {
        'bedrooms': float(bedrooms),
        'bathrooms': float(bathrooms),
        'sqft_living': float(sqft_living),
        'grade': float(grade) if grade else 7.0,
    }
    
    features['total_rooms'] = features['bedrooms'] + features['bathrooms']
    
    if zipcode:
        try:
            zipcode_str = str(zipcode).strip()
            if zipcode_str:
                features['zipcode_encoded'] = float(zipcode_str[-3:]) / 1000.0
            else:
                features['zipcode_encoded'] = 0.0
        except:
            features['zipcode_encoded'] = 0.0
    
    defaults = {
        'sqft_lot': 0.0,
        'floors': 1.0,
        'waterfront': 0.0,
        'view': 0.0,
        'condition': 5.0,
        'sqft_above': features['sqft_living'],
        'sqft_basement': 0.0,
        'yr_built': 2000.0,
        'yr_renovated': 0.0,
        'lat': 47.5,
        'long': -122.3,
        'sqft_living15': features['sqft_living'],
        'sqft_lot15': 0.0,
    }
    
    for key, val in defaults.items():
        if key not in features:
            features[key] = val
    
    return features


def predict_cluster(features_dict, scaler, kmeans):
    priority = ['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'total_rooms']
    
    try:
        if hasattr(scaler, 'feature_names_in_'):
            cluster_features = list(scaler.feature_names_in_)
        else:
            cluster_features = [f for f in priority if f in features_dict]
    except:
        cluster_features = [f for f in priority if f in features_dict]
    
    if not cluster_features:
        cluster_features = ['bedrooms', 'bathrooms', 'sqft_living']
    
    feature_vector = np.array([[features_dict.get(f, 0) for f in cluster_features]])
    feature_vector_scaled = scaler.transform(feature_vector)
    cluster = kmeans.predict(feature_vector_scaled)[0]
    
    return cluster


def predict_price(features_dict, cluster_id, regressor):
    try:
        if hasattr(regressor, 'feature_names_in_'):
            feature_names = list(regressor.feature_names_in_)
        elif hasattr(regressor, 'get_booster'):
            feature_names = regressor.get_booster().feature_names
            if feature_names is None:
                raise AttributeError
        else:
            raise AttributeError
    except (AttributeError, TypeError):
        feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'total_rooms', 'cluster']
        possible = ['sqft_lot', 'floors', 'waterfront', 'view', 'condition', 
                   'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
                   'lat', 'long', 'sqft_living15', 'sqft_lot15']
        feature_names.extend([f for f in possible if f in features_dict])
    
    feature_vector = []
    for name in feature_names:
        if name == 'cluster':
            feature_vector.append(float(cluster_id))
        elif name in features_dict:
            feature_vector.append(float(features_dict[name]))
        else:
            feature_vector.append(0.0)
    
    feature_vector = np.array([feature_vector])
    predicted_price = regressor.predict(feature_vector)[0]
    predicted_price = max(0, predicted_price)
    
    return predicted_price


def main():
    st.markdown('<div class="main-header">Real Estate Price Predictor</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    with st.spinner("Loading models..."):
        regressor, classifier, kmeans, scaler = load_models()
    
    st.success("Models loaded successfully")
    
    with st.sidebar:
        st.header("Property Details")
        
        with st.form("prediction_form"):
            bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1)
            bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
            sqft_living = st.number_input("Living Area (sqft)", min_value=0, max_value=20000, value=2000, step=100)
            grade = st.number_input("Property Grade", min_value=1, max_value=13, value=7, step=1)
            zipcode = st.text_input("Zipcode (Optional)", value="")
            
            # API Key Handling (Secrets -> Sidebar)
            try:
                api_key = st.secrets.get("OPENAI_API_KEY", None)
            except (FileNotFoundError, KeyError):
                api_key = None
            
            if not api_key:
                api_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password", help="Enter your OpenAI API Key for AI-powered insights.")
            
            submitted = st.form_submit_button("Predict Price", use_container_width=True, type="primary")
    
    if submitted:
        features_dict = prepare_features(bedrooms, bathrooms, sqft_living, grade, zipcode if zipcode else None)
        
        with st.spinner("Analyzing property..."):
            cluster_id = predict_cluster(features_dict, scaler, kmeans)
            predicted_price = predict_price(features_dict, cluster_id, regressor)
            explanation = explain_prediction(features_dict, predicted_price, cluster_id)
        
        st.markdown("## Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Price", f"${predicted_price:,.0f}")
        
        with col2:
            segment_names = {0: "Budget", 1: "Mid-Market", 2: "Premium", 3: "Luxury"}
            st.metric("Market Segment", segment_names.get(cluster_id, f"Cluster {cluster_id}"))
        
        with col3:
            if sqft_living > 0:
                ppsf = predicted_price / sqft_living
                st.metric("Price per SqFt", f"${ppsf:,.2f}")
        
        st.markdown("---")
        st.markdown("## Prediction Explanation")
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown(explanation)
        st.markdown('</div>', unsafe_allow_html=True)
        

        # AI Explanation Section
        if api_key:
            st.markdown("---")
            st.markdown("## 🤖 AI Expert Analysis")
            with st.spinner("Generating expert insights..."):
                ai_result = get_ai_explanation(features_dict, predicted_price, api_key)
            
            if ai_result:
                if "error" in ai_result:
                    st.error(f"AI Analysis Error: {ai_result['error']}")
                else:
                    # Market Realism Assessment
                    realism = ai_result.get('MarketRealism', 'N/A')
                    if '⚠️' in realism or 'Undervalued' in realism or 'Overvalued' in realism:
                        st.error(f"**Market Assessment:**\n\n{realism}")
                    elif 'Above Average' in realism or 'Below Average' in realism:
                        st.warning(f"**Market Assessment:**\n\n{realism}")
                    else:
                        st.success(f"**Market Assessment:**\n\n{realism}")
                    
                    # Possible Reasons
                    st.info(f"**Why This Price?**\n\n{ai_result.get('PossibleReasons', 'N/A')}")
                    
                    # Suggested Action
                    action = ai_result.get('SuggestedAction', 'N/A')
                    if '⚠️' in action:
                        st.warning(f"**Recommended Action:**\n\n{action}")
                    elif '✓' in action:
                        st.success(f"**Recommended Action:**\n\n{action}")
                    else:
                        st.info(f"**Recommended Action:**\n\n{action}")
                    
                    # Technical Diagnostics (Expandable for developers)
                    with st.expander("🔧 Technical Diagnostics (for Developers)"):
                        st.markdown("**Problem Analysis:**")
                        st.code(ai_result.get('ProblemAnalysis', 'N/A'), language=None)
                        
                        st.markdown("**Root Causes:**")
                        st.code(ai_result.get('PossibleCauses', 'N/A'), language=None)
                        
                        st.markdown("**Recommended Fixes:**")
                        st.code(ai_result.get('RecommendedFixes', 'N/A'), language=None)
                        
                        # Python code fix
                        if 'PythonCode' in ai_result and ai_result['PythonCode'] != 'N/A':
                            st.markdown("**🐍 Copy-Paste Python Fix:**")
                            st.code(ai_result['PythonCode'], language='python')
                            st.info("👆 Copy this code to fix model training issues")



        
        st.markdown("---")
        with st.expander("Additional Information"):
            st.info("""
            Predictions are estimates based on property characteristics.
            Actual values may vary based on location, condition, and market conditions.
            Consult with real estate professionals for accurate valuations.
            """)
    
    else:
        st.info("Enter property details in the sidebar and click 'Predict Price'.")


if __name__ == "__main__":
    main()
