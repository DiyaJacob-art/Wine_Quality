# Import libraries
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit UI Setup
st.set_page_config(page_title="Wine Quality Prediction", layout="wide")
st.title("🍷 Wine Quality Prediction - Regression Models")
st.markdown("Comparing 6 regression algorithms on WineQT dataset")

# Load data
df = pd.read_csv('WineQT.csv')
st.write(f"**Dataset Shape:** {df.shape[0]} samples, {df.shape[1]} features")

# Show dataset preview
with st.expander("View Dataset"):
    st.dataframe(df.head())

# Separate features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write(f"**Training samples:** {X_train.shape[0]}, **Test samples:** {X_test.shape[0]}")

def evaluate(y_test, y_pred, model_name):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    # Display in Streamlit
    st.subheader(f"📊 {model_name}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MAE", f"{mae:.3f}")
    with col2:
        st.metric("MSE", f"{mse:.3f}")
    with col3:
        st.metric("RMSE", f"{rmse:.3f}")
    with col4:
        st.metric("R² Score", f"{r2:.3f}")
    
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

# Run models
st.header("🤖 Model Performance Comparison")
st.write("Training 6 regression models...")

# Store results for comparison
results = {}

# Linear Regression
with st.spinner("Training Linear Regression..."):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results["Linear Regression"] = evaluate(y_test, lr_pred, "Linear Regression")

# Decision Tree
with st.spinner("Training Decision Tree..."):
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    results["Decision Tree"] = evaluate(y_test, dt_pred, "Decision Tree Regressor")

# Random Forest
with st.spinner("Training Random Forest..."):
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["Random Forest"] = evaluate(y_test, rf_pred, "Random Forest Regressor")

# Support Vector Regressor
with st.spinner("Training SVR..."):
    svr = SVR()
    svr.fit(X_train, y_train)
    svr_pred = svr.predict(X_test)
    results["SVR"] = evaluate(y_test, svr_pred, "Support Vector Regressor")

# Lasso Regression
with st.spinner("Training Lasso Regression..."):
    lasso = Lasso()
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    results["Lasso"] = evaluate(y_test, lasso_pred, "Lasso Regression")

# Ridge Regression
with st.spinner("Training Ridge Regression..."):
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    results["Ridge"] = evaluate(y_test, ridge_pred, "Ridge Regression")

# Comparison Table
st.header("🏆 Model Comparison Summary")
comparison_df = pd.DataFrame(results).T
st.dataframe(comparison_df.style.highlight_max(subset=['R2'], color='lightgreen')
                             .highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='lightcoral'))

# Best model
best_model = max(results, key=lambda x: results[x]['R2'])
best_r2 = results[best_model]['R2']
st.success(f"✅ **Best Model:** {best_model} with R² = {best_r2:.3f}")

# Feature importance (for tree-based models)
st.header("🔍 Feature Importance")
if st.checkbox("Show Random Forest Feature Importance"):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    st.bar_chart(feature_importance.set_index('feature'))

# Prediction demo
st.header("🔮 Try Prediction")
st.write("Enter feature values to predict wine quality:")

# Create input sliders for top 5 features
if 'feature_importance' in locals():
    top_features = feature_importance.head(5)['feature'].tolist()
else:
    top_features = X.columns[:5].tolist()

user_input = {}
cols = st.columns(3)
for i, feature in enumerate(top_features):
    with cols[i % 3]:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        default_val = float(df[feature].mean())
        user_input[feature] = st.slider(
            feature, 
            min_value=min_val, 
            max_value=max_val, 
            value=default_val,
            step=0.1
        )

# Fill remaining features with mean values
for feature in X.columns:
    if feature not in user_input:
        user_input[feature] = float(df[feature].mean())

if st.button("Predict Quality", type="primary"):
    # Prepare input
    input_df = pd.DataFrame([user_input])
    
    # Use best model for prediction
    if best_model == "Linear Regression":
        prediction = lr.predict(input_df)[0]
    elif best_model == "Decision Tree":
        prediction = dt.predict(input_df)[0]
    elif best_model == "Random Forest":
        prediction = rf.predict(input_df)[0]
    elif best_model == "SVR":
        prediction = svr.predict(input_df)[0]
    elif best_model == "Lasso":
        prediction = lasso.predict(input_df)[0]
    else:
        prediction = ridge.predict(input_df)[0]
    
    st.metric("Predicted Wine Quality", f"{prediction:.1f}/10")
    st.write(f"Using model: **{best_model}**")

st.markdown("---")
st.caption("Wine Quality Prediction App | Deployed on Render")
