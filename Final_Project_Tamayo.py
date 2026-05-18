import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- STREAMLIT CONFIGURATION ---
st.set_page_config(
    page_title="Presyo-Patrol Dashboard",
    page_icon="🍲",
    layout="wide"
)

# Set seaborn theme style natively
sns.set_theme(style="whitegrid")

# --- DATA GENERATOR / LOADER ---
@st.cache_data
def load_data():
    file_path = "/content/wfp_food_prices_phl.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    # Structural Fallback Mock Data Generator
    np.random.seed(42)
    dates = pd.date_range(start="2018-01-01", end="2021-12-01", freq="MS")
    regions = ["National Capital Region", "Region III", "Region IV-A", "Region VII", "Region XI"]
    provinces = ["Metro Manila", "Pampanga", "Cavite", "Cebu", "Davao del Sur"]
    categories = ["cereals and tubers", "meat, fish and eggs", "vegetables and fruits", "pulses and nuts"]
    
    commodities = {
        "cereals and tubers": ["Rice (regular, milled)", "Rice (well-milled)"],
        "meat, fish and eggs": ["Meat (pork)", "Fish (fresh)"],
        "vegetables and fruits": ["Onions (red)", "Tomatoes"],
        "pulses and nuts": ["Beans (mung)"]
    }
    
    data_rows = []
    for date in dates:
        for r_idx, region in enumerate(regions):
            province = provinces[r_idx]
            for category in categories:
                for commodity in commodities[category]:
                    base_price = 45.0 if "Rice" in commodity else np.random.uniform(30, 250)
                    trend_inflation = (date.year - 2018) * 8.5
                    final_price = max(10.0, base_price + trend_inflation + np.random.normal(0, 5))
                    
                    data_rows.append({
                        "date": date, "admin1": region, "admin2": province,
                        "market": f"{province} Central Market", "category": category,
                        "commodity": commodity, "unit": "KG", "pricetype": "Retail",
                        "price": round(final_price, 2)
                    })
    return pd.DataFrame(data_rows)

df = load_data()

# --- PREPROCESSING & MODELING (Cached for safety) ---
@st.cache_resource
def run_model_pipeline(dataframe):
    df_proc = dataframe.copy()
    df_proc['date'] = pd.to_datetime(df_proc['date'])
    df_proc['year'] = df_proc['date'].dt.year
    df_proc['month'] = df_proc['date'].dt.month
    df_proc['day_of_week'] = df_proc['date'].dt.dayofweek

    df_proc = df_proc.sort_values(by=['commodity', 'market', 'date'])
    df_proc['price_lag_1'] = df_proc.groupby(['commodity', 'market'])['price'].shift(1).fillna(df_proc['price'])
    df_proc['price_log'] = np.log1p(df_proc['price'])

    le = LabelEncoder()
    categorical_cols = ['admin1', 'admin2', 'market', 'category', 'commodity', 'unit', 'pricetype']
    for col in categorical_cols:
        df_proc[col+'_encoded'] = le.fit_transform(df_proc[col].astype(str))

    scaler = StandardScaler()
    features_to_scale = ['year', 'month', 'day_of_week', 'price_lag_1']
    df_proc[features_to_scale] = scaler.fit_transform(df_proc[features_to_scale])

    train_size = int(len(df_proc) * 0.8)
    train_df = df_proc.iloc[:train_size]
    test_df = df_proc.iloc[train_size:]

    features = ['year', 'month', 'price_lag_1', 'admin1_encoded', 'commodity_encoded', 'pricetype_encoded']
    target = 'price'

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    lr_model = LinearRegression().fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    return df_proc, test_df, y_test, lr_preds, rf_preds, rf_model, features

df_proc, test_df, y_test, lr_preds, rf_preds, rf_model, features = run_model_pipeline(df)

# --- USER INTERFACE ---
st.title("🍲 Presyo-Patrol Dashboard")
st.subheader("Localized Food Price Prediction and Budget Optimization")

tabs = st.tabs(["📊 Exploratory Data Analysis", "🤖 Model Performance Evaluation", "📈 Structural Market Trends"])

# TAB 1: EXPLORATORY DATA ANALYSIS
with tabs[0]:
    st.header("Exploratory Data Analysis Plots")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Distribution of Food Prices")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        # Updated to use direct ax assignment for Streamlit stability
        sns.histplot(df['price'], bins=50, kde=True, color='blue', ax=ax1)
        ax1.set_xlabel('Price (PHP)')
        ax1.set_ylabel('Frequency')
        st.pyplot(fig1) # Replaced plt.show()
        
    with col2:
        st.write("#### Boxplot of Prices (Outlier Detection)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=df['price'], color='salmon', ax=ax2)
        st.pyplot(fig2) # Replaced plt.show()

    st.write("#### Food Price Range by Category")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x='category', y='price', palette='Set2', ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3) # Replaced plt.show()

# TAB 2: MODEL PERFORMANCE EVALUATION
with tabs[1]:
    st.header("Evaluation & Validation Visuals")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("#### Random Forest: Predicted vs Actual")
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.scatter(y_test, rf_preds, alpha=0.3, color='green', s=2)
        ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax4.set_xlabel('Actual Prices')
        ax4.set_ylabel('Predicted Price')
        st.pyplot(fig4) # Replaced plt.show()
        
    with col4:
        st.write("#### Random Forest: Residual Plot")
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        residuals = y_test - rf_preds
        ax5.scatter(rf_preds, residuals, alpha=0.3, color='blue', s=2)
        ax5.axhline(0, color='red', linestyle='--')
        ax5.set_xlabel('Predicted Price')
        ax5.set_ylabel('Residuals')
        st.pyplot(fig5) # Replaced plt.show()

    st.write("#### Feature Importance Weights")
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    feat_importances = pd.Series(rf_model.feature_importances_, index=features)
    feat_importances.sort_values().plot(kind='barh', color='darkorange', ax=ax6)
    ax6.set_xlabel('Importance Score')
    st.pyplot(fig6) # Replaced plt.show()

# TAB 3: STRUCTURAL MARKET TRENDS
with tabs[2]:
    st.header("Market Integration Heatmaps & Aggregations")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.write("#### Correlation Heatmap")
        fig7, ax7 = plt.subplots(figsize=(10, 8))
        corr_features = ['year', 'month', 'price', 'price_lag_1', 'admin1_encoded', 'category_encoded', 'commodity_encoded']
        corr_matrix = df_proc[corr_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax7)
        st.pyplot(fig7) # Replaced plt.show()
        
    with col6:
        st.write("#### Regional Price Tracking Engine")
        fig8, ax8 = plt.subplots(figsize=(12, 6))
        rice_df = df[df['commodity'].str.contains('Rice', case=False, na=False)]
        sns.lineplot(data=rice_df, x='date', y='price', hue='admin1', ax=ax8)
        plt.xticks(rotation=45)
        st.pyplot(fig8) # Replaced plt.show()
