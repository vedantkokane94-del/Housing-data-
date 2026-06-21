# Housing-data-
"""
HOUSING PRICE PREDICTION - COMPLETE INDIVIDUAL PROJECT
Data Loading & Cleaning + Visualization + Regression Model
Perfect for Computer Engineering Student Portfolio
"""

# =============================================================================
# INSTALL REQUIRED LIBRARIES (Run in terminal first if needed)
# =============================================================================
# pip install pandas numpy matplotlib seaborn scikit-learn plotly joblib

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import joblib
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn')
sns.set_palette("husl")

print("=" * 80)
print("HOUSING PRICE PREDICTION - COMPLETE PROJECT")
print("=" * 80)

# =============================================================================
# STEP 1: DATA LOADING & CLEANING (Pandas)
# =============================================================================

print("
" + "=" * 80)
print("STEP 1: DATA LOADING & CLEANING")
print("=" * 80)

# OPTION 1: Load from CSV (replace with your real dataset path)
# df = pd.read_csv('housing_data.csv')

# OPTION 2: Create sample dataset (for demonstration)
np.random.seed(42)
n_samples = 100

data = {
    'house_id': range(1, n_samples + 1),
    'area_sqft': np.random.randint(500, 5000, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'parking': np.random.randint(0, 4, n_samples),
    'age_years': np.random.randint(0, 50, n_samples),
    'location_score': np.random.uniform(1, 10, n_samples),
    'near_school': np.random.choice([0, 1], n_samples),
    'near_market': np.random.choice([0, 1], n_samples),
    'facing': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
}

# Calculate realistic prices based on features
prices = []
for i in range(n_samples):
    base_price = 5
    area_factor = data['area_sqft'][i] * 0.005
    bedroom_factor = data['bedrooms'][i] * 2
    bathroom_factor = data['bathrooms'][i] * 1.5
    parking_factor = data['parking'][i] * 1
    age_factor = -data['age_years'][i] * 0.05
    location_factor = data['location_score'][i] * 3
    school_factor = data['near_school'][i] * 2
    market_factor = data['near_market'][i] * 1.5
    
    price = base_price + area_factor + bedroom_factor + bathroom_factor + parking_factor + age_factor + location_factor + school_factor + market_factor
    price = price + np.random.normal(0, 3)  # Add noise
    prices.append(round(price, 2))

data['price_lakhs'] = prices
df = pd.DataFrame(data)

print(f"
✓ Dataset loaded successfully!")
print(f"  Total rows: {len(df)}")
print(f"  Total columns: {len(df.columns)}")

print("
📊 First 5 rows:")
print(df.head())

print("
🔍 Missing values:")
missing = df.isnull().sum()
print(missing if missing.sum() > 0 else "  No missing values!")

print("
📈 Basic statistics:")
print(df.describe())

# =============================================================================
# DATA CLEANING
# =============================================================================

print("
" + "-" * 80)
print("DATA CLEANING")
print("-" * 80)

df = df.drop_duplicates()
print(f"✓ Removed duplicates: {len(df)} rows")

df = df.dropna()
print(f"✓ Removed missing values: {len(df)} rows")

# Convert categorical 'facing' to numerical
facing_map = {'North': 1, 'South': 2, 'East': 3, 'West': 4}
df['facing_code'] = df['facing'].map(facing_map)
print("✓ Converted 'facing' to numerical code")

# Remove outliers (prices > 3 standard deviations)
price_mean = df['price_lakhs'].mean()
price_std = df['price_lakhs'].std()
df = df[(df['price_lakhs'] >= price_mean - 3*price_std) & (df['price_lakhs'] <= price_mean + 3*price_std)]
print(f"✓ Removed outliers: {len(df)} rows")

print(f"
✓ CLEANING COMPLETE: {len(df)} rows, {len(df.columns)} columns")

# =============================================================================
# STEP 2: VISUALIZATION (Matplotlib)
# =============================================================================

print("
" + "=" * 80)
print("STEP 2: VISUALIZATION")
print("=" * 80)

fig = plt.figure(figsize=(20, 16))

# Plot 1: Price Distribution
ax1 = fig.add_subplot(3, 3, 1)
sns.histplot(df['price_lakhs'], ax=ax1, kde=True, color='steelblue')
ax1.set_title('Price Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('Price (Lakhs)')

# Plot 2: Price vs Area
ax2 = fig.add_subplot(3, 3, 2)
sns.scatterplot(x=df['area_sqft'], y=df['price_lakhs'], ax=ax2, color='coral', alpha=0.6)
ax2.set_title('Price vs Area', fontsize=12, fontweight='bold')
corr = df['area_sqft'].corr(df['price_lakhs'])
ax2.text(0.05, 0.95, f'Corr: {corr:.2f}', transform=ax2.transAxes, fontweight='bold')

# Plot 3: Price vs Bedrooms
ax3 = fig.add_subplot(3, 3, 3)
sns.barplot(x=df['bedrooms'], y=df['price_lakhs'], ax=ax3, palette='viridis')
ax3.set_title('Price vs Bedrooms', fontsize=12, fontweight='bold')

# Plot 4: Price vs Location
ax4 = fig.add_subplot(3, 3, 4)
sns.scatterplot(x=df['location_score'], y=df['price_lakhs'], ax=ax4, color='green', alpha=0.6)
ax4.set_title('Price vs Location Score', fontsize=12, fontweight='bold')

# Plot 5: Correlation Heatmap
ax5 = fig.add_subplot(3, 3, 5)
num_cols = df[['area_sqft', 'bedrooms', 'bathrooms', 'parking', 'age_years', 'location_score', 'price_lakhs']]
sns.heatmap(num_cols.corr(), ax=ax5, annot=True, cmap='coolwarm', fmt='.2f')
ax5.set_title('Correlation Heatmap', fontsize=12, fontweight='bold')

# Plot 6: Price by Facing
ax6 = fig.add_subplot(3, 3, 6)
facing_price = df.groupby('facing')['price_lakhs'].mean().sort_values()
sns.barplot(x=facing_price.index, y=facing_price.values, ax=ax6, palette='plasma')
ax6.set_title('Price by Facing', fontsize=12, fontweight='bold')

# Plot 7: Price vs Age
ax7 = fig.add_subplot(3, 3, 7)
sns.scatterplot(x=df['age_years'], y=df['price_lakhs'], ax=ax7, color='purple', alpha=0.6)
ax7.set_title('Price vs Age', fontsize=12, fontweight='bold')

# Plot 8: Price vs Bathrooms
ax8 = fig.add_subplot(3, 3, 8)
sns.barplot(x=df['bathrooms'], y=df['price_lakhs'], ax=ax8, palette='magma')
ax8.set_title('Price vs Bathrooms', fontsize=12, fontweight='bold')

# Plot 9: Price vs Parking
ax9 = fig.add_subplot(3, 3, 9)
sns.barplot(x=df['parking'], y=df['price_lakhs'], ax=ax9, palette='cividis')
ax9.set_title('Price vs Parking', fontsize=12, fontweight='bold')

plt.suptitle('HOUSING DATA VISUALIZATION', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('housing_visualizations.png', dpi=300, bbox_inches='tight')
print("
✓ Saved: housing_visualizations.png")

# =============================================================================
# STEP 3: REGRESSION MODEL
# =============================================================================

print("
" + "=" * 80)
print("STEP 3: REGRESSION MODEL")
print("=" * 80)

# Select features
features = ['area_sqft', 'bedrooms', 'bathrooms', 'parking', 'age_years', 'location_score', 'near_school', 'near_market', 'facing_code']
X = df[features]
y = df['price_lakhs']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"
✓ Data split: {len(X_train)} train, {len(X_test)} test")

# MODEL 1: Linear Regression
print("
" + "-" * 80)
print("Linear Regression")
print("-" * 80)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

linear_r2 = r2_score(y_test, y_pred_linear)
linear_rmse = np.sqrt(mean_squared_error(y_test, y_pred_linear))

print(f"R2 Score: {linear_r2:.4f}")
print(f"RMSE: {linear_rmse:.4f}")

print("
Coefficients:")
for f, c in zip(features, linear_model.coef_):
    print(f"  {f}: {c:.4f}")

# MODEL 2: Random Forest
print("
" + "-" * 80)
print("Random Forest Regression")
print("-" * 80)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"R2 Score: {rf_r2:.4f}")
print(f"RMSE: {rf_rmse:.4f}")

feature_importance = pd.DataFrame({'Feature': features, 'Importance': rf_model.feature_importances_}).sort_values('Importance', ascending=False)
print("
Feature Importance:")
for f, imp in zip(feature_importance['Feature'], feature_importance['Importance']):
    print(f"  {f}: {imp:.4f}")

# MODEL COMPARISON
print("
" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'R2': [linear_r2, rf_r2],
    'RMSE': [linear_rmse, rf_rmse]
})
print(comparison.to_string(index=False))

best_model = 'Random Forest' if rf_r2 > linear_r2 else 'Linear Regression'
print(f"
🏆 Best: {best_model} (R2 = {max(linear_r2, rf_r2):.4f})")

# PERFORMANCE VISUALIZATION
fig_perf = plt.figure(figsize=(15, 5))

ax1 = fig_perf.add_subplot(1, 3, 1)
ax1.scatter(y_test, y_pred_linear, alpha=0.6)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
ax1.set_title('Linear Regression', fontweight='bold')
ax1.text(0.05, 0.95, f'R2={linear_r2:.3f}', transform=ax1.transAxes, fontweight='bold', verticalalignment='top')

ax2 = fig_perf.add_subplot(1, 3, 2)
ax2.scatter(y_test, y_pred_rf, alpha=0.6, color='green')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
ax2.set_title('Random Forest', fontweight='bold')
ax2.text(0.05, 0.95, f'R2={rf_r2:.3f}', transform=ax2.transAxes, fontweight='bold', verticalalignment='top')

ax3 = fig_perf.add_subplot(1, 3, 3)
ax3.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
ax3.set_title('Feature Importance', fontweight='bold')
ax3.invert_yaxis()

plt.suptitle('MODEL PERFORMANCE', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("
✓ Saved: model_comparison.png")

# PREDICTION EXAMPLE
print("
" + "=" * 80)
print("PREDICTION EXAMPLE")
print("=" * 80)

new_house = pd.DataFrame({
    'area_sqft': [2000], 'bedrooms': [3], 'bathrooms': [2], 'parking': [2],
    'age_years': [10], 'location_score': [8.5], 'near_school': [1], 'near_market': [1], 'facing_code': [2]
})

pred_rf = rf_model.predict(new_house)[0]
pred_linear = linear_model.predict(new_house)[0]

print(f"
House: 2000 sqft, 3BR, 2BA, Parking:2, Age:10yr, Location:8.5")
print(f"Predicted Price:")
print(f"  Linear: {pred_linear:.2f} Lakhs")
print(f"  Random Forest: {pred_rf:.2f} Lakhs ⭐")

# SAVE MODELS
joblib.dump(linear_model, 'linear_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(feature_importance, 'feature_importance.pkl')

print("
✓ Saved models: linear_model.pkl, random_forest_model.pkl, feature_importance.pkl")

# FINAL SUMMARY
print("
" + "=" * 80)
print("✨ PROJECT COMPLETE!")
print("=" * 80)
print(f"
📊 Dataset: {len(df)} samples, {len(features)} features")
print(f"🏆 Best Model: {best_model} (R2 = {max(linear_r2, rf_r2):.4f})")
print(f"
📁 Files created:")
print(f"  • housing_visualizations.png (9 plots)")
print(f"  • model_comparison.png (performance)")
print(f"  • linear_model.pkl")
print(f"  • random_forest_model.pkl")
print(f"  • feature_importance.pkl")

print("
🎯 Next steps:")
print("  1. Upload to GitHub with README.md")
print("  2. Write project report (Abstract, Methodology, Results)")
print("  3. Create PowerPoint presentation")
print("  4. Share on LinkedIn")

print("
" + "=" * 80)
print("🚀 Ready for your portfolio!")
print("=" * 80)
