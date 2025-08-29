import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =========================
# Load CSV with Caching
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", nrows=5000)  # Load only first 5k rows
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df['year'] = df["pickup_datetime"].dt.year
    df['month'] = df["pickup_datetime"].dt.month
    df['day'] = df["pickup_datetime"].dt.day
    df['hour'] = df['pickup_datetime'].dt.hour
    df['minute'] = df["pickup_datetime"].dt.minute
    df['dayname'] = df['pickup_datetime'].dt.day_name()
    
    # Remove only rows with invalid coordinates
    df = df[(df['pickup_latitude'] != 0) & (df['pickup_longitude'] != 0) &
            (df['dropoff_latitude'] != 0) & (df['dropoff_longitude'] != 0)]
    
    return df

df = load_data()

st.title("Taxi Ride Distance Prediction")
st.subheader("Using ML to predict the distance of a taxi ride based on various parameters")
st.subheader("Dataset: https://www.kaggle.com/datasets/dansbecker/new-york-city-taxi-fare-prediction")
st.markdown("---")

st.subheader("Dataset Preview")
st.dataframe(df.head(20))

st.subheader("Statistics of the data")
st.write(df.describe())

# =========================
# Haversine Function
# =========================
def haversine(lat1,long1,lat2,long2):
    R = 6371  # radius of the earth in km
    difference_lat = np.radians(lat2 - lat1)
    difference_long = np.radians(long2 - long1)
    alpha = (np.sin(difference_lat / 2) ** 2 +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
             np.sin(difference_long / 2) ** 2)
    central_angle = 2 * np.arcsin(np.sqrt(alpha))
    dist = R * central_angle
    return dist

# Actual distance in km
df['distance_km'] = haversine(df['pickup_latitude'], df['pickup_longitude'],
                              df['dropoff_latitude'], df['dropoff_longitude'])

# Save unscaled distance for plotting
df['distance_km_original'] = df['distance_km']

# Scale distance for ML models if needed
sc_distance = StandardScaler()
df['distance_km'] = sc_distance.fit_transform(df['distance_km'].values.reshape(-1,1))
# scaling fare amount
df['fare_amount_original'] = df['fare_amount']
sc_fare=StandardScaler()
df['fare_amount'] = sc_fare.fit_transform(df['fare_amount'].values.reshape(-1,1))
# Drop unnecessary columns
df.drop(columns=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude','key','pickup_datetime'], inplace=True)

# Encode dayname
day_name_encoder = LabelEncoder()
df['dayname_Labelled'] = day_name_encoder.fit_transform(df['dayname'])
df.drop(columns=['dayname'], inplace=True)

# Features & Target
x = df[['passenger_count','minute','hour','day','month','year',"dayname_Labelled","fare_amount"]]
y = df[['distance_km']]
y_original = df[['distance_km_original']]  # for plotting

x_train, x_test, y_train, y_test, y_train_orig, y_test_orig = train_test_split(
    x, y, y_original, test_size=0.2, random_state=42
)
# Metrics function
def show_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    st.write(f"**R²:** {r2:.3f} — The model explains about {r2*100:.1f}% of the variance in the data")
    st.write(f"**MSE:** {mse:.3f} — Mean Squared Error of predictions")
    st.write(f"**MAE:** {mae:.3f} — Mean Absolute Error of predictions")
# =========================
# ML Models
def linear():
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    
    y_pred_original = sc_distance.inverse_transform(y_pred.reshape(-1,1))
    st.title("Distance Prediction using Linear Regression")
    st.write("Predicted Distance (km)", round(y_pred_original[0][0], 2))
    st.write("Actual Distance (km)", round(y_test_orig.iloc[0, 0], 2))

    st.subheader("Model Performance")
    show_metrics(y_test_orig, y_pred_original)

    fig,ax=plt.subplots()
    sorted_idx = y_test_orig.reset_index(drop=True).sort_values(by=y_test_orig.columns[0]).index
    y_test_sorted = y_test_orig.reset_index(drop=True).iloc[sorted_idx]
    y_pred_sorted = pd.DataFrame(y_pred_original).iloc[sorted_idx]

    ax.plot(y_test_sorted.values, y_pred_sorted.values, color='blue', label="Predicted Line")
    ax.plot([min(y_test_sorted.values), max(y_test_sorted.values)],
            [min(y_test_sorted.values), max(y_test_sorted.values)], 'r--', label="Ideal Line")
    ax.set_title("Distance Predictor (Linear Regression)")
    ax.set_xlabel("Actual Distance (km)")
    ax.set_ylabel("Predicted Distance (km)")
    ax.legend()
    st.pyplot(fig)

def polynomial():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    poly_reg = PolynomialFeatures(degree=2)  # safer degree
    x_poly = poly_reg.fit_transform(x_train)
    model = LinearRegression()
    model.fit(x_poly,y_train.values.ravel())
    y_pred = model.predict(poly_reg.transform(x_test))

    y_pred_original = sc_distance.inverse_transform(y_pred.reshape(-1,1))
    st.title("Distance Prediction using Polynomial Regression")
    st.write("Predicted Distance (km)", round(y_pred_original[0][0], 2))
    st.write("Actual Distance (km)", round(y_test_orig.iloc[0, 0], 2))

    st.subheader("Model Performance")
    show_metrics(y_test_orig, y_pred_original)

    fig, ax = plt.subplots()
    sorted_idx = y_test_orig.reset_index(drop=True).sort_values(by=y_test_orig.columns[0]).index
    y_test_sorted = y_test_orig.reset_index(drop=True).iloc[sorted_idx]
    y_pred_sorted = pd.DataFrame(y_pred_original).iloc[sorted_idx]

    ax.plot(y_test_sorted.values, y_pred_sorted.values, color='blue', label="Predicted Line") 
    ax.plot([min(y_test_sorted.values), max(y_test_sorted.values)],
            [min(y_test_sorted.values), max(y_test_sorted.values)],'r--')
    ax.set_title("Distance Predictor (Polynomial Regression)")
    ax.set_xlabel("Actual Distance (km)")
    ax.set_ylabel("Predicted Distance (km)")
    st.pyplot(fig)

def SupportVectorRegression():
    from sklearn.svm import SVR
    regressor = SVR(kernel='rbf')
    regressor.fit(x_train, y_train.values.ravel())
    y_pred = regressor.predict(x_test)
    y_pred_original = sc_distance.inverse_transform(y_pred.reshape(-1,1))
    st.title("Distance Prediction using Support Vector Regression")
    st.write("Predicted Distance (km)", round(y_pred_original[0][0], 2))
    st.write("Actual Distance (km)", round(y_test_orig.iloc[0, 0], 2))
    st.subheader("Model Performance")
    show_metrics(y_test_orig, y_pred_original)

    fig, ax = plt.subplots()

# Sort values so line looks smooth
    sorted_idx = y_test_orig.reset_index(drop=True).sort_values(by=y_test_orig.columns[0]).index
    y_test_sorted = y_test_orig.reset_index(drop=True).iloc[sorted_idx]
    y_pred_sorted = pd.DataFrame(y_pred_original).iloc[sorted_idx]

# Blue line = model predictions
    ax.plot(y_test_sorted.values, y_pred_sorted.values, color='blue', label="Predicted Line")

# Red dashed line = ideal predictions (perfect fit)
    ax.plot([min(y_test_orig.values), max(y_test_orig.values)],
        [min(y_test_orig.values), max(y_test_orig.values)],
        'r--', label="Ideal Line")

    ax.set_title("Distance Predictor (Polynomial Regression)")  
    ax.set_xlabel("Actual Distance (km)")                       
    ax.set_ylabel("Predicted Distance (km)")                   
    ax.legend()
    st.pyplot(fig)

def RandomForest():
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=100,random_state=42)
    regressor.fit(x_train, y_train.values.ravel())
    y_pred = regressor.predict(x_test)
    y_pred_original = sc_distance.inverse_transform(y_pred.reshape(-1,1))
    st.title("Distance Prediction using Random Forest")
    st.write("Predicted Distance (km)", round(y_pred_original[0][0], 2))
    st.write("Actual Distance (km)", round(y_test_orig.iloc[0, 0], 2))
    st.subheader("Model Performance")
    show_metrics(y_test_orig, y_pred_original)

    fig, ax = plt.subplots()

    # Sort values so line looks smooth
    sorted_idx = y_test_orig.reset_index(drop=True).sort_values(by=y_test_orig.columns[0]).index
    y_test_sorted = y_test_orig.reset_index(drop=True).iloc[sorted_idx]
    y_pred_sorted = pd.DataFrame(y_pred_original).iloc[sorted_idx]

    # Blue line = model predictions
    ax.plot(y_test_sorted.values, y_pred_sorted.values, color='blue', label="Predicted Line")

    # Red dashed line = ideal predictions (perfect fit)
    ax.plot([min(y_test_orig.values), max(y_test_orig.values)],
        [min(y_test_orig.values), max(y_test_orig.values)],
        'r--', label="Ideal Line")

    ax.set_title("Distance Predictor (Polynomial Regression)")  # <-- change title per model
    ax.set_xlabel("Actual Distance (km)")                       # <-- change if fare
    ax.set_ylabel("Predicted Distance (km)")                    # <-- change if fare
    ax.legend()
    st.pyplot(fig)

    # Feature importance
    st.subheader("Feature Importances (Random Forest)")
    importances = pd.Series(regressor.feature_importances_, index=x_train.columns).sort_values(ascending=False)
    st.bar_chart(importances)

    # =========================
st.markdown("---")
#Training for predicting the fare amount
a = df[['passenger_count','minute','hour','day','month','year',"dayname_Labelled","distance_km"]]
b = df[['fare_amount']]
b_original=df[["fare_amount_original"]]
a_train,a_test,b_train,b_test,b_train_org,b_test_org = train_test_split(a,b,b_original,test_size=0.2,random_state=42)

def linear1():
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(a_train, b_train)
    b_pred = regressor.predict(a_test)

    # Inverse transform back to original fare units
    b_pred_orig = sc_fare.inverse_transform(b_pred.reshape(-1,1))
    st.subheader("Fare Amount Prediction using Linear Regression")
    st.write("Predicted Fare ($)", round(b_pred_orig[0][0], 2))
    st.write("Actual Fare ($)", round(b_test_org.iloc[0, 0], 2))

    st.subheader("Model Performance")
    show_metrics(b_test_org, b_pred_orig)

    # --- Fixed Plot ---
    fig, ax = plt.subplots()
    sorted_idx = b_test_org.reset_index(drop=True).sort_values(by=b_test_org.columns[0]).index
    b_test_sorted = b_test_org.reset_index(drop=True).iloc[sorted_idx]
    b_pred_sorted = pd.DataFrame(b_pred_orig).iloc[sorted_idx]

    ax.plot(b_test_sorted.values, b_pred_sorted.values, color='blue', label="Predicted Line")
    ax.plot([min(b_test_org.values), max(b_test_org.values)],
            [min(b_test_org.values), max(b_test_org.values)], 'r--', label="Ideal Line")
    ax.set_title("Fare Amount Predictor (Linear Regression)")
    ax.set_xlabel("Actual Fare Amount ($)")
    ax.set_ylabel("Predicted Fare Amount ($)")
    ax.legend()
    st.pyplot(fig)


def polynomial1():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    poly_reg = PolynomialFeatures(degree=2)
    a_poly = poly_reg.fit_transform(a_train)
    model = LinearRegression()
    model.fit(a_poly, b_train.values.ravel())
    b_pred = model.predict(poly_reg.transform(a_test))

    b_pred_original = sc_fare.inverse_transform(b_pred.reshape(-1,1))
    st.title("Fare Amount Prediction using Polynomial Regression")
    st.metric("Predicted Fare ($)", round(b_pred_original[0][0], 2))
    st.write("Actual Fare ($)", round(b_test_org.iloc[0, 0], 2))
    st.subheader("Model Performance")
    show_metrics(b_test_org, b_pred_original)

    # --- Fixed Plot ---
    fig, ax = plt.subplots()
    sorted_idx = b_test_org.reset_index(drop=True).sort_values(by=b_test_org.columns[0]).index
    b_test_sorted = b_test_org.reset_index(drop=True).iloc[sorted_idx]
    b_pred_sorted = pd.DataFrame(b_pred_original).iloc[sorted_idx]

    ax.plot(b_test_sorted.values, b_pred_sorted.values, color='blue', label="Predicted Line")
    ax.plot([min(b_test_org.values), max(b_test_org.values)],
            [min(b_test_org.values), max(b_test_org.values)], 'r--', label="Ideal Line")
    ax.set_title("Fare Amount Predictor (Polynomial Regression)")
    ax.set_xlabel("Actual Fare Amount ($)")
    ax.set_ylabel("Predicted Fare Amount ($)")
    ax.legend()
    st.pyplot(fig)


def SupportVectorRegression1():
    from sklearn.svm import SVR
    regressor = SVR(kernel='rbf')
    regressor.fit(a_train, b_train.values.ravel())
    b_pred = regressor.predict(a_test)

    # Convert back to original fare scale
    b_pred_original = sc_fare.inverse_transform(b_pred.reshape(-1,1))
    st.title("Fare Amount Prediction using Support Vector Regression")
    st.write("Predicted Fare ($)", round(b_pred_original[0][0], 2))
    st.write("Actual Fare ($)", round(b_test_org.iloc[0, 0], 2))

    st.subheader("Model Performance")
    show_metrics(b_test_org, b_pred_original)

    # --- Fixed Plot ---
    fig, ax = plt.subplots()
    sorted_idx = b_test_org.reset_index(drop=True).sort_values(by=b_test_org.columns[0]).index
    b_test_sorted = b_test_org.reset_index(drop=True).iloc[sorted_idx]
    b_pred_sorted = pd.DataFrame(b_pred_original).iloc[sorted_idx]

    ax.plot(b_test_sorted.values, b_pred_sorted.values, color='blue', label="Predicted Line")
    ax.plot([min(b_test_org.values), max(b_test_org.values)],
            [min(b_test_org.values), max(b_test_org.values)], 'r--', label="Ideal Line")
    ax.set_title("Fare Amount Predictor (Support Vector Regression)")
    ax.set_xlabel("Actual Fare Amount ($)")
    ax.set_ylabel("Predicted Fare Amount ($)")
    ax.legend()
    st.pyplot(fig)


def RandomForest1():
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(a_train, b_train.values.ravel())
    b_pred = regressor.predict(a_test)

    b_pred_original = sc_fare.inverse_transform(b_pred.reshape(-1,1))
    st.title("Fare Amount Prediction using Random Forest")
    st.write("Predicted Fare ($)", round(b_pred_original[0][0], 2))
    st.write("Actual Fare ($)", round(b_test_org.iloc[0, 0], 2))

    st.subheader("Model Performance")
    show_metrics(b_test_org, b_pred_original)

    # --- Fixed Plot ---
    fig, ax = plt.subplots()
    sorted_idx = b_test_org.reset_index(drop=True).sort_values(by=b_test_org.columns[0]).index
    b_test_sorted = b_test_org.reset_index(drop=True).iloc[sorted_idx]
    b_pred_sorted = pd.DataFrame(b_pred_original).iloc[sorted_idx]

    ax.plot(b_test_sorted.values, b_pred_sorted.values, color='blue', label="Predicted Line")
    ax.plot([min(b_test_org.values), max(b_test_org.values)],
            [min(b_test_org.values), max(b_test_org.values)], 'r--', label="Ideal Line")
    ax.set_title("Fare Amount Predictor (Random Forest)")
    ax.set_xlabel("Actual Fare Amount ($)")
    ax.set_ylabel("Predicted Fare Amount ($)")
    ax.legend()
    st.pyplot(fig)

    # Feature importance
    st.subheader("Feature Importances (Random Forest)")
    importances = pd.Series(regressor.feature_importances_, index=a_train.columns).sort_values(ascending=False)
    st.bar_chart(importances)



st.subheader("Choose what to predict ")
predict = st.selectbox("Select a prediction target", ["Fare Amount", "Distance(km)"])
if predict=='Distance(km)':
    model = st.selectbox("Select a model", ["Linear Regression", "Polynomial Regression", 
                                            "Support Vector Regression", "Random Forest"])
    if model == 'Linear Regression':
        linear()
    elif model == 'Polynomial Regression':
        polynomial()
    elif model == 'Support Vector Regression':
        SupportVectorRegression()
    else:
        RandomForest()
else :
    model1 = st.selectbox("Select a model", ["Linear Regression", "Polynomial Regression", 
                                            "Support Vector Regression", "Random Forest"])
    if model1 == 'Linear Regression':
        linear1()
    elif model1 == 'Polynomial Regression':
        polynomial1()
    elif model1 == 'Support Vector Regression':
        SupportVectorRegression1()
    else:
        RandomForest1()