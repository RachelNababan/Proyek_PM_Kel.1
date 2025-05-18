from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load datasets
df_cities = pd.read_csv("cities.csv")
df_countries = pd.read_csv("countries.csv")

# Merge datasets
merged_data = pd.merge(df_cities, df_countries, on='country', how='left')

# Handle missing values
for col in merged_data.select_dtypes(include='object').columns:
    merged_data[col] = merged_data[col].fillna("Unknown")
for col in merged_data.select_dtypes(include=['float64', 'int64']).columns:
    merged_data[col] = merged_data[col].fillna(merged_data[col].mean())

# Simulasi data dengan pola yang lebih realistis
np.random.seed(42)
merged_data['month'] = np.random.randint(1, 13, size=len(merged_data))
merged_data['temp'] = 15 + 10 * np.sin(2 * np.pi * (merged_data['month'] - 1) / 12) + np.random.normal(0, 3, size=len(merged_data))
merged_data['humidity'] = 30 + 40 * np.abs(np.sin(2 * np.pi * (merged_data['month'] - 1) / 12)) + np.random.normal(0, 10, size=len(merged_data))
merged_data['wind_speed'] = 5 + 5 * np.random.random(size=len(merged_data))
merged_data['day'] = np.random.randint(1, 32, size=len(merged_data))
merged_data['year'] = np.random.randint(2000, 2031, size=len(merged_data))

# Tentukan kondisi cuaca berdasarkan suhu dan kelembapan
conditions = []
for i in range(len(merged_data)):
    temp = merged_data['temp'].iloc[i]
    hum = merged_data['humidity'].iloc[i]
    if temp > 30 and hum < 50:
        conditions.append('☀️ Sunny')
    elif temp < 20 or hum > 80:
        conditions.append('🌧️ Rainy')
    elif 20 <= temp <= 30 and 50 <= hum <= 80:
        conditions.append('⛅ Cloudy')
    else:
        conditions.append('⛈️ Stormy')
merged_data['weather_condition'] = conditions

# Feature engineering: Tambah fitur musim
def get_season(month):
    if month in [12, 1, 2]:
        return 0
    elif month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    else:
        return 3
merged_data['season'] = merged_data['month'].apply(get_season)
merged_data['humidity_wind_interaction'] = merged_data['humidity'] * merged_data['wind_speed']

# Siapkan data untuk regresi dan klasifikasi
scaler_reg = StandardScaler()
X_reg = merged_data[['humidity', 'wind_speed', 'day', 'month', 'year', 'season', 'humidity_wind_interaction']]
X_reg_scaled = scaler_reg.fit_transform(X_reg)
y_reg = merged_data['temp']

scaler_clf = StandardScaler()
X_clf = merged_data[['temp', 'humidity', 'wind_speed', 'day', 'month', 'year', 'season', 'humidity_wind_interaction']]
X_clf_scaled = scaler_clf.fit_transform(X_clf)
label_encoder = LabelEncoder()
y_clf = label_encoder.fit_transform(merged_data['weather_condition'])

# Periksa distribusi kelas di y_clf
class_counts = pd.Series(y_clf).value_counts()
stratify_y_clf = y_clf if (class_counts > 1).all() else None

# Train-test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf_scaled, y_clf, test_size=0.2, random_state=42, stratify=stratify_y_clf)

# Latih model utama
model_reg = LinearRegression()
model_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = model_reg.predict(X_test_reg)
mae_reg = mean_absolute_error(y_test_reg, y_pred_reg)

model_clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
model_clf.fit(X_train_clf, y_train_clf)
y_pred_clf = model_clf.predict(X_test_clf)
accuracy_clf = accuracy_score(y_test_clf, y_pred_clf)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = {}
    form_data = {}

    if request.method == "POST":
        city = request.form["city"]
        humidity = int(request.form["humidity"])
        wind = int(request.form["wind"])
        day = int(request.form["day"])
        month = int(request.form["month"])
        year = int(request.form["year"])

        form_data = {
            "city": city,
            "humidity": humidity,
            "wind": wind,
            "day": day,
            "month": month,
            "year": year
        }

        season = get_season(month)
        humidity_wind = humidity * wind
        input_reg = np.array([[humidity, wind, day, month, year, season, humidity_wind]])
        input_reg_scaled = scaler_reg.transform(input_reg)
        temp_pred = model_reg.predict(input_reg_scaled)[0]

        input_clf = np.array([[temp_pred, humidity, wind, day, month, year, season, humidity_wind]])
        input_clf_scaled = scaler_clf.transform(input_clf)
        weather_pred_encoded = model_clf.predict(input_clf_scaled)[0]
        weather_pred = label_encoder.inverse_transform([weather_pred_encoded])[0]

        # ========= GRAFIK DINAMIS =========
        new_input_df = pd.DataFrame([{
            "humidity": humidity,
            "wind_speed": wind,
            "day": day,
            "month": month,
            "year": year,
            "season": season,
            "humidity_wind_interaction": humidity_wind
        }])
        combined_X_reg = pd.concat([X_reg, new_input_df], ignore_index=True)
        combined_y_reg = pd.concat([y_reg, pd.Series([temp_pred])], ignore_index=True)

        scaler_temp = StandardScaler()
        combined_X_scaled = scaler_temp.fit_transform(combined_X_reg)

        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
            combined_X_scaled, combined_y_reg, test_size=0.2, random_state=None
        )

        temp_model = LinearRegression()
        temp_model.fit(X_train_new, y_train_new)
        y_pred_new = temp_model.predict(X_test_new)

        errors = y_pred_new - y_test_new
        colors = np.where(np.abs(errors) < 1, "#66ff66", np.where(errors > 0, "#ff6666", "#6699ff"))

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test_new, y_pred_new, c=colors, alpha=0.7, edgecolors="black", label="Data")
        
        # Titik prediksi input user
        user_scaled = scaler_temp.transform(new_input_df)
        user_y = temp_model.predict(user_scaled)[0]
        ax.scatter([temp_pred], [user_y], c="black", marker="x", s=100, label="Input Pengguna")
        
        ax.set_xlabel("Suhu Aktual (°C)")
        ax.set_ylabel("Suhu Prediksi (°C)")
        ax.set_title("📉 Grafik Prediksi Dinamis")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("static/plot.png")
        plt.close()

        # ==================================

        prediction = {
            "city": city,
            "temperature": f"{temp_pred:.2f}",
            "condition": weather_pred,
            "mae": f"{mae_reg:.2f}",
            "accuracy": f"{accuracy_clf:.2f}",
            "has_result": True
        }

    return render_template("index.html", cities=merged_data['city_name'].unique(),
                           prediction=prediction, form_data=form_data)

if __name__ == "__main__":
    app.run(debug=True)
