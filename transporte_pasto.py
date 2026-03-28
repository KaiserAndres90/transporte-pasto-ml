library(reticulate)
py_install("pandas")
py_install("numpy")
py_install("scikit-learn")
repl_python()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
RUTAS = ['C1', 'C6', 'C15', 'E5', 'C3', 'C8', 'C12', 'E2', 'C10', 'S1']
ZONAS = ['Centro', 'Catambuco', 'Briceño', 'Anganoy', 'Comuna 10']
fechas = pd.date_range(start='2024-01-01', end='2024-12-31', freq='30min')
n_registros = len(fechas)
dataset = pd.DataFrame({
    'timestamp': fechas,
    'hora': fechas.hour,
    'dia_semana': fechas.dayofweek,
    'mes': fechas.month,
    'festivo': [1 if fecha.day in [1,25] and fecha.month in [1,12] else 0 
                for fecha in fechas]
})
dataset['temperatura'] = 13 + 4 * np.sin(2 * np.pi * (dataset['mes'] - 3) / 12) + \
                         np.random.normal(0, 1.5, n_registros)
dataset['precipitacion'] = np.maximum(0, 80 + 40 * np.sin(2 * np.pi * (dataset['mes'] - 4) / 12) + \
                                      np.random.normal(0, 15, n_registros))



dataset['ruta'] = np.random.choice(RUTAS, n_registros)
dataset['zona_origen'] = np.random.choice(ZONAS, n_registros)
dataset['flota_disponible'] = np.random.randint(15, 45, n_registros)
demanda_base = 100 + 150 * np.exp(-((dataset['hora'] - 7) ** 2) / 20) + \
               100 * np.exp(-((dataset['hora'] - 13) ** 2) / 15) + \
               200 * np.exp(-((dataset['hora'] - 18) ** 2) / 25)

factor_laboral = np.where(dataset['dia_semana'] < 5, 1.4, 1.0)
factor_festivo = np.where(dataset['festivo'] == 1, 0.6, 1.0)
factor_clima = np.maximum(0.7, 1 - 0.003 * dataset['precipitacion'])

dataset['pasajeros'] = (demanda_base * factor_laboral * factor_festivo * 
                         factor_clima + np.random.normal(0, 25, n_registros)).astype(int)
dataset['pasajeros'] = np.maximum(0, dataset['pasajeros'])

X = pd.get_dummies(dataset.drop(columns=['pasajeros', 'timestamp']), 
                   columns=['ruta', 'zona_origen'])
y = dataset['pasajeros']

num_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Entrenamiento: {X_train.shape}")
print(f"Prueba: {X_test.shape}")

modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

print("Modelo entrenado exitosamente")

y_pred = modelo.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.2f} pasajeros")

importancias = pd.DataFrame({
    'caracteristica': X.columns,
    'importancia': modelo.feature_importances_
}).sort_values('importancia', ascending=False)

print(importancias.head(10))




