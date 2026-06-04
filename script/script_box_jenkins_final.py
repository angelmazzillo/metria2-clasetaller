# %% =========================
# Importar paquetes
# ============================

import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path
from statsmodels.tsa.stattools import kpss

warnings.filterwarnings("ignore")

# %% =========================
# Cargar datos
# ============================

# Obtener la ruta del directorio raíz
BASE_DIR = Path(__file__).resolve().parents[1]

# Obtener la ruta del directorio con los datos
DATA_DIR = BASE_DIR / "data"

# Rutas de las bases de datos
ruta_exp = DATA_DIR / "ExpoColombia1844-2024.xlsx"  # Base de datos de exportaciones

# Base de datos con la serie importada a python
expocolombia = pd.read_excel(ruta_exp)

# %% =========================
# Crear índice temporal
# ============================

fechas = pd.date_range(start="1844", periods=len(expocolombia), freq="YS")

expocolombia.index = fechas

print(expocolombia.head())
print(expocolombia.tail())


# %% =========================
# Crear serie de tiempo
# ============================

y = expocolombia["Valor de las exportaciones como porcentaje del PIB"].copy()
y = pd.to_numeric(y, errors="coerce")
y = y.dropna()

print(y.head())
print(y.tail())
print(y.describe())


# %% =========================
# PASO 1: IDENTIFICACIÓN
# ============================

plt.figure(figsize=(10, 5))
plt.plot(y)
plt.title("Valor de exportaciones (% del PIB) para Colombia, 1844-2024")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()


# %% FAC y FACP de la serie original

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(y, lags=24, alpha=0.05, bartlett_confint=False, ax=axes[0])

axes[0].set_title("FAC de la serie original")

plot_pacf(y, lags=24, alpha=0.05, ax=axes[1])

axes[1].set_title("FACP de la serie original")

plt.tight_layout()
plt.show()

# %% Prueba de estacionariedad Dickey-Fuller Aumentada
print("=== Prueba Dickey-Fuller Aumentada: Serie original ===")
resultado_diff = adfuller(y.dropna())
print(f"Estadístico ADF: {resultado_diff[0]:.4f}")
print(f"p-valor: {resultado_diff[1]:.4f}")
print("Valores críticos:")
for clave, valor in resultado_diff[4].items():
    print(f"   {clave}: {valor:.4f}")

# H0: la serie tiene raíz unitaria (no es estacionaria).
# Si p-value > 0.05, no se rechaza H0.

# %% Prueba de estacionariedad KPSS

print("=== Prueba KPSS: Serie original ===")
resultado_kpss = kpss(y.dropna(), regression="c", nlags="auto")
print(f"Estadístico KPSS: {resultado_kpss[0]:.4f}")
print(f"p-valor: {resultado_kpss[1]:.4f}")
print("Valores críticos:")
for clave, valor in resultado_kpss[3].items():
    print(f"   {clave}: {valor:.4f}")

# H0: la serie es estacionaria.
# Si p-value > 0.05, no se rechaza H0.

# Resultados: la serie es estacionaria de acuerdo a ambas pruebas, por lo que en principio no sería necesario aplicar ninguna
# transformación de diferenciación. Sin embargo, la serie en realidad presenta una casi-raíz unitaria, como se verá a continuación,
# por lo que es posible que un modelo ARIMA con d=1 sea más adecuado para capturar esa dinámica.

# %% Serie diferenciada
y_diff = y.diff().dropna()

# Gráfica de la serie de tiempo de la "diferencia exportaciones tradicionales"
plt.figure(figsize=(10, 5))
plt.plot(y_diff)
plt.title(
    "Valor de exportaciones (% del PIB) para Colombia, 1844-2024, serie diferenciada"
)
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# %% FAC y FACP de la serie diferenciada

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(y_diff, lags=24, alpha=0.05, bartlett_confint=False, ax=axes[0])

axes[0].set_title("FAC de la serie diferenciada")

plot_pacf(y_diff, lags=24, alpha=0.05, ax=axes[1])

axes[1].set_title("FACP de la serie diferenciada")

plt.tight_layout()
plt.show()

# %% Prueba de estacionariedad Dickey-Fuller Aumentada
print("=== Prueba Dickey-Fuller Aumentada: Serie diferenciada ===")
resultado_diff = adfuller(y_diff.dropna())
print(f"Estadístico ADF: {resultado_diff[0]:.4f}")
print(f"p-valor: {resultado_diff[1]:.4f}")
print("Valores críticos:")
for clave, valor in resultado_diff[4].items():
    print(f"   {clave}: {valor:.4f}")

# H0: la serie tiene raíz unitaria (no es estacionaria).
# Si p-value > 0.05, no se rechaza H0.

# %% Prueba de estacionariedad KPSS

print("=== Prueba KPSS: Serie diferenciada ===")
resultado_kpss = kpss(y_diff.dropna(), regression="c", nlags="auto")
print(f"Estadístico KPSS: {resultado_kpss[0]:.4f}")
print(f"p-valor: {resultado_kpss[1]:.4f}")
print("Valores críticos:")
for clave, valor in resultado_kpss[3].items():
    print(f"   {clave}: {valor:.4f}")

# H0: la serie es estacionaria.
# Si p-value > 0.05, no se rechaza H0.

# Luego de aplicar transformación diferenciación, la serie probablemente es estacionariade acuerdo a ambas pruebas.
# Por lo tanto, proponemos que de ser necesario aplicar una diferenciación,
# entonces el orden de diferenciación debe ser d=1 para el modelo ARIMA.

# %% ========================================================
# Antes y después: serie original vs serie diferenciada
# ===========================================================

# Creamos el lienzo principal
fig, ax1 = plt.subplots(figsize=(11, 5.5))

# --- Serie original (eje Y izquierdo - Azul) ---
color_original = "#1f77b4"
ax1.set_xlabel("Fecha", fontsize=11, fontweight="semibold")
ax1.set_ylabel(
    "Serie Original (% del PIB)",
    color=color_original,
    fontsize=11,
    fontweight="semibold",
)

# Usamos directamente el objeto 'y' (su .index ya contiene las fechas)
line1 = ax1.plot(
    y.index,
    y,
    color=color_original,
    linewidth=1.8,
    label="Valor de exportaciones (% PIB)",
)
ax1.tick_params(axis="y", labelcolor=color_original)
ax1.grid(True, linestyle="--", alpha=0.5)  # Cuadrícula sutil de fondo


ax2 = ax1.twinx()
color_dif = "#ff0e0e"
ax2.set_ylabel(
    "Serie Diferenciada", color=color_dif, fontsize=11, fontweight="semibold"
)

line2 = ax2.plot(
    y_diff.index,
    y_diff,
    color=color_dif,
    linewidth=1.2,
    alpha=0.85,
    label="Serie Diferenciada (d=1)",
)
ax2.tick_params(axis="y", labelcolor=color_dif)

# --- Agregar una línea horizontal de referencia en 0 para la serie diferenciada ---
ax2.axhline(0, color="gray", linestyle=":", alpha=0.6)

# --- Configurar Leyenda Combinada ---
# Sumamos las listas de líneas y etiquetas para agruparlas en un solo cuadro
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(
    lines, labels, loc="upper right", frameon=True, facecolor="white", edgecolor="none"
)

# Título del gráfico
plt.title(
    "Transformación de Exportaciones en Colombia (1844-2024)\nSerie Original vs. Serie Diferenciada",
    fontsize=13,
    fontweight="bold",
    pad=15,
)

plt.tight_layout()
plt.show()

# %% Criterios de información AIC y BIC (SERIE ORIGINAL)

# Listas con los órdenes que vamos a probar (de 0 a 2)
p_values = [0, 1, 2]
q_values = [0, 1, 2]
d = 0  # Diferencia fija

resultados = []

# Bucle para evaluar cada combinación
for p in p_values:
    for q in q_values:
        try:
            # Estimamos el modelo sobre la serie original (especificando order=(p, d, q))
            # Es mejor pasarle 'y' y dejar que SARIMAX haga la diferencia con d=1
            modelo = SARIMAX(
                np.log(y),
                order=(p, d, q),
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            resultado_fit = modelo.fit(disp=False)

            # Guardamos la información de cada modelo
            resultados.append(
                {
                    "Modelo": f"ARIMA({p}, {d}, {q})",
                    "AIC": resultado_fit.aic,
                    "BIC": resultado_fit.bic,
                }
            )
        except Exception:
            continue

# Convertir los resultados en un DataFrame de Pandas
tabla_criterios = pd.DataFrame(resultados)

# Ordenar la tabla de menor a mayor según el criterio BIC (el más estricto)
tabla_criterios = tabla_criterios.sort_values(by="BIC").reset_index(drop=True)

# Activar de nuevo las advertencias
warnings.filterwarnings("default")

# Mostrar la tabla final
print("=== RESULTADOS DE CÁLCULO DE AIC Y BIC (SERIE ORIGINAL) ===")
print(tabla_criterios)

# %% Criterios de información AIC y BIC (SERIE DIFERENCIADA)

# Listas con los órdenes que vamos a probar (de 0 a 2)
p_values = [0, 1, 2]
q_values = [0, 1, 2]
d = 1  # Diferencia fija

resultados = []

# Bucle para evaluar cada combinación
for p in p_values:
    for q in q_values:
        try:
            # Estimamos el modelo sobre la serie original (especificando order=(p, d, q))
            # Es mejor pasarle 'y' y dejar que SARIMAX haga la diferencia con d=1
            modelo = SARIMAX(
                np.log(y),
                order=(p, d, q),
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            resultado_fit = modelo.fit(disp=False)

            # Guardamos la información de cada modelo
            resultados.append(
                {
                    "Modelo": f"ARIMA({p}, {d}, {q})",
                    "AIC": resultado_fit.aic,
                    "BIC": resultado_fit.bic,
                }
            )
        except Exception:
            continue

# Convertir los resultados en un DataFrame de Pandas
tabla_criterios = pd.DataFrame(resultados)

# Ordenar la tabla de menor a mayor según el criterio BIC (el más estricto)
tabla_criterios = tabla_criterios.sort_values(by="BIC").reset_index(drop=True)

# Activar de nuevo las advertencias
warnings.filterwarnings("default")

# Mostrar la tabla final
print("=== RESULTADOS DE CÁLCULO DE AIC Y BIC (SERIE DIFERENCIADA) ===")
print(tabla_criterios)

# %% =========================
# PASO 2: ESTIMACIÓN
# ============================

# %% Modelo sobre la serie original, de acuerdo a la FAC y FACP y a los criterios de información

modelo_ar1 = SARIMAX(
    y,
    order=(1, 0, 0),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False,
)

resultado_ar1 = modelo_ar1.fit(disp=False)

print(resultado_ar1.summary())

# %% Modelo sobre la serie diferenciada, de acuerdo a la FAC y FACP y a los criterios de información

modelo_arima02 = SARIMAX(
    y,
    order=(0, 1, 2),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False,
)

resultado_arima02 = modelo_arima02.fit(disp=False)

print(resultado_arima02.summary())


# %% =========================
# PASO 3: VALIDACIÓN DEL MODELO
# ============================

residuos = resultado_arima02.resid.dropna()

print(residuos.describe())


# %% FAC de los residuos

ig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(residuos, lags=24, alpha=0.05, bartlett_confint=False, ax=axes[0])

axes[0].set_title("FAC de los residuos")

plot_pacf(residuos, lags=24, alpha=0.05, ax=axes[1])

axes[1].set_title("FACP de los residuos")

plt.tight_layout()
plt.show()

# %% FAC de los residuos al cuadrado

ig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(residuos**2, lags=24, alpha=0.05, bartlett_confint=False, ax=axes[0])

axes[0].set_title("FAC de los residuos al cuadrado")

plot_pacf(residuos**2, lags=24, alpha=0.05, ax=axes[1])

axes[1].set_title("FACP de los residuos al cuadrado")

plt.tight_layout()
plt.show()


# %% Prueba Ljung-Box

ljung_box = acorr_ljungbox(residuos, lags=[6, 12, 18, 24], return_df=True)

print("Prueba Ljung-Box")
print(ljung_box)

# H0: no hay autocorrelación en los residuos.
# Si p-value > 0.05, no se rechaza H0.


# %% Prueba ARCH de heterocedasticidad

arch_test = het_arch(residuos, nlags=12)

print("Prueba ARCH")
print("LM statistic:", arch_test[0])
print("LM p-value:", arch_test[1])

# H0: no hay efectos ARCH.
# Si p-value > 0.05, no se rechaza H0.


# %% Prueba de normalidad Jarque-Bera

jb_test = jarque_bera(residuos)

print("Prueba Jarque-Bera")
print("Jarque-Bera statistic:", jb_test.statistic)
print("Jarque-Bera p-value:", jb_test.pvalue)

# H0: los residuos siguen una distribución normal.
# Si p-value > 0.05, no se rechaza H0.

# %% Q-Q plot de los residuos

plt.figure(figsize=(6, 6))
stats.probplot(residuos, dist="norm", plot=plt)
plt.title("Q-Q plot de los residuos")
plt.grid(True)
plt.show()

# %% =========================
# PASO 4: PRONÓSTICO
# ============================

modelo_pronostico = SARIMAX(
    y,
    order=(0, 1, 2),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False,
)

resultado_pronostico = modelo_pronostico.fit(disp=False)

pronostico_nivel = resultado_pronostico.get_forecast(steps=12)

media = pronostico_nivel.predicted_mean
intervalos_nivel = pronostico_nivel.conf_int()

tabla_pronostico = pd.DataFrame(
    {
        "pronostico": media,
        "limite_inferior": intervalos_nivel.iloc[:, 0],
        "limite_superior": intervalos_nivel.iloc[:, 1],
    }
)

print(tabla_pronostico)
# %%
plt.figure(figsize=(10, 5))
plt.plot(y, label="Datos históricos")

plt.plot(media, label="Pronóstico", color="orange")

plt.fill_between(
    media.index,
    intervalos_nivel.iloc[:, 0],
    intervalos_nivel.iloc[:, 1],
    color="orange",
    alpha=0.3,
    label="Intervalo de confianza",
)

plt.title("Pronóstico del Modelo ARIMA(0,1,2)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Año", fontsize=11)
plt.ylabel("Exportaciones (% del PIB)", fontsize=11)

plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

# Mostrar gráfica limpia
plt.tight_layout()
plt.show()
