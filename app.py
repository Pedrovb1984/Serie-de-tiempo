import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from prophet import Prophet
from datos import cargar_datos

# Configuración de la página
st.set_page_config(
    page_title="Pronóstico de Pasajeros",
    page_icon="✈️",
    layout="wide"
)

# Cargar los datos
@st.cache_data
def obtener_datos():
    df = cargar_datos()
    df["Month"] = pd.to_datetime(df["Month"])
    df = df.sort_values("Month")
    return df

df = obtener_datos()

# Título principal
st.title("¿Cuántos pasajeros tendremos el próximo año?")

# Calcular métricas para los 3 indicadores superiores
pasajeros_1949 = df[df["Month"].dt.year == 1949]["Passengers"].sum()
pasajeros_1960 = df[df["Month"].dt.year == 1960]["Passengers"].sum()
crecimiento = pasajeros_1960 / pasajeros_1949

# Mostrar los 3 indicadores arriba
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Pasajeros Totales en 1949", f"{pasajeros_1949:,}")
with col2:
    st.metric("Pasajeros Totales en 1960", f"{pasajeros_1960:,}")
with col3:
    st.metric("Crecimiento (Veces)", f"{crecimiento:.2f}x")

st.markdown("---")

# Crear las cuatro pestañas
pestana1, pestana2, pestana3, pestana4 = st.tabs([
    "1. La historia",
    "2. La temporada",
    "3. El examen",
    "4. El pronóstico"
])

# ---------------------------------------------------------
# PESTANA 1: La historia
# ---------------------------------------------------------
with pestana1:
    st.subheader("Evolución mensual de pasajeros")
    
    fig1 = px.line(
        df, 
        x="Month", 
        y="Passengers", 
        labels={"Month": "Mes", "Passengers": "Número de Pasajeros"},
        title="Historial completo de pasajeros de avión"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    st.info(
        "Este gráfico muestra cómo ha crecido el número de pasajeros mes a mes a lo largo de los años. "
        "Se puede observar una tendencia clara al alza acompañada de altibajos constantes que se repiten cada año. "
        "Para el negocio, esto significa que la demanda general va creciendo, pero es crucial prepararse para las fluctuaciones anuales."
    )

# ---------------------------------------------------------
# PESTANA 2: La temporada
# ---------------------------------------------------------
with pestana2:
    st.subheader("Comportamiento estacional (mes a mes)")
    
    df_temp = df.copy()
    df_temp["Mes_Nombre"] = df_temp["Month"].dt.month_name()
    df_temp["Mes_Num"] = df_temp["Month"].dt.month
    
    promedio_meses = df_temp.groupby(["Mes_Num", "Mes_Nombre"])["Passengers"].mean().reset_index()
    promedio_meses = promedio_meses.sort_values("Mes_Num")
    promedio_general = df["Passengers"].mean()
    
    fig2 = px.bar(
        promedio_meses,
        x="Mes_Nombre",
        y="Passengers",
        labels={"Mes_Nombre": "Mes", "Passengers": "Promedio de Pasajeros"},
        title="Promedio de pasajeros por mes (Temporada alta y baja)"
    )
    fig2.add_hline(
        y=promedio_general, 
        line_dash="dot", 
        line_color="red", 
        annotation_text=f"Promedio general ({promedio_general:.1f})", 
        annotation_position="bottom right"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    st.info(
        "Aquí comparamos el promedio de pasajeros de cada mes con la línea roja que marca el promedio general del año. "
        "Los meses que superan la línea roja son la temporada alta (como los meses de verano), mientras que los demás son temporada baja. "
        "Esto le indica a la empresa en qué periodos se debe reforzar el personal y la flota, y cuándo se pueden planear mantenimientos."
    )

# ---------------------------------------------------------
# PESTANA 3: El examen
# ---------------------------------------------------------
with pestana3:
    st.subheader("Evaluación de múltiples modelos con los últimos 24 meses")
    
    # Separar datos: entrenamiento y prueba (últimos 24 meses)
    train = df.iloc[:-24].copy()
    test = df.iloc[-24:].copy()
    
    # 1. Copiar el año pasado
    copia_ano_pasado = df.iloc[-36:-12]["Passengers"].values
    
    # 2. Suavizamiento Exponencial (Holt-Winters)
    ts_train = train.set_index("Month")["Passengers"]
    try:
        modelo_hw = ExponentialSmoothing(
            ts_train, 
            trend="add", 
            seasonal="mul", 
            seasonal_periods=12
        ).fit()
        pred_hw = modelo_hw.forecast(24).values
    except Exception:
        pred_hw = copia_ano_pasado
        
    # 3. SARIMA
    try:
        modelo_sarima = SARIMAX(
            ts_train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        pred_sarima = modelo_sarima.forecast(24).values
    except Exception:
        pred_sarima = copia_ano_pasado
        
    # 4. Regresión Lineal
    try:
        train_lr = train.copy()
        train_lr["Tiempo"] = range(len(train_lr))
        X_train = train_lr[["Tiempo"]]
        y_train = train_lr["Passengers"]
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        tiempo_test = range(len(train_lr), len(train_lr) + 24)
        X_test = pd.DataFrame({"Tiempo": list(tiempo_test)})
        pred_lr = lr.predict(X_test)
    except Exception:
        pred_lr = copia_ano_pasado
        
    # 5. Prophet
    try:
        df_prophet = train[["Month", "Passengers"]].rename(columns={"Month": "ds", "Passengers": "y"})
        m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m_prophet.fit(df_prophet)
        future = m_prophet.make_future_dataframe(periods=24, freq='MS')
        forecast_p = m_prophet.predict(future)
        pred_prophet = forecast_p.tail(24)["yhat"].values
    except Exception:
        pred_prophet = copia_ano_pasado

    # Crear DataFrame de evaluación
    eval_df = test[["Month", "Passengers"]].copy()
    eval_df["Real"] = eval_df["Passengers"]
    eval_df["Copiar_Anio_Pasado"] = copia_ano_pasado
    eval_df["Suavizamiento_Exponencial"] = pred_hw
    eval_df["SARIMA"] = pred_sarima
    eval_df["Regresion_Lineal"] = pred_lr
    eval_df["Prophet"] = pred_prophet
    
    # Gráfico comparativo
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=eval_df["Month"], y=eval_df["Real"], mode="lines+markers", name="Real", line=dict(color="black", width=3)))
    fig3.add_trace(go.Scatter(x=eval_df["Month"], y=eval_df["Copiar_Anio_Pasado"], mode="lines+markers", name="Copiar año pasado", line=dict(dash="dash")))
    fig3.add_trace(go.Scatter(x=eval_df["Month"], y=eval_df["Suavizamiento_Exponencial"], mode="lines+markers", name="Suavizamiento Exponencial", line=dict(dash="dot")))
    fig3.add_trace(go.Scatter(x=eval_df["Month"], y=eval_df["SARIMA"], mode="lines+markers", name="SARIMA", line=dict(dash="dashdot")))
    fig3.add_trace(go.Scatter(x=eval_df["Month"], y=eval_df["Regresion_Lineal"], mode="lines+markers", name="Regresión Lineal", line=dict(dash="longdash")))
    fig3.add_trace(go.Scatter(x=eval_df["Month"], y=eval_df["Prophet"], mode="lines+markers", name="Prophet"))
    
    fig3.update_layout(title="Comparación de todos los modelos vs Real (últimos 24 meses)", xaxis_title="Mes", yaxis_title="Pasajeros")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Calcular errores (MAPE)
    def calc_mape(real, pred):
        return (abs(real - pred) / real).mean() * 100

    err_copia = calc_mape(eval_df["Real"], eval_df["Copiar_Anio_Pasado"])
    err_hw = calc_mape(eval_df["Real"], eval_df["Suavizamiento_Exponencial"])
    err_sarima = calc_mape(eval_df["Real"], eval_df["SARIMA"])
    err_lr = calc_mape(eval_df["Real"], eval_df["Regresion_Lineal"])
    err_prophet = calc_mape(eval_df["Real"], eval_df["Prophet"])
    
    tabla_errores = pd.DataFrame({
        "Modelo": [
            "Copiar el año pasado", 
            "Suavizamiento Exponencial", 
            "SARIMA", 
            "Regresión Lineal", 
            "Prophet"
        ],
        "Error Promedio (%)": [
            f"{err_copia:.2f}%", 
            f"{err_hw:.2f}%", 
            f"{err_sarima:.2f}%", 
            f"{err_lr:.2f}%", 
            f"{err_prophet:.2f}%"
        ]
    })
    
    # Encontrar el menor error numérico para el ganador
    errores_dict = {
        "Copiar el año pasado": err_copia,
        "Suavizamiento Exponencial": err_hw,
        "SARIMA": err_sarima,
        "Regresión Lineal": err_lr,
        "Prophet": err_prophet
    }
    ganador = min(errores_dict, key=errores_dict.get)
    
    st.table(tabla_errores)
    st.write(f"**Ganador del examen:** {ganador}")
    
    st.info(
        "En este examen pusimos a prueba múltiples modelos matemáticos y estadísticos para adivinar los últimos dos años. "
        "La tabla muestra el porcentaje de error promedio de cada estrategia (cuanto menor sea, mejor). "
        "El modelo ganador demuestra ser el más preciso y confiable para anticiparse al comportamiento futuro del negocio."
    )

# ---------------------------------------------------------
# PESTANA 4: El pronóstico
# ---------------------------------------------------------
with pestana4:
    st.subheader("Pronóstico para los 12 meses de 1961")
    
    # Selector de modelo para el pronóstico final
    modelo_elegido = st.selectbox(
        "Selecciona el modelo para generar el pronóstico de 1961:",
        ["Suavizamiento Exponencial", "SARIMA", "Regresión Lineal", "Prophet", "Copiar el año pasado"]
    )
    
    ts_total = df.set_index("Month")["Passengers"]
    fechas_1961 = pd.date_range(start="1961-01-01", periods=12, freq="MS")
    
    if modelo_elegido == "Suavizamiento Exponencial":
        m_fin = ExponentialSmoothing(ts_total, trend="add", seasonal="mul", seasonal_periods=12).fit()
        pred_vals = m_fin.forecast(12).values
    elif modelo_elegido == "SARIMA":
        m_fin = SARIMAX(ts_total, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        pred_vals = m_fin.forecast(12).values
    elif modelo_elegido == "Regresión Lineal":
        df_lr = df.copy()
        df_lr["Tiempo"] = range(len(df_lr))
        lr = LinearRegression()
        lr.fit(df_lr[["Tiempo"]], df_lr["Passengers"])
        tiempo_1961 = range(len(df_lr), len(df_lr) + 12)
        pred_vals = lr.predict(pd.DataFrame({"Tiempo": list(tiempo_1961)}))
    elif modelo_elegido == "Prophet":
        df_p = df.rename(columns={"Month": "ds", "Passengers": "y"})
        m_p = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m_p.fit(df_p)
        future_p = m_p.make_future_dataframe(periods=12, freq='MS')
        forecast_p = m_p.predict(future_p)
        pred_vals = forecast_p.tail(12)["yhat"].values
    else: # Copiar el año pasado
        pred_vals = df.iloc[-12:]["Passengers"].values
        
    df_pronostico = pd.DataFrame({
        "Month": fechas_1961,
        "Passengers_Forecast": pred_vals
    })
    
    # Gráfico del histórico y el pronóstico
    fig4 = go.Figure()
    df_reciente = df.tail(36)
    fig4.add_trace(go.Scatter(x=df_reciente["Month"], y=df_reciente["Passengers"], mode="lines+markers", name="Histórico"))
    fig4.add_trace(go.Scatter(x=df_pronostico["Month"], y=df_pronostico["Passengers_Forecast"], mode="lines+markers", name=f"Pronóstico 1961 ({modelo_elegido})", line=dict(dash="dash", color="green")))
    fig4.update_layout(title=f"Histórico y Pronóstico para el año 1961 usando {modelo_elegido}", xaxis_title="Mes", yaxis_title="Pasajeros")
    st.plotly_chart(fig4, use_container_width=True)
    
    # Tabla mes a mes
    tabla_mostrar = df_pronostico.copy()
    tabla_mostrar["Mes"] = tabla_mostrar["Month"].dt.strftime("%Y-%m")
    tabla_mostrar["Pasajeros Esperados"] = tabla_mostrar["Passengers_Forecast"].round().astype(int)
    tabla_mostrar = tabla_mostrar[["Mes", "Pasajeros Esperados"]]
    
    st.table(tabla_mostrar)
    
    st.info(
        "Este gráfico y tabla presentan el pronóstico oficial de pasajeros mes a mes para todo el año 1961 utilizando el modelo seleccionado. "
        "Permite a la administración planificar con antelación los recursos financieros, la capacidad de los flota y la contratación de personal. "
        "De esta manera, el negocio puede anticiparse al crecimiento de la demanda de forma ordenada."
    )
