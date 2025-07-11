import streamlit as st
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import joblib

# Базовые настройки страницы
st.set_page_config(
    page_title="EDF7: catboost model", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
# Заголовок приложения
st.title("Команда EDF 7 представляет")
st.header("Модель :blue[CatBoost] для предсказания _медианы суммарных остатков на всех счетах клиента на горизонте двух месяцев_")

@st.cache_resource
def load_model():
    model = joblib.load("catboost_model.joblib")
    return model

model = load_model()

# Функция для предсказания
def make_prediction(features):
    # Преобразуем в numpy array и reshape для одной строки
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)
    pred_final = np.exp(prediction) - 1
    return pred_final[0]

# Создаем форму для ввода данных
with st.form("prediction_form"):
    st.subheader("Введите значения признаков")

    # Используем колонки для лучшего расположения
    col1, col2 = st.columns(2)

    with col1:
        savings_sum_sa_credit_1m = st.number_input(
            "Сумма поступлений на счета с продуктом Save account за последний месяц :gray[(руб.)]", 
            value=10000, 
            min_value=0, 
            max_value=10**8, 
            step=1000
        )
        savings_sum_sa_debet_1m = st.number_input(
            "Сумма списаний со счетов с продуктом Save account за последний месяц :gray[(руб.)]", 
            value=2000, 
            min_value=0, 
            max_value=10**8, 
            step=1000
        )

    with col2:
        savings_sum_sa_1m = st.number_input(
            "Суммарный остаток по продукту Save account на отчётную дату минус последний месяц :gray[(руб.)]", 
            value=50000, 
            min_value=0, 
            max_value=10**8, 
            step=1000
        )
        min_max_dep_balance_amt_term_savings = st.number_input(
            "Минимальный остаток по срочным депозитам и накопительным счетам клиента за последние 3 года :gray[(руб.)]",
            value=0,
            min_value=0,
            max_value=10**9,
            step=1000,
        )

    submitted = st.form_submit_button("Сделать предсказание")

# При нажатии кнопки делаем предсказание
if submitted:
    features = [savings_sum_sa_credit_1m, savings_sum_sa_debet_1m, 
                savings_sum_sa_1m, min_max_dep_balance_amt_term_savings]
    prediction = make_prediction(features)

    st.subheader("Результат предсказания")
    st.success(f"Предсказанная медиана суммарных остатков по всем счетам: **{prediction:.2f}** рублей")

    # Дополнительная информация
    with st.expander("Детали предсказания"):
        st.write("Использованные признаки:")
        feature_data = pd.DataFrame(
            {
                "Признак": [
                    "savings_sum_sa_credit_1m",
                    "savings_sum_sa_debet_1m",
                    "savings_sum_sa_1m",
                    "min_max_dep_balance_amt_term_savings",
                ],
                "Значение": features,
            }
        )
        st.table(feature_data)

        st.write("Пример важности признаков (для демонстрации):")
        # Получаем важность признаков (если модель поддерживает)
        try:
            importance = model.get_feature_importance()
            importance_data = pd.DataFrame(
                {
                    "Признак": [
                        "savings_sum_sa_credit_1m",
                        "savings_sum_sa_debet_1m",
                        "savings_sum_sa_1m",
                        "min_max_dep_balance_amt_term_savings",
                ],
                    "Важность": importance,
                }
            )
            st.bar_chart(importance_data.set_index("Признак"))
        except:
            st.info("Эта модель не поддерживает получение важности признаков")


# Боковая панель с информацией
st.sidebar.header("Информация")
st.sidebar.info(
    """
    Приложение является упрощённой моделью на основе **CatBoostRegressor**. 
    Упрощённая версия использует **4 признака**.
    """
)
st.sidebar.markdown(
    """
"""
)