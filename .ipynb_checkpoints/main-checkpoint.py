!pip install -q numpy pandas scikit-learn streamlit matplotlib os joblib wget
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import joblib
import os

id_slice = slice(32, 65)

data_path = "https://drive.google.com/file/d/1hl5RG-hh8T67bzoiSSAv57lo9q-cyjUu/view?usp=sharing"
model_path = "https://drive.google.com/file/d/1POdEp3vw34kgnKyN2VytUeY_zaIAYEQ_/view?usp=sharing"

data_file_id = data_path[id_slice]
model_file_id = model_path[id_slice]

os.system(f"wget -q -O train_tmp.csv https://drive.google.com/uc?id={data_file_id}")
os.system(f"wget -q -O idealmodel.pkl https://drive.google.com/uc?id={model_file_id}")

# Функция для предобработки данных
def preprocess_data(df):
    """
    Performs transformations on df and returns transformed df.
    """
    # Заполнение числовых значений медианой
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                df[label+"_is_missing"] = pd.isnull(content)
                df[label] = content.fillna(content.median())
                
        # Заполнение пропущенных значений категориальных признаков и преобразование их в числа
        if not pd.api.types.is_numeric_dtype(content):
            df[label+"_is_missing"] = pd.isnull(content)
            df[label] = pd.Categorical(content).codes + 1
            
    return df

# Функция для построения графика важности признаков
def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({"features": columns,
                        "feature_importances": importances})
         .sort_values("feature_importances", ascending=False)
         .reset_index(drop=True))
    # Построение графика
    fig, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:20])
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature importance")
    ax.invert_yaxis()

# Функция для загрузки модели с помощью pickle
def load_model(model_path):
    joblib_model = joblib.load(model_path)
    return joblib_model

# Страница "График важности признаков"
def feature_importance_page(data, model):
    st.title("График важности признаков")
    st.subheader("Анализ важности признаков в модели")

    # Получение важности признаков из модели
    feature_importances = model.feature_importances_
    columns = data.columns

    # Построение графика важности признаков
    plot_features(columns, feature_importances)
    st.pyplot()

# Страница "Обработка данных"
def data_processing_page(data, model):
    st.title("Обработка данных")
    st.subheader("Предобработка данных перед обучением модели")

    # Вывод обработанных данных
    st.dataframe(data)

# Основная часть

st.title("Модель машинного обучения")

# Загрузка данных
data = pd.read_csv("train_tmp.csv")

# Предобработка данных
processed_data = preprocess_data(data)

# Загрузка модели
model = load_model("idealmodel.pkl") 

df_val = data[data["saleYear"] == 2012]
df_train = data[data["saleYear"] != 2012]

X_train, y_train = df_train.drop("SalePrice", axis=1), df_train["SalePrice"]
X_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val["SalePrice"]

model.fit(X_train, y_train)

# Навигационная панель
pages = {
    "График важности признаков": feature_importance_page,
    "Обработка данных": data_processing_page
}
page = st.sidebar.selectbox("Выберите страницу", list(pages.keys()))
pages[page](X_train, model)
