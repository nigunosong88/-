
# import data_operations
from data.data import data_csv
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import streamlit as st
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
import plotly.offline as pyo
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
import numpy as np
import os

st.set_page_config(layout="wide")

def app_main():
    run_optuna(n_estimators, max_depth, subsample, n_trials)
        
X,y=data_csv()
def run_optuna(n_estimators, max_depth, subsample, n_trials):

    hyper_params = {"n_estimators": n_estimators, "max_depth": max_depth, "subsample": subsample}
    st.info("您選擇的超參數為：{}".format(
        {str(hp): value for hp, value in hyper_params.items()}
    ))
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", n_estimators[0], n_estimators[1]),
            "max_depth": trial.suggest_int("max_depth", max_depth[0], max_depth[1]),
            "subsample": trial.suggest_float("subsample", subsample[0],subsample[1], step=0.1)
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        LGBM_model = LGBMRegressor(**params, verbose=-1, random_state=0)

        mse_mean  = mean(cross_val_score(LGBM_model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1))
        return mse_mean 

    with st.spinner('請稍候... 正在進行模型調參，共 {} 次試驗'.format(n_trials)):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
    try:
        if len(study.trials) > 0:
            st.success('完成！模型訓練與調參已經完成。這裡是您最佳的超參數和最高準確率！')
            param1, param2, param3, score = st.columns(4)
            cols = [param1, param2, param3]
            for (param, value), col in zip(study.best_params.items(), cols):
                with col:
                    st.info("{} : {}".format(param, value))
            with score:
                st.success("最佳準確率：{}".format(round(study.best_value, 3)))
        else:
            st.error("未完成任何試驗，請檢查模型配置和數據。")
    except Exception as e:
        print("錯誤：", e)
        return None 
    
    if not os.path.exists('figures'):
        os.makedirs('figures')
    fig = plt.figure(figsize=(10, 5))
    fig_par_importances = optuna.visualization.plot_param_importances(study)
    plt.savefig('figures/fig_par_importances.jpeg', bbox_inches='tight')
    fig_history = optuna.visualization.plot_optimization_history(study)
    plt.savefig('figures/fig_history.jpeg', bbox_inches='tight')

    st.markdown("## 視覺分析")
    st.info("**超參數重要性**向我們展示了每個參數對我們模型的重要性。__**歷史圖表**__向我們展示了 Optuna 試驗的歷史記錄。目標值是目標函數的返回值，在我們的案例中是準確率，我們正在嘗試將其最大化。如果您用許多試驗調整模型，您可以看到 Optuna 所做的更改。")
    fig_par_importances_col, fig_history_col = st.columns(2)
    with fig_par_importances_col:
        st.image("figures/fig_par_importances.jpeg")
    with fig_history_col:
        st.image("figures/fig_history.jpeg")

    st.plotly_chart(fig_par_importances, use_container_width=True)
    st.plotly_chart(fig_history, use_container_width=True)
    
    # 切片圖和平行坐標圖分析
    slice_plot = optuna.visualization.plot_slice(study)
    st.info("當我們查看 __**切片圖**__ 的洞察時，我們可以看到每個超參數的特定範圍內有一些強度。如果一個超參數在高限處獲得高分，我們可能想要提高限制，反之亦然。")
    st.plotly_chart(slice_plot, use_container_width=True)
    st.info("__**平行坐標**__ 圖與 __**切片圖**__ 類似。我們可以看到超參數之間的每一個連接。分析圖表我們可以說，模型在較低的最大特徵、較低的最大深度和較低的估計數量下給我們更好的分數。注意：這份數據僅有150個實例，因此分析可能會有所不同！這個應用的主要焦點是理解 Optuna 和使用 Streamlit 創建應用！")
    parallel_coordinates = optuna.visualization.plot_parallel_coordinate(study)
    st.plotly_chart(parallel_coordinates, use_container_width=True)

st.sidebar.markdown("# 使用 Optuna 進行超參數調整")


# related links and contact
sidebar_info_html_string = """調整參數"""
st.sidebar.markdown(sidebar_info_html_string, unsafe_allow_html=True)

n_estimators = st.sidebar.slider('n_estimators --決策樹的數量。',100, 1000, (150, 300))
max_depth = st.sidebar.slider('max_depth --決策樹的最大深度',2, 50, (3, 10))
subsample = st.sidebar.slider('subsample --數據樣本的比率',0.1, 1.0, (0.4, 0.8))
n_trials = st.sidebar.slider('您想要多少次迭代?',10, 100, 15)

st.markdown("# HyperParameter Tuning With Optuna")

info, data_preview = st.columns([1, 4])

with info:
    main_page_info_html_string = """Hello
    """
    st.markdown(main_page_info_html_string, unsafe_allow_html=True)

with data_preview:
    data = pd.read_csv("data/E_lvr_land_A.csv")
    st.dataframe(data)


if __name__ == '__main__':
    if st.button("...開始使用選定的超參數調整模型。..."):
        app_main()
