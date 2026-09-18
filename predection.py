import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score,classification_report,mean_squared_error,r2_score,silhouette_score)

import next_purchese as npmod


def ml_pre(df,numeric_cols,category_cols,date_cols=None):
    if date_cols is None:
        date_cols = []
    st.subheader("ML PREDECTION")
    
    ML_type=st.radio(
        "CHOOSE ML TYPE",
        ["Supervised","Unsupervised","Next Purchase (N Days)"]
    )
    
    if ML_type == "Supervised":
        supervised_learning(df,numeric_cols,category_cols)
    elif ML_type == "Unsupervised":
        unsupervised_learning(df,numeric_cols)
    else:
        npmod.next_purchase_prediction(df,date_cols,category_cols,numeric_cols)
         
def supervised_learning(df,numeric_cols,category_cols):
    st.markdown("superviced learning")
        
    all_col=list(df.columns)
    target_col=st.selectbox("CHOOSE THE COLUMNS  AND (WHAT YOU WANT HE PREDECT)",all_col)
        
    default_feature=[c for c in numeric_cols if c !=target_col]
    feature_cols=st.multiselect("select the columns(input)",[c for c in all_col if c !=target_col],
                                default=default_feature)
    if not feature_cols:
        st.warning("please select the columns")
        return
    if st.button("Train Model"):
        try:
            data=df[feature_cols + [target_col]].dropna()
            x=data[feature_cols].copy()
            y=data[target_col].copy()
            
            #encode and categoryical  and an object feature columns
            for col in x.select_dtypes(include=["object","category"]).columns:
                le=LabelEncoder()
                x[col]=le.fit_transform(x[col].astype(str))
            
            is_classfication = (
    y.dtype == "object"
    or str(y.dtype) == "category"
    or (
        y.nunique() <= 15
        and not pd.api.types.is_float_dtype(y)
    )
)
            
            if is_classfication and (y.dtype == "object" or str(y.dtype) == "category"):
                y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=y.index)

            x_train, x_test, y_train, y_test = train_test_split(
                x,y,test_size=0.2,random_state=42
                )
            
            
            scaler=StandardScaler()
            x_train_scaled=scaler.fit_transform(x_train)
            x_test_scaled=scaler.transform(x_test)
            
            if is_classfication:
                model=RandomForestClassifier(n_estimators=200,random_state=42)
                model.fit(x_train_scaled,y_train)
                pred=model.predict(x_test_scaled)
                        
                        
                acc=accuracy_score(y_test,pred)
                st.success(f"MODEL HAS BEED TRAINED: {acc :.2%}")
                        
                st.markdown("-- Classfication report--")
                report=classification_report(y_test,pred,output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
            else:
                model=RandomForestRegressor(n_estimators=200,random_state=42)
                model.fit(x_train_scaled,y_train)
                pred=model.predict(x_test_scaled)
                        
                        
                rmse=np.sqrt(mean_squared_error(y_test,pred))
                r2=r2_score(y_test,pred)
                        
                st.success("MODEL TRAINED")
                col1,col2=st.columns(2)
                col1.metric("RMSE",f"{rmse :.4f}")
                col2.metric("R2 score",f"{r2 :.4f}")
                        
                st.dataframe(pd.DataFrame({"ACTUVAL DATA" :y_test,"PREDECT VALUE" :pred}).head(20)
                )
                st.markdown("-- Actual vs Predicted Trend --")
                fig, ax = plt.subplots()
                ax.scatter(y_test, pred, alpha=0.5, label="Predictions")
                
                min_val = min(y_test.min(), pred.min())
                max_val = max(y_test.max(), pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Fit")
                
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.legend()
                st.pyplot(fig)   
            st.markdown("-- feature imporatance")
            importance_df = pd.DataFrame(
    {
        "feature": feature_cols,
        "importance": model.feature_importances_
    }
).sort_values(
    "importance",
    ascending=False
)

            st.bar_chart(importance_df.set_index("feature")
)
                    
        except Exception as e:
            st.error(f"TRANING IS ERROR : {e}")
                
    # unsupervices function
def unsupervised_learning(df, numeric_cols):
    st.markdown(" UNSUPERVICED LEARNING (CLUSTERING)")

    if len(numeric_cols) < 2:
        st.warning("NEED TO CHOOSE AT LEAST 2 COLUMNS FOR CLUSTERING")
        return

    selected_cols = st.multiselect(
        "SELECTED COLUMNS FOR CLUSTERING",
        numeric_cols,
        default=numeric_cols[:min(4, len(numeric_cols))],
    )
    if len(selected_cols) < 2:
        st.warning("pls select the 2 columns")
        return

    n_clusters = st.slider("number of cluster (k)", min_value=2, max_value=10, value=3)

    if st.button("RUN CLUSTERING"):
        try:
            data = df[selected_cols].dropna()
            scaler = StandardScaler()
            scaled = scaler.fit_transform(data)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled)

            result_df = data.copy()
            result_df["cluster"] = clusters

            sil_score = silhouette_score(scaled, clusters)

            # save everything needed for display so it survives reruns
            st.session_state["cluster_scaled"] = scaled
            st.session_state["cluster_result_df"] = result_df
            st.session_state["cluster_sil_score"] = sil_score

        except Exception as e:
            st.error(f"clustering error :{e}")

    # --- everything below runs OUTSIDE the button, using saved session_state ---
    if "cluster_result_df" in st.session_state:
        scaled = st.session_state["cluster_scaled"]
        result_df = st.session_state["cluster_result_df"]

        st.success(f"clustering done! silhouste score :{st.session_state['cluster_sil_score']:.4f}")

        fg = st.number_input(
            "How many rows do you want to see?",
            min_value=10, max_value=99999999999999,
            step=1, key="cluster_result_rows"
        )
        st.dataframe(result_df.head(fg))

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled)

        plot_df = pd.DataFrame({
            "PCA1": pca_result[:, 0],
            "PCA2": pca_result[:, 1],
            "clusters": result_df["cluster"].astype(str),
        })

        st.markdown("-- CLUSTERS VISVALIATION (PCA)--")
        st.scatter_chart(plot_df, x="PCA1", y="PCA2", color="clusters")

        if st.checkbox("Show Elbow Method Trend"):
            inertias = []
            k_range = range(2, 11)
            for k in k_range:
                km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
                km_test.fit(scaled)
                inertias.append(km_test.inertia_)

            fig, ax = plt.subplots()
            ax.plot(list(k_range), inertias, marker='o', linestyle='-', color='b')
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("Inertia")
            ax.set_title("Elbow Method - Trend Line")
            st.pyplot(fig)

        st.markdown("cluster size")
        st.bar_chart(result_df["cluster"].value_counts())

if __name__ == "__main__":
    import file_selection as fs
    import title as t
    import data_utils as du

    st.set_page_config(page_title="MADHESH VELAYUDHAM PROJECTS", layout="wide")
    t.show_title()

    mad =fs.uploaded_data()
    if mad is not None:
        cleaned = du.clean_data(mad)
        final_df, date_cols, numeric_cols, category_cols = du.catogry_sep(cleaned)

        st.subheader("Cleaned Data")
        st.dataframe(final_df.head(20))

        st.divider()
        ml_pre(final_df, numeric_cols, category_cols, date_cols)