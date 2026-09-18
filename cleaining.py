import streamlit as st
import pandas as pd
import file_selection as fs
import title as t
import data_utils as du
import predection as pred

st.set_page_config(page_title="MADHESH VELAYUDHAM PROJECTS", layout="wide")
t.show_title()

mad=fs.uploaded_data()
if mad is not None:
    st.subheader("raw data")

    cleaned=du.clean_data(mad)
    final_df,date_cols,numeric_cols,category_cols=du.catogry_sep(cleaned)
    st.subheader("Automaticaly Cleaned Data")
    fg=st.number_input("How many rows do you want to see?",min_value=10,max_value=99999999999999,step=1,key="cleaned_data_rows")
    st.dataframe(final_df.head(int(fg)))
 
    st.subheader("Detected Columns")
    col1,col2,col3=st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("📅 Date Columns:")
            st.table(pd.DataFrame({"column":date_cols
                                   }))
    with col2:
        with st.container(border=True):
            st.markdown("🔢 Numeric Columns:")
            st.table(pd.DataFrame({"column":numeric_cols
                                   }))
    with col3:
        with st.container(border=True):
            st.markdown("🏷️ Category Columns:")
            st.table(pd.DataFrame({"column":category_cols
                                   }))


wantbox = st.checkbox("If You need to see Data Types")
if wantbox:
    st.subheader("Data Types")
    st.write(final_df.dtypes.astype(str))

    st.divider()
    pred.ml_pre(final_df, numeric_cols, category_cols, date_cols)

    
