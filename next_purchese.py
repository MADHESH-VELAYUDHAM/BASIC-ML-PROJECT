import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def next_purchase_prediction(df, date_cols, category_cols, numeric_cols):
    st.markdown("### PREDICT NEXT CUSTOMER SELECTION (WITHIN N DAYS)")

    all_cols = list(df.columns)

    customer_col = st.selectbox(
        "SELECT CUSTOMER ID COLUMN", all_cols, key="npc_customer_col"
    )
    date_options = date_cols if date_cols else all_cols
    date_col = st.selectbox("SELECT DATE COLUMN", date_options, key="npc_date_col")

    target_options = [c for c in category_cols if c != customer_col] or [
        c for c in all_cols if c not in (customer_col, date_col)
    ]
    target_col = st.selectbox(
        "SELECT COLUMN TO PREDICT (product / category customer selects)",
        target_options,
        key="npc_target_col",
    )

    window_days = st.slider(
        "PREDICTION WINDOW (DAYS)", min_value=7, max_value=90, value=30, key="npc_window"
    )

    if st.button("BUILD NEXT-PURCHASE MODEL", key="npc_train_btn"):
        try:
            data = df[[customer_col, date_col, target_col]].dropna().copy()
            data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
            data = data.dropna(subset=[date_col])
            data = data.sort_values([customer_col, date_col])

            # ovoru customer oda purchase sequence la irundhu (current -> next within window) pairs edukurom
            rows = []
            for cust, grp in data.groupby(customer_col):
                grp = grp.reset_index(drop=True)
                for i in range(len(grp) - 1):
                    current = grp.loc[i]
                    nxt = grp.loc[i + 1]
                    gap = (nxt[date_col] - current[date_col]).days
                    if gap <= window_days:
                        rows.append(
                            {
                                "customer": cust,
                                "purchase_count_so_far": i + 1,
                                "days_since_prev": (
                                    (current[date_col] - grp.loc[i - 1][date_col]).days
                                    if i > 0
                                    else 0
                                ),
                                "current_category": current[target_col],
                                "month": current[date_col].month,
                                "dayofweek": current[date_col].dayofweek,
                                "next_category": nxt[target_col],
                            }
                        )

            if not rows:
                st.warning(
                    f"NO CUSTOMER HAS 2+ PURCHASES WITHIN A {window_days}-DAY GAP. TRY A BIGGER WINDOW."
                )
                return

            train_df = pd.DataFrame(rows)
            st.write(
                f"BUILT {len(train_df)} TRAINING EXAMPLES FROM {train_df['customer'].nunique()} CUSTOMERS"
            )

            le_current = LabelEncoder()
            le_next = LabelEncoder()
            train_df["current_category_enc"] = le_current.fit_transform(
                train_df["current_category"].astype(str)
            )
            train_df["next_category_enc"] = le_next.fit_transform(
                train_df["next_category"].astype(str)
            )

            feature_cols = [
                "purchase_count_so_far",
                "days_since_prev",
                "current_category_enc",
                "month",
                "dayofweek",
            ]
            x = train_df[feature_cols]
            y = train_df["next_category_enc"]

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)

            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(x_train_scaled, y_train)
            pred = model.predict(x_test_scaled)

            acc = accuracy_score(y_test, pred)
            st.success(f"MODEL TRAINED - TEST ACCURACY: {acc:.2%}")

            # next run kum use panna session_state la vechikurom
            st.session_state["npc_model"] = model
            st.session_state["npc_scaler"] = scaler
            st.session_state["npc_le_current"] = le_current
            st.session_state["npc_le_next"] = le_next
            st.session_state["npc_history"] = data
            st.session_state["npc_customer_col_used"] = customer_col
            st.session_state["npc_date_col_used"] = date_col
            st.session_state["npc_target_col_used"] = target_col
            st.session_state["npc_window_used"] = window_days

        except Exception as e:
            st.error(f"TRAINING ERROR: {e}")

    if "npc_model" in st.session_state:
        st.divider()
        st.markdown("#### PREDICT FOR A SPECIFIC CUSTOMER")

        hist_df = st.session_state["npc_history"]
        ccol = st.session_state["npc_customer_col_used"]
        dcol = st.session_state["npc_date_col_used"]
        tcol = st.session_state["npc_target_col_used"]
        win = st.session_state["npc_window_used"]

        cust_list = sorted(hist_df[ccol].astype(str).unique().tolist())
        chosen_cust = st.selectbox("SELECT CUSTOMER", cust_list, key="npc_predict_customer")

        if st.button("PREDICT NEXT SELECTION", key="npc_predict_btn"):
            try:
                hist = hist_df[hist_df[ccol].astype(str) == chosen_cust].sort_values(dcol)
                if hist.empty:
                    st.warning("NO HISTORY FOUND FOR THIS CUSTOMER")
                    return

                last_row = hist.iloc[-1]
                purchase_count_so_far = len(hist)
                days_since_prev = (
                    (hist.iloc[-1][dcol] - hist.iloc[-2][dcol]).days if len(hist) > 1 else 0
                )

                le_current = st.session_state["npc_le_current"]
                le_next = st.session_state["npc_le_next"]

                current_cat = str(last_row[tcol])
                if current_cat not in le_current.classes_:
                    st.warning("THIS CUSTOMER'S LAST CATEGORY WASN'T SEEN DURING TRAINING")
                    return
                current_enc = le_current.transform([current_cat])[0]

                feat = pd.DataFrame(
                    [
                        {
                            "purchase_count_so_far": purchase_count_so_far,
                            "days_since_prev": days_since_prev,
                            "current_category_enc": current_enc,
                            "month": last_row[dcol].month,
                            "dayofweek": last_row[dcol].dayofweek,
                        }
                    ]
                )

                feat_scaled = st.session_state["npc_scaler"].transform(feat)
                model = st.session_state["npc_model"]
                pred_enc = model.predict(feat_scaled)[0]
                pred_label = le_next.inverse_transform([pred_enc])[0]

                st.success(
                    f"PREDICTED NEXT SELECTION (WITHIN {win} DAYS): **{pred_label}**"
                )

                proba = model.predict_proba(feat_scaled)[0]
                top_n = min(5, len(proba))
                top_idx = np.argsort(proba)[::-1][:top_n]
                top_labels = le_next.inverse_transform(top_idx)
                top_df = pd.DataFrame(
                    {"category": top_labels, "probability": proba[top_idx]}
                )
                st.dataframe(top_df)

            except Exception as e:
                st.error(f"PREDICTION ERROR: {e}")

        st.divider()
        st.markdown("#### PREDICT FOR ALL CUSTOMERS (TABLE + DOWNLOAD)")

        if st.button("PREDICT FOR ALL CUSTOMERS", key="npc_predict_all_btn"):
            try:
                model = st.session_state["npc_model"]
                scaler = st.session_state["npc_scaler"]
                le_current = st.session_state["npc_le_current"]
                le_next = st.session_state["npc_le_next"]

                results = []
                for cust in cust_list:
                    hist_c = hist_df[hist_df[ccol].astype(str) == cust].sort_values(dcol)
                    if hist_c.empty:
                        continue

                    last_row = hist_c.iloc[-1]
                    purchase_count_so_far = len(hist_c)
                    days_since_prev = (
                        (hist_c.iloc[-1][dcol] - hist_c.iloc[-2][dcol]).days
                        if len(hist_c) > 1
                        else 0
                    )
                    current_cat = str(last_row[tcol])

                    if current_cat not in le_current.classes_:
                        results.append(
                            {
                                "customer": cust,
                                "last_purchase_date": last_row[dcol].date(),
                                "last_category": current_cat,
                                "purchase_count": purchase_count_so_far,
                                "predicted_next_category": "UNKNOWN (new category)",
                                "confidence": None,
                            }
                        )
                        continue

                    current_enc = le_current.transform([current_cat])[0]
                    feat = pd.DataFrame(
                        [
                            {
                                "purchase_count_so_far": purchase_count_so_far,
                                "days_since_prev": days_since_prev,
                                "current_category_enc": current_enc,
                                "month": last_row[dcol].month,
                                "dayofweek": last_row[dcol].dayofweek,
                            }
                        ]
                    )
                    feat_scaled = scaler.transform(feat)
                    pred_enc = model.predict(feat_scaled)[0]
                    pred_label = le_next.inverse_transform([pred_enc])[0]
                    proba = model.predict_proba(feat_scaled)[0]
                    confidence = round(float(proba.max()), 4)

                    results.append(
                        {
                            "customer": cust,
                            "last_purchase_date": last_row[dcol].date(),
                            "last_category": current_cat,
                            "purchase_count": purchase_count_so_far,
                            "predicted_next_category": pred_label,
                            "confidence": confidence,
                        }
                    )

                if not results:
                    st.warning("NO CUSTOMERS TO PREDICT FOR")
                else:
                    result_df = pd.DataFrame(results)
                    st.dataframe(result_df)

                    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "DOWNLOAD PREDICTIONS AS CSV",
                        data=csv_bytes,
                        file_name="next_purchase_predictions.csv",
                        mime="text/csv",
                        key="npc_download_csv",
                    )

            except Exception as e:
                st.error(f"BATCH PREDICTION ERROR: {e}")