
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Getz Pharma Customer Segmentation",
    page_icon="💊",
    layout="wide",
)

st.title("💊 Getz Pharma Customer Segmentation")
st.subheader("Unsupervised Learning with K-Means")
st.caption(
    "Goal: group customers by how they behave, so the sales and credit "
    "teams can treat each group differently."
)

# ------------------------------------------------------------------
# Load the data
# ------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "getz_pharma_customer_segmentation_1000.csv"


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


if not DATA_FILE.exists():
    st.error(f"Could not find {DATA_FILE.name}. Keep it in the same folder as pharma.py")
    st.stop()

df = load_data(DATA_FILE)


features = [
    "Monthly_Orders",
    "Avg_Order_Value_PKR",
    "Products_Purchased",
    "Payment_Delay_Days",
    "Days_Since_Last_Order",
]

X_scaled = StandardScaler().fit_transform(df[features])


@st.cache_data
def get_k_scores(scaled):
    # try K = 2..8 and keep inertia (elbow) + silhouette (second opinion)
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled)
        rows.append(
            {
                "K": k,
                "Inertia": km.inertia_,
                "Silhouette": silhouette_score(scaled, km.labels_),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def run_kmeans(scaled, k):
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")

selected_k = st.sidebar.slider("Number of clusters (K)", 2, 6, 3)

st.sidebar.subheader("🔎 View filters")
st.sidebar.caption("Filters change what you see. The model always learns from all customers.")

all_cities = sorted(df["City"].unique())
all_types = sorted(df["Customer_Type"].unique())

city_filter = st.sidebar.multiselect("City", all_cities, default=all_cities)
type_filter = st.sidebar.multiselect("Customer type", all_types, default=all_types)


df_clustered = df.copy()
df_clustered["Cluster"] = run_kmeans(X_scaled, selected_k)

# name the clusters by how much they spend (lowest -> highest)
spend_order = (
    df_clustered.groupby("Cluster")["Monthly_Spend_PKR"]
    .mean()
    .sort_values()
    .index.tolist()
)

if selected_k == 3:
    names = ["Low Engagement", "Regular Customers", "High Value Loyal"]
else:
    names = [f"Segment {i + 1}" for i in range(selected_k)]

df_clustered["Segment"] = df_clustered["Cluster"].map(dict(zip(spend_order, names)))

# PCA: squeeze 5 features into 2 so we can draw the clusters
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df_clustered["PC1"] = coords[:, 0]
df_clustered["PC2"] = coords[:, 1]
variance_kept = pca.explained_variance_ratio_.sum()

# apply the sidebar filters (display only)
view = df_clustered[
    df_clustered["City"].isin(city_filter)
    & df_clustered["Customer_Type"].isin(type_filter)
]

if view.empty:
    st.warning("No customers match these filters. Please widen your selection.")
    st.stop()


m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers", f"{len(view):,}")
m2.metric("Cities", view["City"].nunique())
m3.metric("Avg Monthly Orders", f"{view['Monthly_Orders'].mean():.1f}")
m4.metric("Avg Monthly Spend", f"PKR {view['Monthly_Spend_PKR'].mean():,.0f}")


segment_summary = (
    view.groupby("Segment")
    .agg(
        Customers=("Customer_ID", "count"),
        Avg_Monthly_Orders=("Monthly_Orders", "mean"),
        Avg_Order_Value_PKR=("Avg_Order_Value_PKR", "mean"),
        Avg_Products=("Products_Purchased", "mean"),
        Avg_Payment_Delay=("Payment_Delay_Days", "mean"),
        Avg_Days_Since_Last_Order=("Days_Since_Last_Order", "mean"),
        Avg_Return_Rate=("Return_Rate_Percent", "mean"),
        Avg_Months_With_Us=("Months_With_Company", "mean"),
        Avg_Monthly_Spend_PKR=("Monthly_Spend_PKR", "mean"),
    )
    .round(1)
    .sort_values("Avg_Monthly_Spend_PKR", ascending=False)
    .reset_index()
)

revenue = view.groupby("Segment")["Monthly_Spend_PKR"].sum().reset_index()
revenue["Share_%"] = (
    revenue["Monthly_Spend_PKR"] / revenue["Monthly_Spend_PKR"].sum() * 100
).round(1)


tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 1. Dataset", "📈 2. Choose K", "🤖 3. Customer Segments", "🧠 4. What We Learned"]
)


with tab1:
    st.header("Step 1: Understand the Data")
    st.write(
        "Before using machine learning, we first understand what every column "
        "means. A good data scientist never trains a model without understanding the data."
    )

    st.dataframe(
        view.drop(columns=["Cluster", "Segment", "PC1", "PC2"]).head(20),
        width="stretch",
        hide_index=True,
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Missing values check")
        missing = df.isna().sum().reset_index()
        missing.columns = ["Column", "Missing_Values"]
        st.dataframe(missing, width="stretch", hide_index=True)
        if missing["Missing_Values"].sum() == 0:
            st.success("No missing values - the data is clean.")

    with right:
        st.subheader("Basic statistics")
        st.dataframe(df[features + ["Monthly_Spend_PKR"]].describe().round(1), width="stretch")

    st.subheader("Features used by the model")
    st.write(
        "We feed only **buying behaviour** to K-Means: "
        + ", ".join(f"`{f}`" for f in features)
        + ". Monthly spend is left out on purpose because it is orders × order value. "
        "We use it later to judge the groups."
    )

    feature_to_plot = st.selectbox("Distribution of a feature", features)
    # simple histogram: cut into 20 bins and count
    bins = pd.cut(view[feature_to_plot], bins=20)
    hist = bins.value_counts().sort_index()
    hist.index = [round(i.mid) for i in hist.index]
    st.bar_chart(hist, x_label=feature_to_plot, y_label="Customers")


with tab2:
    st.header("Step 2: How many groups should we make?")

    scores = get_k_scores(X_scaled)
    best_k = int(scores.loc[scores["Silhouette"].idxmax(), "K"])

    st.write(
        "We train K-Means for different values of K. **Inertia** tells us how "
        "tightly customers sit around their cluster centre. Lower is better, but it "
        "always drops as K grows, so we look for the *elbow* where the improvement slows down."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Elbow chart")
        st.line_chart(scores, x="K", y="Inertia", x_label="Number of clusters (K)")
    with c2:
        st.subheader("Silhouette score (higher is better)")
        st.line_chart(scores, x="K", y="Silhouette", x_label="Number of clusters (K)")

    st.success(
        f"Silhouette is highest at K = {best_k} ({scores['Silhouette'].max():.2f}), "
        "and the elbow bends near the same place. Two methods agreeing "
        "gives us confidence. Use the sidebar slider to try other values."
    )
    st.dataframe(scores.round(3), width="stretch", hide_index=True)


with tab3:
    st.header("Step 3: Explore the Groups K-Means Discovered")

    st.subheader("How many customers are in each group?")
    segment_counts = (
        view["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
    )
    st.bar_chart(
        segment_counts, x="Segment", y="Customers",
        x_label="Customer Segments", y_label="Number of Customers",
    )

    st.subheader("Visualize the clusters")
    s1, s2 = st.columns(2)
    with s1:
        st.caption("Orders vs order value (bubble size = products purchased)")
        st.scatter_chart(
            view, x="Monthly_Orders", y="Avg_Order_Value_PKR",
            color="Segment", size="Products_Purchased",
            x_label="Monthly Orders", y_label="Average Order Value (PKR)",
        )
    with s2:
        st.caption(f"All 5 features squeezed into 2D ({variance_kept:.0%} of information kept)")
        st.scatter_chart(view, x="PC1", y="PC2", color="Segment")

    st.subheader("Cluster summary")
    st.dataframe(segment_summary, width="stretch", hide_index=True)

    st.subheader("Who actually brings in the money?")
    st.bar_chart(revenue, x="Segment", y="Share_%", y_label="Share of monthly revenue (%)")

    st.subheader("Who is in each segment?")
    mix_by = st.radio("Break down by", ["Customer_Type", "City"], horizontal=True)
    mix = pd.crosstab(view["Segment"], view[mix_by], normalize="index").mul(100).round(1)
    st.bar_chart(mix, stack=True, y_label="Percent of segment")

    if selected_k == 3:
        st.success(
            "The model discovered three behaviour groups on its own. "
            "We give them names only AFTER studying their behaviour."
        )
        b1, b2, b3 = st.columns(3)
        with b1:
            st.subheader("🌸 High Value Loyal")
            st.write("Many orders, big baskets, wide product range, pay on time. "
                     "**Protect them:** priority delivery, dedicated rep.")
        with b2:
            st.subheader("🙂 Regular Customers")
            st.write("The steady middle. **Grow them** with bundles and cross-selling.")
        with b3:
            st.subheader("😴 Low Engagement")
            st.write("Few small orders, slower payments, more returns. "
                     "**Re-engage** them or tighten credit terms.")
    else:
        st.info("Business names are assigned only for K = 3. Compare the summary table instead.")

    st.subheader("Explore individual customers")
    segment_pick = st.selectbox("Choose a segment", sorted(view["Segment"].unique()))
    picked = view[view["Segment"] == segment_pick].drop(columns=["Cluster", "PC1", "PC2"])
    st.dataframe(picked, width="stretch", hide_index=True)

    st.download_button(
        "⬇️ Download all customers with segments (CSV)",
        data=df_clustered.drop(columns=["PC1", "PC2"]).to_csv(index=False).encode("utf-8"),
        file_name="getz_customers_with_segments.csv",
        mime="text/csv",
    )


with tab4:
    st.header("Step 4: What We Learned")

    st.markdown(
        """
**About the method**
- K-Means found the groups **without labels** - that is unsupervised learning.
- We scaled features first, so PKR values don't dominate day counts.
- We chose K using two checks (elbow and silhouette), not by guessing.
- Names came **after** studying the groups, never before.
        """
    )

    if selected_k == 3:
        top = segment_summary.iloc[0]
        bottom = segment_summary.iloc[-1]
        top_share = revenue.loc[revenue["Segment"] == "High Value Loyal", "Share_%"].sum()
        top_size = (view["Segment"] == "High Value Loyal").mean() * 100

        st.markdown("**About the business**")
        st.markdown(
            f"- **High Value Loyal** customers are **{top_size:.0f}%** of customers "
            f"but bring **{top_share:.0f}%** of monthly revenue.\n"
            f"- They place about **{top['Avg_Monthly_Orders']:.0f}** orders/month, "
            f"vs **{bottom['Avg_Monthly_Orders']:.0f}** for Low Engagement.\n"
            f"- Low Engagement customers pay later "
            f"(**{bottom['Avg_Payment_Delay']:.0f}** days vs **{top['Avg_Payment_Delay']:.0f}**) "
            f"and return more goods."
        )

        st.subheader("⚠️ Customers to call this week")
        st.write("Low Engagement customers who have not ordered in over 45 days.")
        at_risk = (
            view[(view["Segment"] == "Low Engagement") & (view["Days_Since_Last_Order"] > 45)]
            .sort_values("Days_Since_Last_Order", ascending=False)
            [["Customer_ID", "Outlet_Name", "City", "Customer_Type",
              "Days_Since_Last_Order", "Payment_Delay_Days", "Monthly_Spend_PKR"]]
        )
        st.metric("At-risk customers", len(at_risk))
        st.dataframe(at_risk, width="stretch", hide_index=True)
    else:
        st.info("Set K = 3 in the sidebar to see the business findings and the at-risk list.")

    st.markdown(
        """
**Limits to remember**
- The dataset is synthetic, so groups are cleaner than real sales data.
- K-Means assumes roughly round clusters and is sensitive to outliers.
- Re-train regularly, because customer behaviour changes.
        """
    )



