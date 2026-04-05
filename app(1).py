import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter


st.set_page_config(page_title="AI Feature Analyzer", layout="wide")


st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
}
h1 {
    color: #ffffff !important;
    text-align: center;
}
[data-testid="stFileUploader"] {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
}
[data-testid="stDataFrame"] {
    background-color: #1c1f26;
    border-radius: 10px;
}
div[data-testid="metric-container"] {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


st.title(" AI Feature Request Analyzer")
st.write("Upload product feedback and get smart Solutions")


uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    requests = df.iloc[:, 0].tolist()

    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(requests)

    
    kmeans = KMeans(n_clusters=3)
    df['Cluster'] = kmeans.fit_predict(X)

    
    def impact(text):
        text = text.lower()
        if "security" in text or "payment" in text:
            return "High"
        elif "speed" in text or "performance" in text:
            return "Medium"
        return "Low"

    
    def difficulty(text):
        text = text.lower()
        if "ai" in text or "system" in text:
            return "Hard"
        elif "integration" in text:
            return "Medium"
        return "Easy"

    df['Impact'] = df.iloc[:, 0].apply(impact)
    df['Difficulty'] = df.iloc[:, 0].apply(difficulty)

    
    st.markdown("##  Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Requests", len(df))
    col2.metric("High Impact", sum(df['Impact'] == "High"))
    col3.metric("Hard Features", sum(df['Difficulty'] == "Hard"))

    
    cluster_names = {}
    stop_words = ["add", "improve", "enable", "for", "the", "to", "and", "of"]

    for i in range(3):
        group = df[df['Cluster'] == i]

        if group.empty:
            cluster_names[i] = f"Group {i}"
            continue

        texts = group.iloc[:, 0].tolist()
        words = " ".join(texts).lower().split()
        words = [w for w in words if w not in stop_words]

        if len(words) == 0:
            cluster_names[i] = f"Group {i}"
        else:
            common_words = Counter(words).most_common(2)
            group_name = " ".join([word for word, count in common_words])
            cluster_names[i] = group_name.title() + " Features"

    
    df['Cluster_Name'] = df['Cluster'].map(cluster_names)

    
    st.markdown("##  Cluster Distribution")
    chart_data = df['Cluster_Name'].value_counts()
    st.bar_chart(chart_data)

    st.markdown("##  Detailed Analysis")
    st.dataframe(df)

    
    st.markdown("##  Feature Groups")

    for i in range(3):
        group = df[df['Cluster'] == i]
        name = cluster_names[i]

        st.markdown(f"### 🔹 {name}")
        st.write(group.iloc[:, 0].tolist())

    
    st.markdown("##  Final Recommendations")

    high_impact = df[df['Impact'] == "High"]
    easy_tasks = df[df['Difficulty'] == "Easy"]
    hard_tasks = df[df['Difficulty'] == "Hard"]

    st.write("###  High Priority Features (Do First)")
    st.write(high_impact.iloc[:, 0].tolist())

    st.write("###  Easy To Implement ")
    st.write(easy_tasks.iloc[:, 0].tolist())

    st.write("###  Complex Features (Plan Carefully)")
    st.write(hard_tasks.iloc[:, 0].tolist())

    

    
    