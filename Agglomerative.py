import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import pairwise_distances

class HierarchicalClustering:
    def __init__(self, k):
        self.k = k

    def fit(self, data, distances):
        n = len(data)
        # Initialize clusters
        clusters = [[i] for i in range(n)]

        # Hierarchical clustering algorithm
        while len(clusters) > self.k:
            min_dist = np.inf
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    cluster1 = clusters[i]
                    cluster2 = clusters[j]
                    avg_dist = 0
                    count = 0
                    for idx1 in cluster1:
                        for idx2 in cluster2:
                            if not np.isnan(distances[idx1, idx2]):
                                avg_dist += distances[idx1, idx2]
                                count += 1
                    if count > 0:
                        avg_dist /= count
                        if avg_dist < min_dist:
                            min_dist = avg_dist
                            merge_index = (i, j)
            i, j = merge_index
            new_cluster = clusters[i] + clusters[j]
            clusters[i] = new_cluster
            clusters.pop(j)

        self.labels_ = np.zeros(n)
        for i, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels_[idx] = i

def euclidean_distance(x1, x2):
    x1 = np.array(x1, dtype=np.float64)
    x2 = np.array(x2, dtype=np.float64)
    if np.any(np.isnan(x1)) or np.any(np.isnan(x2)):
        return np.nan
    return np.sqrt(np.sum((x1 - x2) ** 2))

def city_block_distance(x1, x2):
    x1 = np.array(x1, dtype=np.float64)
    x2 = np.array(x2, dtype=np.float64)
    if np.any(np.isnan(x1)) or np.any(np.isnan(x2)):
        return np.nan
    return np.sum(np.abs(x1 - x2))

def home_pageCAH():
    st.title("Hierarchical Ascendant Classification")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        file_path = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

    with col2:
        k = st.number_input("Choose K for threshold level :", min_value=2, max_value=10)

    # Select distance metric
    with col3:
        distance_metric = st.selectbox("Select Distance Metric:", ["Euclidean", "City Block"], index=0)

    with col4:
        run_button_style = """
            <style>
            .run-button {
                background-color: red;
                color: white;
                padding: 0.5rem 1rem;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                width : 152.5px;
                height : 40.39px;
                margin-top: 20px;
            }
            </style>
        """
        st.markdown(run_button_style, unsafe_allow_html=True)
        if st.button("Run Clustering"):
            if file_path is not None:
                try:
                    data = pd.read_excel(file_path, index_col=0)
                    if data.empty or data.isnull().values.any():
                        st.warning("The uploaded file is empty or contains missing values.")
                    else:
                        numeric_data = data.select_dtypes(include=np.number)
                        X = numeric_data.values
                        if len(X) < k:
                            st.warning("Number of clusters should be less than or equal to the number of data points.")
                        else:
                            if distance_metric == 'Euclidean':
                                distances = pairwise_distances(X, metric='euclidean')
                            elif distance_metric == 'City Block':
                                distances = pairwise_distances(X, metric='cityblock')
                            else:
                                st.error("Invalid distance metric selected. Choose either 'Euclidean' or 'City Block'.")
                                return
                            clustering_algorithm = HierarchicalClustering(k=k)
                            clustering_algorithm.fit(X, distances)
                            labels = clustering_algorithm.labels_
                            st.session_state['clustering_result'] = {'X': X, 'labels': labels}
                            st.session_state['page'] = "Dendrogram Visualization"
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

def dendrogram_visualization_page():
    st.title("Dendrogram Visualization")
    clustering_result = st.session_state.get('clustering_result')
    if clustering_result:
        X = clustering_result['X']
        labels = clustering_result['labels']
        distance_metric = st.session_state.get('distance_metric', 'euclidean').lower()
        if distance_metric == 'euclidean':
            Z = linkage(X, method='ward')
        elif distance_metric == 'city block':
            Z = linkage(X, method='average')
        else:
            st.error("Invalid distance metric selected. Choose either 'Euclidean' or 'City Block'.")
            return
        
        # Calculate threshold based on k
        k = len(np.unique(labels))
        if k <= 1:
            st.warning("Please choose k > 1.")
            return
        threshold = Z[-k, 2]  # Maximum distance between (k-1) and k clusters
        
        # Add a small offset to the threshold for better visualization
        offset = 0.05 * (max(Z[:, 2]) - min(Z[:, 2]))  # Adjust this factor as needed
        threshold += offset

        # Define color palette based on k
        if k <= 2:
            colors = ['tab:blue', 'tab:orange']
        else:
            colors = plt.cm.tab10(np.linspace(0, 1, k))
        
        # Plot dendrogram
        fig, ax = plt.subplots(figsize=(10, 5))
        dn = dendrogram(Z, ax=ax, color_threshold=threshold)

        # Change color of dendrogram based on labels
        for i, d in zip(dn['icoord'], dn['dcoord']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            ax.plot(x, y, color=colors[int(labels[int(i[1]-5)//10])])

        # Draw threshold line with adjusted position
        ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        
        ax.set_title("Dendrogram")
        ax.set_xlabel("Data Points")
        ax.set_ylabel("Distance")
        ax.legend()  # Show legend
        
        # Display the plot
        st.pyplot(fig)

        # Calculate inter-cluster distance
        inter_cluster_distances = np.diff(Z[:, 2])

        # Calculate intra-cluster distance
        intra_cluster_distances = []
        for i, d in zip(dn['icoord'], dn['dcoord']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            cluster_label = labels[int(i[1] - 5) // 10]  # Assuming dendrogram plot settings
            cluster_indices = np.where(labels == cluster_label)[0]
            cluster_points = X[cluster_indices]
            centroid = np.mean(cluster_points, axis=0)
            intra_cluster_distance = np.mean(np.linalg.norm(cluster_points - centroid, axis=1))
            intra_cluster_distances.append(intra_cluster_distance)

        # Display statistics as horizontal tables side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Inter-Cluster Distances:")
            inter_cluster_table = pd.DataFrame({"Distance": inter_cluster_distances})
            st.table(inter_cluster_table)
        
        with col2:
            st.write("Intra-Cluster Distances:")
            intra_cluster_table = pd.DataFrame({"Distance": intra_cluster_distances})
            st.table(intra_cluster_table)
        
    else:
        st.warning("Please run clustering on the Home page first.")

def contact_pageCAH():
    st.title("Contact")

    # LinkedIn profile link
    linkedin_link = "https://www.linkedin.com/in/wassim-kerdoun-494a89246/"
    st.markdown("""
    <div style="background-color:#242526; border-radius:5px; padding:10px">
        <b>LinkedIn Profile Link:</b>
        <a href="{0}" target="_blank">{0}</a>
    </div>
    """.format(linkedin_link), unsafe_allow_html=True)
    
    # Text input for user's GitHub profile link
    github_link = "https://github.com/Riemann222"
    st.markdown("""
    <div style="background-color:#242526; border-radius:5px; padding:10px">
        <b>GitHub Profile Link:</b>
        <a href="{0}" target="_blank">{0}</a>
    </div>
    """.format(github_link), unsafe_allow_html=True)
    
    documentation_link = "https://github.com/Riemann222"
    st.markdown("""
    <div style="background-color:#242526; border-radius:5px; padding:10px">
        <b>Documentation file:</b>
        <a href="{0}" target="_blank">{0}</a>
    </div>
    """.format(documentation_link), unsafe_allow_html=True)

