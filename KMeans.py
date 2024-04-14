import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
import base64
from io import BytesIO

def home_page():
    st.title("K-Means clustering")
    # Define column layout
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    # Button to import data
    with col1:
        file_path = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

    # Choose K for KMeans
    with col2:
        k = st.number_input("Choose K for K-Means:", min_value=1, max_value=10, value=3)

    # Select distance metric
    with col3:
        distance_metric = st.selectbox("Select Distance Metric:", ["Euclidean", "City Block"], index=0)

    # Red "Run Clustering" button
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
                margin-top: 20px; /* Adjust the margin-top to align with other buttons */
            }
            </style>
        """
        st.markdown(run_button_style, unsafe_allow_html=True)
        if st.button("Run Clustering"):
            if file_path is not None:
                try:
                    data = pd.read_excel(file_path, index_col=0)
                    if data.empty:
                        st.warning("The uploaded file is empty.")
                    else:
                        X = data.values
                        X = X[:, 2:]  # Exclude non-numeric columns if necessary
                        if len(X) < k:
                            st.warning("Number of clusters should be less than or equal to the number of data points.")
                        else:
                            # Standardize data
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)

                            # Perform clustering
                            kmeans = KMeansClustering(k=k, distance_metric=distance_metric.lower(), random_state=42)
                            labels, centroids, inertia_history = kmeans.fit(X_scaled)

                            # Store clustering result in session state
                            st.session_state['cluster_data'] = {
                                'X_scaled': X_scaled,
                                'labels': labels,
                                'centroids': centroids,
                                'inertia_history': inertia_history,
                                'k': k  # Add 'k' to session state
                            }

                            # Redirect to clustering page
                            st.session_state['page'] = "Clustering Plot"

                except Exception as e:
                    st.error(f"Error: {str(e)}")

def plot_inertia(inertia):
    # Plot the inertia values against the number of clusters (K)
    fig = go.Figure(data=go.Scatter(x=np.arange(1, len(inertia) + 1), y=inertia, mode='lines+markers'))
    fig.update_layout(title="Elbow Method: Inertia vs Number of Clusters",
                      xaxis_title="Number of Clusters (K)",
                      yaxis_title="Inertia",
                      )
    fig.update_traces(line=dict(color='#eb4034'))
    return fig

def export_data_to_csv(data, file_name="exported_data.csv"):
    """
    Export data to a CSV file.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data to be exported.
        file_name (str): The name of the CSV file to be exported. Default is "exported_data.csv".
    """
    csv_file = data.to_csv(index=False)
    b64 = base64.b64encode(csv_file.encode()).decode()  # Encode to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download {file_name}</a>'
    st.markdown(href, unsafe_allow_html=True)

def plot_clustering_page(cluster_data):
    st.title("Clustering Plot")
    if cluster_data:
        X_scaled = cluster_data['X_scaled']
        labels = cluster_data['labels']
        centroids = cluster_data['centroids']
        plot_result(X_scaled, labels, centroids)
        # Add export button
        if st.button("Export Data to CSV"):
            if cluster_data is not None:
                X_scaled_df = pd.DataFrame(X_scaled)
                file_name = "clustered_data.csv"  # Fixed name for the exported file
                export_data_to_csv(X_scaled_df, file_name=file_name)
    else:
        st.info("Please run clustering on the Home page first.")

def plot_elbow_method():
    st.title("Elbow Method")
    cluster_data = st.session_state.get('cluster_data')
    if cluster_data:
        X_scaled = cluster_data['X_scaled']
        inertia = calculate_inertia(X_scaled)
        st.plotly_chart(plot_inertia(inertia), use_container_width=True)
    else:
        st.info("Please run clustering on the Home page first.")

def plot_convergence(cluster_data):
    st.title("Convergence Plot")
    if cluster_data:
        inertia_history = cluster_data['inertia_history']
        convergence_iteration = len(inertia_history)
        final_inertia = inertia_history[-1]
        fig = go.Figure(data=go.Scatter(x=np.arange(1, len(inertia_history) + 1), y=inertia_history, mode='lines+markers'))
        fig.update_layout(title="Convergence: Inertia vs Iterations",
                          xaxis_title="Iterations",
                          yaxis_title="Inertia",
                          )
        
        fig.update_traces(line=dict(color='#eb4034'))

        # Display additional information
        st.plotly_chart(fig)
        st.info(f"Convergence reached at iteration {convergence_iteration}.")
        st.info(f"Final inertia: {final_inertia}")

    else:
        st.info("Please run clustering on the Home page first.")

def more_features_page():
    # Navigation bar specific to More Features page
    def more_features_navigation_bar():
        selected = option_menu(
            menu_title=None,
            options=["PCA dimensionality reduction", "Elbow method", "Convergence Plot", "K-Means++"],
            icons=['', '', '', ''],
            menu_icon="chart-network",
            orientation="vertical",
            default_index=0,
            styles={
                "nav-link": {
                    "text-align": "left",
                    "--hover-color": "#eee",
                }
            }
        )
        return selected

    # Display the More Features navigation bar
    more_features_option = more_features_navigation_bar()

    # Based on the selected option, display corresponding content
    if more_features_option == "PCA dimensionality reduction":
        st.title("PCA dimensionality reduction")
        # Get the clustered data from session state
        cluster_data = st.session_state.get('cluster_data')
        if cluster_data:
            X_scaled = cluster_data['X_scaled']
            labels = cluster_data['labels']
            centroids = cluster_data['centroids']

            # Apply PCA dimensionality reduction
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            centroids_pca = pca.transform(centroids)  # Transform centroids into PCA space

            # Plot the clustered data with PCA dimensionality reduction
            plot_result(X_pca, labels, centroids_pca, display_statistics=False)

        else:
            st.warning("Please run clustering on the Home page first.")

    elif more_features_option == "Elbow method":
        plot_elbow_method()

    elif more_features_option == "Convergence Plot":
        cluster_data = st.session_state.get('cluster_data')
        plot_convergence(cluster_data)

    elif more_features_option == "K-Means++":
        kmeans_plusplus_page()

def kmeans_plusplus_page():
    st.title("K-Means++ Initialization")
    
    # Get the clustered data from session state
    cluster_data = st.session_state.get('cluster_data')
    
    if cluster_data:
        X_scaled = cluster_data['X_scaled']
        labels = cluster_data['labels']
        centroids = cluster_data['centroids']
        
        # Get the desired number of centroids (k)
        k = len(centroids)
        
        # Initialize centroids using k-means++
        kmeans_plusplus = KMeansClustering(k=k, distance_metric='euclidean', random_state=42)
        kmeans_plusplus_centroids = kmeans_plusplus._initialize_centroids(X_scaled)
        
        # Plot only the k-means++ centroids
        plot_result(X_scaled, labels, kmeans_plusplus_centroids, display_statistics=False)
        
    else:
        st.warning("Please run clustering on the Home page first.")



def contact_page():
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

def navigation_bar():
    with st.container():
        selected = option_menu(
            menu_title=None,
            options=["Home", "Clustering Plot", "More Features", 'Contact'],
            icons=['house', 'graph-up arrow', "list-task", 'phone'],
            menu_icon="cast",
            orientation="horizontal",
            default_index=0,
            styles={
                "nav-link": {
                    "text-align": "left",
                    "--hover-color": "#eee",
                }
            }
        )
        return selected

class KMeansClustering:
    def __init__(self, k=3, distance_metric='euclidean', random_state=None):
        self.k = k
        self.distance_metric = distance_metric
        self.centroids = None
        self.random_state = random_state
        self.inertia_history = []

    def fit(self, X, max_iterations=200):
        np.random.seed(self.random_state)  # Set the random seed

        self.centroids = self._initialize_centroids(X)  # Initialize centroids using k-means++

        for iteration in range(max_iterations):
            # Assign each data point to the nearest centroid
            distances = self.calculate_distances(X)
            cluster_assignment = np.argmin(distances, axis=0)

            # Update centroids
            new_centroids = np.array([X[cluster_assignment == i].mean(axis=0) for i in range(self.k)])

            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids
            
            # Calculate inertia and store it in history
            inertia = np.sum(np.min(distances, axis=0))
            self.inertia_history.append(inertia)

        return cluster_assignment, self.centroids, self.inertia_history

    def calculate_distances(self, X):
        if self.distance_metric == 'euclidean':
            return np.vstack([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids])
        elif self.distance_metric == 'city_block':
            return np.vstack([np.sum(np.abs(centroid - X), axis=1) for centroid in self.centroids])
        else:
            raise ValueError("Invalid distance metric specified.")
    
    def _initialize_centroids(self, X):
        # Randomly select the first centroid
        centroids = [X[np.random.randint(X.shape[0])]]

        # Select the remaining centroids using k-means++ initialization
        for _ in range(1, self.k):
            # Calculate squared distances from each data point to the nearest centroid
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids]) for x in X])
            # Choose new centroid from data points with probability proportional to squared distance
            probabilities = distances / distances.sum()
            centroids.append(X[np.random.choice(len(X), p=probabilities)])

        return np.array(centroids)

def calculate_inertia(X, max_clusters=10, distance_metric='euclidean'):
    inertia = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeansClustering(k=k, distance_metric=distance_metric, random_state=42)
        labels, _, _ = kmeans.fit(X)
        centroids = kmeans.centroids
        inertia.append(np.sum((X - centroids[labels]) ** 2))
    return inertia

def plot_result(X, labels, centroids, display_statistics=True):
    # Define colors for clusters
    colors = ['rgba(31, 119, 180, 0.5)', 'rgba(255, 127, 14, 0.5)', 'rgba(44, 160, 44, 0.5)', 'rgba(214, 39, 40, 0.5)',
              'rgba(148, 103, 189, 0.5)', 'rgba(140, 86, 75, 0.5)', 'rgba(227, 119, 194, 0.5)', 'rgba(127, 127, 127, 0.5)',
              'rgba(188, 189, 34, 0.5)', 'rgba(23, 190, 207, 0.5)']

    # Create trace for each cluster
    traces = []
    cluster_statistics = {}
    for i, cluster in enumerate(sorted(np.unique(labels))):
        data = X[labels == cluster]
        trace = go.Scatter(
            x=data[:, 0],
            y=data[:, 1],
            mode='markers',
            name=f'Cluster {cluster}',
            marker=dict(size=10, color=colors[i]),
            line=dict(color=colors[i].replace('0.5', '1'), width=2)
        )
        traces.append(trace)

        # Calculate statistics for the cluster
        cluster_size = len(data)
        cluster_statistics[f'Cluster {cluster}'] = cluster_size

    # Add centroids trace
    centroid_trace = go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode='markers',
        name='Centroids',
        marker=dict(size=10, color='white', symbol='star'),
    )
    traces.append(centroid_trace)

    # Create layout with wider parameters
    layout = go.Layout(
        title='K-Means Clustering',
        xaxis=dict(title='Component 1'),
        yaxis=dict(title='Component 2'),
        hovermode='closest',
        width=1085,  # Set the width to 1200 pixels
        height=457.59,  # Set the height to 800 pixels
    )

    # Create figure
    fig = go.Figure(data=traces, layout=layout)

    # Display figure
    st.plotly_chart(fig)

    # Display cluster statistics if requested
    if display_statistics:
        st.subheader("Cluster Statistics")
        for cluster, size in cluster_statistics.items():
            st.info(f"**{cluster}**: {size} points")
        # Display distance between centroids
        st.subheader("Distance Between Centroids")
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                distance = np.linalg.norm(centroids[i] - centroids[j])
                st.info(f"Distance between Centroid {i} and Centroid {j}: {distance:.2f}")

