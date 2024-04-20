import streamlit as st
from Agglomerative import *
from KMeans import *

def main():
    # Define the styled option menu
    with st.container():
        clustering_method = st.session_state.get('clustering_method', "KMeans")  # Default to KMeans
        if clustering_method == "KMeans":
            navigation_option = option_menu(None, ["Home", "Clustering Plot", "More Features", "Contact"], orientation="horizontal")
        elif clustering_method == "Agglomerative":
            navigation_option = option_menu(None, ["Home", "Dendrogram Visualization", "Contact"], orientation="horizontal")

    # Define the sidebar
    st.sidebar.title("Sidebar")
    clustering_method = st.sidebar.radio("Select Clustering Method", ["KMeans", "Agglomerative"])

    # Update session state with selected clustering method
    st.session_state['clustering_method'] = clustering_method

    # Add custom CSS to set the background image
    st.markdown(
        f"""
        <style>
        body {{
            background-image: url("https://downloader.la/temp/[Downloader.la]-661b3593b5717.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Content based on selected option
    if clustering_method == "KMeans":
        # Content related to KMeans clustering
        if navigation_option == "Home":
            home_page()
        elif navigation_option == "Clustering Plot":
            cluster_data = st.session_state.get('cluster_data')
            plot_clustering_page(cluster_data)
        elif navigation_option == "More Features":
            more_features_page()
        elif navigation_option == "Contact":
            contact_page()
    elif clustering_method == "Agglomerative":
        if navigation_option == "Home":
            home_pageCAH()
        elif navigation_option == "Dendrogram Visualization":
            dendrogram_visualization_page()
        elif navigation_option == "Contact":
            contact_pageCAH()

if __name__ == "__main__":
    main()
