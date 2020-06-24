# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 17:05:25 2020
A ploting experiment with clustering algorithm. Worked with bokeh library
@author: Tanumay Misra
"""
# agglomerative clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot
import streamlit as st

def plot(cluster_type):
# define dataset
    X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
    # define the model
    if cluster_type == 'Agglo':
        model = AgglomerativeClustering(n_clusters=2)
    if cluster_type == 'KMeans':
        model = KMeans(n_clusters=2)
    if cluster_type == 'Gaussian':
        model = GaussianMixture(n_components=2)
    
    
    # fit model and predict clusters
    yhat = model.fit_predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    return(X,yhat)
  
def main():
    st.title('Clustering Plot')
    html_temp = """<p style="color:green">Dataset make classification !!!.</p>"""
    st.markdown(html_temp,unsafe_allow_html = True)
    cluster_name = st.sidebar.selectbox('select cluster type',('Agglo','KMeans','Gaussian'))
        
    if cluster_name == 'Agglo':
        X,yhat = plot(cluster_name)
    if cluster_name == 'KMeans':
        X,yhat = plot(cluster_name)        
    if cluster_name == 'Gaussian':
        X,yhat = plot(cluster_name)
        
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])   
    st.pyplot()
        
    
if __name__=='__main__':
    main()