# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 03:40:41 2021

@author: destine
"""

import streamlit as st
import pandas as pd
import io
import numpy as np
import plotly.express as px
from pathlib import Path
import base64
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score


def img_to_bytes(img_path):
    """
        Return image in order to be used 
        in a markdown component. 
      
    """
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded  

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a style="font-size: 10px; color: purple; text-decoration: none;" href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

### Title of the app        
st.markdown("""
            <h1 style="font-size: 25px; color:purple; text-align: center;" >Churn Prediction App</h1>
            """, 
            unsafe_allow_html=True)

header_html = """<div style="text-align: center;"> <img src='data:image/png;base64,{}' class='img-fluid'>
                <h6 style="font-size: 10px;">Welcome in this churn prediction app, you gonna run different steps of the churn prediction. Check sidebar and have good experience!</h6>
 </div><br><br>""".format(
    img_to_bytes("retent.gif")
)
 
### image & description
st.markdown(
    header_html, unsafe_allow_html=True,
)
 
st.sidebar.markdown("""
            <h2 style="font-size: 15px;">Navigation</h2>
            """,
            unsafe_allow_html=True)

### github  link
st.sidebar.markdown("""
            <h2 style="font-size: 10px;">check out the <a href="https://github.com/destoone/Data_projects/blob/master/TP-Churn.ipynb" style="color: purple; text-decoration: none;"> NoteBook</a></h2>
            """,
            unsafe_allow_html=True)

def navigue():
    """    
    This function group all of the features and sidebar components, 
    like uploaded the file csv, the elements for the visualization,
    the differents model for our predictions.
    
    """
    
    uploaded= st.sidebar.file_uploader("upload", type='csv') 
    if uploaded:
        df= pd.read_csv(uploaded)
        st.sidebar.markdown("""
            <h2 style="font-size: 15px;">Frame</h2>
            """,
            unsafe_allow_html=True)
        if st.sidebar.button("Display DataFrame"):
            st.write(df.head(10))
            st.write(df.shape)
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue() 
            st.write(s) 
        if st.sidebar.button("Statistics"):
            st.write(df.describe(include= "all"))
            
        st.sidebar.markdown("""
            <h2 style="font-size: 15px;">Visualizing</h2>
            """,
            unsafe_allow_html=True)
        columns= st.sidebar.multiselect(
                "Choose until 2 columns",
                df.columns)    
        if len(columns)<=2 and len(columns)>0:
            chart= st.sidebar.selectbox(
                "choose the chart",
                ["line","histogram","bar","scatter","map"]) 
            if len(columns)==2:
                title= st.sidebar.text_input("Enter a title")
                if (chart=="line" and title and df[columns[0]].dtypes=="int64" and df[columns[1]].dtypes=="int64") or (chart=="line" and title and df[columns[0]].dtypes=="float64" and df[columns[1]].dtypes=="float64"):
                    fig = px.line(df, x=columns[0], y=columns[1], title=title, width=620, height=420)
                    st.plotly_chart(fig)
                if chart=="bar" and title:
                    fig = px.bar(df, x=columns[0], y=columns[1], title=title, width=620, height=420)
                    st.plotly_chart(fig)
                if (chart=="scatter" and title and df[columns[0]].dtypes=="int64" and df[columns[1]].dtypes=="int64") or (chart=="scatter" and title and df[columns[0]].dtypes=="float64" and df[columns[1]].dtypes=="float64"):
                    fig = px.scatter(df, x=columns[0], y=columns[1], title=title, width=620, height=420)
                    st.plotly_chart(fig)
                if chart=="map" and title:
                    st.sidebar.warning("Make sure your are getting longitude and latitude columns.")
                    st.sidebar.warning(columns[0]+": latitude and "+columns[1]+": longitude")
                    name= st.sidebar.selectbox(
                            "choose one column",
                            df.columns)
                    if name:
                        fig = px.scatter_mapbox(df, lat=columns[0], lon=columns[1], color=name,
                          color_continuous_scale=px.colors.cyclical.IceFire, size_max=20, zoom=3, 
                          mapbox_style="carto-positron", width=620, height=420)
                        st.plotly_chart(fig)
                    
            if len(columns)==1:
                title= st.sidebar.text_input("Enter a title")
                if chart=="histogram" and title:
                    bins= st.sidebar.number_input("Enter bins number")
                    fig = px.histogram(df, x=columns[0], nbins=int(bins), title=title, width=620, height=420)
                    st.plotly_chart(fig)
            
        else:
            st.sidebar.error("Sorry! you have making no choice or more than 2.")
        
        st.sidebar.markdown("""
            <h2 style="font-size: 15px;">Distinguish of the columns</h2>
            """,
            unsafe_allow_html=True)
        target= st.sidebar.selectbox(
                "Target",
                df.columns)
        if 0 in df[target].unique() and 1 in df[target].unique():
            st.sidebar.success("Nice! You have get the good choice.")
        else:
            st.sidebar.error("Sorry! column it's not appropriated.")

        features = st.sidebar.multiselect(
                "Features",
                df.select_dtypes(include="number").drop(target,axis=1).columns)
        
        st.sidebar.markdown("""
            <h2 style="font-size: 15px;">Split Dataframe</h2>
            """,
            unsafe_allow_html=True)
        length= st.sidebar.slider("Train length",0.0, 100.0, (0.0,80.0))
        if length[1]<=80.0 and length[1]>=60.0:
            st.sidebar.success("Nice! so the remain length will be your test")
        if length[1]>80.0 or length[1]<60.0:
            st.sidebar.error("Error! you need to get less than or equal to 80 or more than or equal to")
        cv= st.sidebar.selectbox(
                "Cross Validation on the train",
                [0,5,10,15,20])

        model= st.sidebar.selectbox(
                "Which model do you like!",
                ["Decision Tree",
                 "Random Forest",
                 "KnnClassifier",
                 "Logistic Regression",
                 "SVClassification"
                    ])
        if model=="Decision Tree":
            params= ["criterion","max_depth","max_features","min_samples_leaf","min_samples_split"]                  
            check_param = [st.sidebar.checkbox(param, key=param) for param in params]
            criterion,max_depth,max_features,min_samples_leaf,min_samples_split= "gini",None,None,1,2
            for p in range(len(params)):
                if check_param[p] and params[p]=="criterion":
                    criterion= st.sidebar.selectbox(
                            "enter criterion value",
                            ["gini", "entropy"]
                            )
                if check_param[p] and params[p]=="max_depth":
                    max_depth= st.sidebar.selectbox(
                            "enter max_depth value",
                            [None,2,5,10,15]
                            )
                if check_param[p] and params[p]=="max_features":
                    max_features= st.sidebar.selectbox(
                            "enter max_features value",
                            [None,"auto", "sqrt", "log2"]
                            )
                if check_param[p] and params[p]=="min_samples_leaf":
                    min_samples_leaf= st.sidebar.selectbox(
                            "enter min_samples_leaf value",
                            [1, 5, 8, 12]
                            )
                if check_param[p] and params[p]=="min_samples_split":
                    min_samples_split= st.sidebar.selectbox(
                            "enter min_samples_split value",
                            [2, 3, 5, 8]
                            )
            if st.sidebar.button("Predicting"):
                dt= DecisionTreeClassifier(random_state=0,criterion=criterion,max_depth=max_depth,max_features=max_features,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split)
                predictions,predictions_p,accuracy,f_score,p,r,ras,accuracy_cv= core(df,features,target,dt,cv,length)
                df_t,test= view(df,target,length,predictions,predictions_p)
                tab= pd.DataFrame({"accuracy": [accuracy], "f1_score": [f_score],
                                   "precision_score": [p], "recall_score": [p],
                                   "roc_auc_score": [ras], "accuracy_cross_validation": [accuracy_cv]})
                tab.index = [""] * len(tab)
                st.markdown("""
                <h2 style="font-size: 15px; text-decoration-line: underline;">Differents metrics</h2>
                """,
                unsafe_allow_html=True)
                
                st.table(tab)
                
                st.markdown("""
                <h2 style="font-size: 15px; text-decoration-line: underline;">Calcul of your retention and churn rate</h2>
                """,
                unsafe_allow_html=True)
                retention= (len(df_t.loc[df_t["predictions"]==0,"predictions"])/len(df_t))*100
                churn= (len(df_t.loc[df_t["predictions"]==1,"predictions"])/len(df_t))*100
                st.write("Retention rate: "+str(retention)+"%")
                st.write("Churn rate: "+str(churn)+"%")
                df_t[test.columns]= test[test.columns]                

                st.sidebar.markdown(download_link(df_t, "result.csv", "Download predicting results"), unsafe_allow_html=True)
            
        if model=="Random Forest":
            params= ["n_estimators","criterion","max_depth","max_features","min_samples_leaf","min_samples_split"]                  
            check_param = [st.sidebar.checkbox(param, key=param) for param in params]
            n_estimators,criterion,max_depth,max_features,min_samples_leaf,min_samples_split= 100,"gini",None,None,1,2
            for p in range(len(params)):
                if check_param[p] and params[p]=="n_estimators":
                    n_estimators= st.sidebar.selectbox(
                            "enter n_estimators value",
                            [100, 4, 6, 9]
                            )
                if check_param[p] and params[p]=="criterion":
                    criterion= st.sidebar.selectbox(
                            "enter criterion value",
                            ["gini", "entropy"]
                            )
                if check_param[p] and params[p]=="max_depth":
                    max_depth= st.sidebar.selectbox(
                            "enter max_depth value",
                            [None,2,5,10,15]
                            )
                if check_param[p] and params[p]=="max_features":
                    max_features= st.sidebar.selectbox(
                            "enter max_features value",
                            [None,"auto", "sqrt", "log2"]
                            )
                if check_param[p] and params[p]=="min_samples_leaf":
                    min_samples_leaf= st.sidebar.selectbox(
                            "enter min_samples_leaf value",
                            [1, 5, 8, 12]
                            )
                if check_param[p] and params[p]=="min_samples_split":
                    min_samples_split= st.sidebar.selectbox(
                            "enter min_samples_split value",
                            [2, 3, 5, 8]
                            )
            if st.sidebar.button("Predicting"):
                rf= RandomForestClassifier(random_state=0,n_estimators=n_estimators,criterion=criterion,max_depth=max_depth,max_features=max_features,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split)
                predictions,predictions_p,accuracy,f_score,p,r,ras,accuracy_cv= core(df,features,target,rf,cv,length)
                df_t,test= view(df,target,length,predictions,predictions_p)
                tab= pd.DataFrame({"accuracy": [accuracy], "f1_score": [f_score],
                                   "precision_score": [p], "recall_score": [p],
                                   "roc_auc_score": [ras], "accuracy_cross_validation": [accuracy_cv]})
                tab.index = [""] * len(tab)
                st.markdown("""
                <h2 style="font-size: 15px; text-decoration-line: underline;">Differents metrics</h2>
                """,
                unsafe_allow_html=True)
                
                st.table(tab)
                
                st.markdown("""
                <h2 style="font-size: 15px; text-decoration-line: underline;">Calcul of your retention and churn rate</h2>
                """,
                unsafe_allow_html=True)
                retention= (len(df_t.loc[df_t["predictions"]==0,"predictions"])/len(df_t))*100
                churn= (len(df_t.loc[df_t["predictions"]==1,"predictions"])/len(df_t))*100
                st.write("Retention rate: "+str(retention)+"%")
                st.write("Churn rate: "+str(churn)+"%")   
                df_t[test.columns]= test[test.columns]

                st.sidebar.markdown(download_link(df_t, "result.csv", "Download predicting results"), unsafe_allow_html=True)
            
        if model=="KnnClassifier":
            params= ["n_neighbors","weights","algorithm"]                  
            check_param = [st.sidebar.checkbox(param, key=param) for param in params]
            n_neighbors,weights,algorithm= 5,"uniform","auto"
            for p in range(len(params)):
                if check_param[p] and params[p]=="n_neighbors":
                    n_neighbors= st.sidebar.selectbox(
                            "enter n_neighbors value",
                            [5,10,15,20,25]
                            )
                if check_param[p] and params[p]=="weights":
                    weights= st.sidebar.selectbox(
                            "enter weights value",
                            ["uniform", "distance"]
                            )
                if check_param[p] and params[p]=="algorithm":
                    algorithm= st.sidebar.selectbox(
                            "enter algorithm value",
                            ["auto", "ball_tree", "kd_tree", "brute"]
                            )
            if st.sidebar.button("Predicting"):
                knn= KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm)
                predictions,predictions_p,accuracy,f_score,p,r,ras,accuracy_cv= core(df,features,target,knn,cv,length)
                df_t,test= view(df,target,length,predictions,predictions_p)
                tab= pd.DataFrame({"accuracy": [accuracy], "f1_score": [f_score],
                                   "precision_score": [p], "recall_score": [p],
                                   "roc_auc_score": [ras], "accuracy_cross_validation": [accuracy_cv]})
                tab.index = [""] * len(tab)
                st.markdown("""
                <h2 style="font-size: 15px; text-decoration-line: underline;">Differents metrics</h2>
                """,
                unsafe_allow_html=True)
                
                st.table(tab)
                
                st.markdown("""
                <h2 style="font-size: 15px; text-decoration-line: underline;">Calcul of your retention and churn rate</h2>
                """,
                unsafe_allow_html=True)
                retention= (len(df_t.loc[df_t["predictions"]==0,"predictions"])/len(df_t))*100
                churn= (len(df_t.loc[df_t["predictions"]==1,"predictions"])/len(df_t))*100
                st.write("Retention rate: "+str(retention)+"%")
                st.write("Churn rate: "+str(churn)+"%") 
                df_t[test.columns]= test[test.columns]

                st.sidebar.markdown(download_link(df_t, "result.csv", "Download predicting results"), unsafe_allow_html=True)
            
        if model=="Logistic Regression":
            params= ["penalty","solver"]                  
            check_param = [st.sidebar.checkbox(param, key=param) for param in params]
            penalty,solver= "l2","lbfgs"
            for p in range(len(params)):
                if check_param[p] and params[p]=="penalty":
                    penalty= st.sidebar.selectbox(
                            "enter penalty value",
                            ["l2", "l1", "none"]
                            )
                if check_param[p] and params[p]=="solver":
                    solver= st.sidebar.selectbox(
                            "enter solver value",
                            ["lbfgs", "newton-cg", "liblinear", "sag", "saga"]
                            )
            if st.sidebar.button("Predicting"):
                lr= LogisticRegression(random_state=0,penalty=penalty,solver=solver)
                predictions,predictions_p,accuracy,f_score,p,r,ras,accuracy_cv= core(df,features,target,lr,cv,length)
                df_t,test= view(df,target,length,predictions,predictions_p)
                tab= pd.DataFrame({"accuracy": [accuracy], "f1_score": [f_score],
                                   "precision_score": [p], "recall_score": [p],
                                   "roc_auc_score": [ras], "accuracy_cross_validation": [accuracy_cv]})
                tab.index = [""] * len(tab)
                st.markdown("""
                <h2 style="font-size: 15px; text-decoration-line: underline;">Differents metrics</h2>
                """,
                unsafe_allow_html=True)
                
                st.table(tab)
                
                st.markdown("""
                <h2 style="font-size: 15px; text-decoration-line: underline;">Calcul of your retention and churn rate</h2>
                """,
                unsafe_allow_html=True)
                retention= (len(df_t.loc[df_t["predictions"]==0,"predictions"])/len(df_t))*100
                churn= (len(df_t.loc[df_t["predictions"]==1,"predictions"])/len(df_t))*100
                st.write("Retention rate: "+str(retention)+"%")
                st.write("Churn rate: "+str(churn)+"%") 
                df_t[test.columns]= test[test.columns]

                st.sidebar.markdown(download_link(df_t, "result.csv", "Download predicting results"), unsafe_allow_html=True)
            
        if model=="SVClassification":
            params= ["kernel","degree"]                  
            check_param = [st.sidebar.checkbox(param, key=param) for param in params]
            kernel,degree= "rbf",3
            for p in range(len(params)):
                if check_param[p] and params[p]=="kernel":
                    kernel= st.sidebar.selectbox(
                            "enter kernel value",
                            ["rbf", "poly", "sigmoid"]
                            )
                if check_param[p] and params[p]=="degree":
                    degree= st.sidebar.selectbox(
                            "enter degree value",
                            [3,6,9]
                            )
            if st.sidebar.button("Predicting"):
                sv= SVC(random_state=0,kernel=kernel,degree=degree,probability=True)
                predictions,predictions_p,accuracy,f_score,p,r,ras,accuracy_cv= core(df,features,target,sv,cv,length)
                df_t,test= view(df,target,length,predictions,predictions_p)
                tab= pd.DataFrame({"accuracy": [accuracy], "f1_score": [f_score],
                                   "precision_score": [p], "recall_score": [p],
                                   "roc_auc_score": [ras], "accuracy_cross_validation": [accuracy_cv]})
                tab.index = [""] * len(tab)
                st.markdown("""
                <h2 style="font-size: 15px; text-decoration-line: underline;">Differents metrics</h2>
                """,
                unsafe_allow_html=True)
                
                st.table(tab)
                
                st.markdown("""
                <h2 style="font-size: 15px; text-decoration-line: underline;">Calcul of your retention and churn rate</h2>
                """,
                unsafe_allow_html=True)
                retention= (len(df_t.loc[df_t["predictions"]==0,"predictions"])/len(df_t))*100
                churn= (len(df_t.loc[df_t["predictions"]==1,"predictions"])/len(df_t))*100
                st.write("Retention rate: "+str(retention)+"%")
                st.write("Churn rate: "+str(churn)+"%")  
                df_t[test.columns]= test[test.columns]

                st.sidebar.markdown(download_link(df_t, "result.csv", "Download predicting results"), unsafe_allow_html=True)
            
def core(df,features,target,model,cv,length):
    """
        Core function make training of data and return metrics.
        
    """
    
    train= df[:int(len(df)*(length[1]*0.01))]
    test= df[int(len(df)*(length[1]*0.01)):]
    
    model.fit(train[features],train[target])
    predictions= model.predict(test[features])
    predictions_p= model.predict_proba(test[features])
    accuracy= accuracy_score(test[target],predictions)
    f_score= f1_score(test[target],predictions,average="macro")
    p= precision_score(test[target],predictions, average="macro")
    r= recall_score(test[target],predictions, average="macro")
    ras= roc_auc_score(test[target],predictions_p[:,1])
    accuracy_cv=0
    if cv>0:
        scores= cross_validate(model,df[features],df[target],cv=cv)
        accuracy_cv= np.mean(scores["test_score"])
    return predictions,predictions_p,accuracy,f_score,p,r,ras,accuracy_cv
    
def view(df,target,length,predictions,predictions_p):
    """
        view function, display our predictions and return dataframe who 
        contains all results.
    """
    
    test= df[int(len(df)*(length[1]*0.01)):]
    df_t= pd.DataFrame({"actual": test[target],
                        "predictions": predictions,
                        "predictions_proba": predictions_p[:,1]}) 
    st.write(df_t)
    st.markdown("""
            <h6 style="font-size: 10px;">The column "predictions_proba" allows to determine the probability of success of the predicted value compared to 1.</h6>
            """, 
            unsafe_allow_html=True)

    labels = ['actual_1','predictions_1','actual_0','predictions_0']
    values = [len(df_t.loc[df_t["actual"]==1,"actual"]), len(df_t.loc[df_t["predictions"]==1,"predictions"]),
              len(df_t.loc[df_t["actual"]==0,"actual"]), len(df_t.loc[df_t["predictions"]==0,"predictions"])]

    fig = px.bar(x=labels, y=values,width=620, height=420, title="Actual and Predicted values of 0 and 1")
    fig.update_xaxes(title_text='values')
    fig.update_yaxes(title_text='number of values ​​present')
    st.plotly_chart(fig)
    return df_t,test
        
def main():
    navigue()
    
if __name__=="__main__":
    main()