import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from youtubeData_v3 import download_comments_and_content
from comments_downlod_to_hate_module import get_ham_comments

from sklearn.preprocessing import StandardScaler
import scipy.stats as stat

from dataframe_modify import data_modification


app = Flask(__name__)
model = pickle.load(open('gboostv1.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    id = [x for x in request.form.values()]
    val = id[0] 
    str_val = str(val)
    download_comments_and_content(str_val) # gmsUIoMSgsY
    
    df = pd.read_csv('data.csv')
    df.loc[len(df.index)] = ['kkkkkkkkkk',0.2426999584083827,0.2337711999415263,7,0.1428571428571429,1,39,1,1,0.0,4550827.0,0.0,1,1,1,0.0]
    df.loc[len(df.index)] = ['kkkkkkkkkk',0.2426999584083827,0.2337711999415263,7,0.1428571428571429,1,39,2,0,0.0,4550827.0,0.0,0,0,1,0.0]
    df.loc[len(df.index)] = ['kkkkkkkkkk',0.2426999584083827,0.2337711999415263,7,0.1428571428571429,1,39,3,0,0.0,4550827.0,0.0,0,0,1,0.0]
    df.loc[len(df.index)] = ['kkkkkkkkkk',0.2426999584083827,0.2337711999415263,7,0.1428571428571429,1,39,4,0,0.0,4550827.0,0.0,0,0,1,0.0]
    df.loc[len(df.index)] = ['kkkkkkkkkk',0.2426999584083827,0.2337711999415263,7,0.1428571428571429,1,39,5,0,0.0,4550827.0,0.0,0,0,1,0.0]
    df.loc[len(df.index)] = ['kkkkkkkkkk',0.2426999584083827,0.2337711999415263,7,0.1428571428571429,1,39,6,0,0.0,4550827.0,0.0,0,0,1,0.0]
    data_frame = data_modification(df)
    data_frame = data_frame.drop(data_frame.index[-6:-1])
    data_frame = data_frame.drop(data_frame.index[-1])
    ldd= data_frame.columns
    #X = df.drop('cid',axis =1)
    
    
    #scaler = StandardScaler()
    #scaler.fit(df)
    #X_scaled = scaler.transform(X)
    #input_list = df.values.tolist()
    
    prediction = model.predict(data_frame.values.tolist())
    print(prediction)
    #output = round(prediction[0], 2)
    #x = [-0.501935,-0.0559087,-0.0161989,-0.297034,-0.601665,-0.235129,-0.267696,-0.372999,-0.622836,-0.407599,1.26674,-0.58052,-0.328634]
    #y = [2.18214,-0.813473,-0.453712,-0.370705,-0.609161,-0.356038,-0.37072,-0.37333,-0.614514,-0.405674,-0.880783,-0.585333,-0.32857]
    #list_a = [[-0.0559087,-0.0161989,-0.297034,-0.601665,-0.235129,-0.267696,-0.372999,-0.622836,-0.407599,1.26674,-0.58052,-0.328634],
    #         [-0.813473,-0.453712,-0.370705,-0.609161,-0.356038,-0.37072,-0.37333,-0.614514,-0.405674,-0.880783,-0.585333,-0.32857]]
    
    
    
    # concatinate the output values to comment excel file
    #comment_file = pd.read_excel('commentData.xlsx')
    prediction_series = pd.Series(prediction)
    
    df1= pd.concat([df,prediction_series], axis=1)
    df1 = df1.rename(columns={0: 'is_spam'})
    id_group = df1.groupby(['is_spam'])
    spam_group = id_group.get_group(1)
    ham_group = id_group.get_group(0)
    spam_list = spam_group['cid'].tolist()
    ham_list = ham_group['cid'].tolist()
    
    total_comments = len(prediction_series)
    spam_commnets = len(spam_list)
    spam_presentage = (spam_commnets/total_comments)*100
    
    #spam_id_list = spam_comments_ids.tolist()
    #string_ids = [str(li) for li in spam_id_list]
    
    get_ham_comments(str_val,spam_list,ham_list)
    #return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(prediction))
    return render_template('index.html', prediction_text="spam comments presentage of this video is {}".format(spam_presentage))
    #return render_template('index.html', prediction_text="download success")

# =============================================================================
# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
# 
#     output = prediction[0]
#     return jsonify(output)
# =============================================================================

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)