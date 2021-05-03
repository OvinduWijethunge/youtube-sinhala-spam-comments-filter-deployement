import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from youtubeData_v3 import main_call
from comments_downlod_to_hate_module import get_ham_comments

from sklearn.preprocessing import StandardScaler



app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    id = [x for x in request.form.values()]
    #final_features = [np.array(int_features)]
    val = id[0] 
    str_val = str(val)
    #main_call(str_val)
    
    df = pd.read_csv('data.csv')
    df = df.drop('comment_id',axis =1)
    
    scaler = StandardScaler()
    scaler.fit(df)
    df = scaler.transform(df)
    #input_list = df.values.tolist()
    
    prediction = model.predict(df)

    #output = round(prediction[0], 2)
    #x = [-0.501935,-0.0559087,-0.0161989,-0.297034,-0.601665,-0.235129,-0.267696,-0.372999,-0.622836,-0.407599,1.26674,-0.58052,-0.328634]
    #y = [2.18214,-0.813473,-0.453712,-0.370705,-0.609161,-0.356038,-0.37072,-0.37333,-0.614514,-0.405674,-0.880783,-0.585333,-0.32857]
    #list_a = [[-0.0559087,-0.0161989,-0.297034,-0.601665,-0.235129,-0.267696,-0.372999,-0.622836,-0.407599,1.26674,-0.58052,-0.328634],
    #         [-0.813473,-0.453712,-0.370705,-0.609161,-0.356038,-0.37072,-0.37333,-0.614514,-0.405674,-0.880783,-0.585333,-0.32857]]
    
    
    
    # concatinate the output values to comment excel file
    comment_file = pd.read_excel('commentData.xlsx')
    prediction_series = pd.Series(prediction)
    comment_file = pd.concat([comment_file,prediction_series], axis=1)
    comment_file = comment_file.rename(columns={0: 'classifier'})
    id_group = comment_file.groupby(['classifier'])
    spam_group = id_group.get_group(1)
    spam_comments_ids = spam_group['comment_id']
    
    spam_id_list = spam_comments_ids.tolist()
    string_ids = [str(li) for li in spam_id_list]
    
    get_ham_comments(str_val,string_ids)
    #return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(prediction))
    return render_template('index.html', prediction_text=prediction)

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
    app.run(debug=True, use_reloader=True)