import pickle

from flask import Flask,render_template,request

app=Flask(__name__)
with open('crop_model.pkl', 'rb') as f:
    model=pickle.load(f)

@app.route('/',methods=['GET','POST'])
def crop_recommend_view():
    if request.method == 'POST':
        nitrogen=float(request.form['N'])
        phosphorus=float(request.form['P'])
        potassium=float(request.form['K'])
        temperature=float(request.form['T'])
        humadity=float(request.form['H'])
        ph=float(request.form['ph'])
        rainfall=float(request.form['R'])
        y_pred = model.predict([[nitrogen,phosphorus,potassium,temperature,humadity,ph,rainfall]])
        y_pred=(y_pred[0])
        return render_template('prediction.html',prediction=y_pred)
    return render_template('crop_prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
