from flask import Flask, render_template, redirect, url_for, request, jsonify
import modules.AI as AI
import globals
import asyncio
import time
from werkzeug.datastructures import MultiDict

app = Flask(__name__)


@app.route('/')
def estimation_page():
    return render_template('estimation.html')


@app.route('/result_page', methods=['post'])
def result_page():
    age_to_payment = AI.get_age_list(request.form)
    best_occ = AI.get_best_occ(request.form)
    best_ind = AI.get_best_ind(request.form)
    best_edu = AI.get_best_edu(request.form)
    best_workplace = AI.get_best_work_place(request.form)
    return render_template('index.html',
                           expect_payment=age_to_payment[2],
                           age=age_to_payment[0],
                           age_to_payment=age_to_payment[1],
                           best_occ=best_occ,
                           best_ind=best_ind,
                           best_edu=best_edu,
                           best_workplace=best_workplace
                           )


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')


if __name__ == '__main__':
    globals.initialize()
    ini_predict = MultiDict(
        [('REL', '1'), ('IMR', '1'), ('AGE', '20'), ('SEX', '1'), ('EDU', '1'), ('IND', '0'), ('OCC', '0'),
         ('WKCLASS', '1'), ('WORKPLACE', '1'), ('MRG', '90'), ('PT', '1')])
    AI.get_age_list(ini_predict)
    app.run(host='0.0.0.0', port=5000)
