import numpy as np
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource, fields


app = Flask(__name__)

#Descricao do App
app_infos = dict(version='1.0', title='Primeira API',
    description='Essa api faz o que quiser', contact_email = 'romulo-ferr@hotmail.com',
    doc='/documentacao', prefix='/eee')

#inicia o swagger app
rest_app = Api(app, **app_infos)

import pickle
with open("./utils/model.pickle", "rb") as f:
    modelo_carregado = pickle.load(f)


db_model = rest_app.model('Variáveis usadas no primeiro modelo',
    {'array':fields.List(cls_or_instance=fields.Float, required=True,
    description='Strings que contem um array: 1,9,8',
    help='Ex. 1,9,8'),
    'argumento_2':fields.String(required=False,
    description='Só para exemplificar que podemos ter mais entradas de dados.')})

## Vamos organizar os endpoints por aqui!
# link gerado será: http://127.0.0.1:5000/primeiro_endpoint_swagger
nome_do_endpoint = rest_app.namespace('primeiro_endpoint_swagger', 
                    description = 'Esse endpoint é responsável por fazer uma análise estatística.')

@nome_do_endpoint.route("/")
class Classe_que_contem_funcoes(Resource):
    @rest_app.expect(db_model)
    def post(self):
        array = request.json['array']
        array = np.array(array).reshape(-1,1)
        pred = modelo_carregado.predict(np.array(array))
        
        return{
            "status":"Array recebido",
            "Quantidade_de_numeros_recebidos_para_prever:": array.shape[0],
            "valores_requisicao:": array.T[0].tolist(),
            "valores_preditos:": pred.T[0].tolist(),
        }


@app.route("/")
def primeiro_endpoint_get():
    return("Funcionando corretamente! ", 200)

@app.route("/segundo_endpoint/<int:array_do_usuario>")
def segundo_endpoint(array_do_usuario):
    array_do_usuario = np.array([array_do_usuario])
    pred = modelo_carregado.predict(array_do_usuario.reshape(1,-1))
    return(f"Sua solicitação foi predita como: {pred[0,0]}", 200)

if __name__ == "__main__":
    debug = True
    app.run(host="0.0.0.0", port=5000, debug=debug)