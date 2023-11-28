from flask import Flask, Response, request, render_template, jsonify, json
from flask_socketio import SocketIO, emit
from src.model import *
from src.utils import *
from time import time
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['CACHE_TYPE'] = 'SimpleCache'
socketio = SocketIO(app, cors_allowed_origins="*")


"""
Home page
"""
src_vocab, tgt_vocab = Vocabulary(), Vocabulary()
src_vocab.load("./asset/src_vocab.txt")
tgt_vocab.load("./asset/trg_vocab.txt")
print("Vocab loaded.")
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)
Memsizer_model = Memsizer(src_vocab_size, tgt_vocab_size)
Memsizer_model.load_state_dict(torch.load("./asset/memsizer.pth"))
Memsizer_model.eval()

Transformer_model = TransformerModel(src_vocab_size, tgt_vocab_size)
Transformer_model.load_state_dict(torch.load("./asset/transformer.pth"))
Transformer_model.eval()
print("Model loaded.")


def predict(model, question, model_name="Transformer"):
    t = time()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    question = question.strip()
    question = list(question)
    question = [src_vocab.get_idx(token) for token in question]
    question = torch.tensor(question, device=DEVICE, dtype=torch.int64)
    question = question.unsqueeze(0)
    question = question.permute(1, 0)
    question = question.to(DEVICE)
    tgt = torch.tensor([tgt_vocab.sos_id, 0, 0, 2], device=DEVICE, dtype=torch.int64)
    tgt = tgt.unsqueeze(0)
    y_input = tgt[:,:-1]
    y_input = y_input.permute(1,0).to(DEVICE)
    output = model(question, y_input)
    output = output.permute(1, 2, 0)
    answer = model.greedy_search(question, y_input)
    answer = answer.squeeze(1)
    answer = answer.transpose(0, 1)
    answer = answer.tolist()
    answer = [tgt_vocab.id_to_string[idx] for idx in answer[0]]
    answer = "".join(answer)
    s = time() - t
    answer = model_name + ": " + answer + " " + '(Time used {:.2f}'.format(s) + 's)'
    return answer


@app.route('/')
def index():
    return render_template("./chat.html")

@app.route('/transformer/', methods=['PUT'])
def predict_transformer():
    print("transformer", request.get_data())
    question = request.get_data().decode('utf-8')
    answer = predict(Transformer_model, question)
    return answer

@app.route('/memsizer/', methods=['PUT'])
def predict_memsizer():
    question = request.get_data().decode('utf-8')
    answer = predict(Memsizer_model, question, model_name = "Memsizer")
    return answer

if __name__ == '__main__':
    socketio.run(app, debug=True)