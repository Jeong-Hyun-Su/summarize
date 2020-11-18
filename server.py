from transformers import AutoModelWithLMHead, AutoTokenizer
from flask import Flask, request, Response, jsonify
from queue import Queue, Empty
import torch
import time
import threading

# Server & Handling Setting
app = Flask(__name__)

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1

model = AutoModelWithLMHead.from_pretrained("t5-base", return_dict=True)
tokenizer = AutoTokenizer.from_pretrained("t5-base")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Queue 핸들링
def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (len(requests_batch) >= BATCH_SIZE):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in requests_batch:
                requests['output'] = run(requests['input'])


# 쓰레드
threading.Thread(target=handle_requests_by_batch).start()


def run(article):
    try:
        inputs = tokenizer.encode("summarize: " + article, return_tensors="pt", max_length=1024)
        tokens_tensor = inputs.to(device)
        outputs = model.generate(tokens_tensor, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        return tokenizer.decode(outputs[0])

    except Exception as e:
        return 500


@app.route("/summary", methods=['GET'])
def summary():
    # 큐에 쌓여있을 경우,
    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'error': 'Too Many Requests'}), 429

    try:
        article = request.args.get("article")

    except Exception:
        print("Empty Text")
        return Response("fail", status=400)

    # Queue - put data
    req = {
        'input': article
    }
    requests_queue.put(req)

    # Queue - wait & check
    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    return req['output']


# Health Check
@app.route("/healthz", methods=["GET"])
def healthCheck():
    return "", 200


if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=80)
