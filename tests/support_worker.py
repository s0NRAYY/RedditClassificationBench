import json
import sys


for line in sys.stdin:
    message = json.loads(line)
    if message["type"] == "hello":
        response = {
            "type": "capabilities",
            "protocol": 1,
            "probabilities": "none",
            "max_questions": 2,
            "max_options": 2,
            "modalities": ["text"],
            "batching": True,
        }
    elif message["type"] == "close":
        print(json.dumps({"type": "closed"}), flush=True)
        break
    else:
        answers = {
            qid: {"choice": next(iter(question["criteria"]))}
            for qid, question in message["questions"].items()
        }
        response = {
            "type": "result",
            "id": message["id"],
            "result": {
                "answers": answers,
                "usage": {"input_tokens": 3, "forward_count": 1},
            },
            "timing_ms": 2.5,
        }
    print(json.dumps(response), flush=True)
