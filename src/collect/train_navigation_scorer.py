#!/usr/bin/env python3
"""Train a low-cost learned edge scorer on observable-only navigation features.

CPU logistic baseline, not RLCD. Targets/teacher distances are NOT model inputs.
Save JSON weights for cheap inference without OpenAI calls.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from build_navigation_dataset import FEATURES, read_jsonl


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))


def dot(weights, features, bias=0):
    return bias + sum(w * float(features.get(k, 0)) for k, w in zip(FEATURES, weights))


def fit(rows, epochs=100, learning_rate=.1, l2=.001):
    data = [x for x in rows if x['label'] is not None and x.get('sample_weight', 1) > 0]
    if not data:
        raise ValueError('No supervised navigation steps')
    labels = {int(x['label']) for x in data}
    if len(labels) < 2:
        raise ValueError('Need both positive and negative steps. Collect more diverse cases.')
    weights, bias = [0.0] * len(FEATURES), 0.0
    for epoch in range(epochs):
        gradient, bias_grad = [0.0] * len(FEATURES), 0.0
        for row in data:
            pred = sigmoid(dot(weights, row['features'], bias))
            grad = (pred - row['label']) * float(row.get('sample_weight', 1))
            for i, name in enumerate(FEATURES):
                gradient[i] += grad * float(row['features'].get(name, 0))
            bias_grad += grad
        rate = learning_rate / (1 + epoch / 100)
        for i in range(len(FEATURES)):
            weights[i] -= rate * (gradient[i] / len(data) + l2 * weights[i])
        bias -= rate * bias_grad / len(data)
    return {'model_type': 'logistic_graph_edge_scorer_v1', 'features': FEATURES,
            'weights': weights, 'bias': bias, 'training_samples': len(data),
            'target_dependent_features': False, 'llm_at_inference': False}


def evaluate(model, rows):
    valid = [x for x in rows if x['label'] is not None]
    if not valid:
        return {'rows': 0}
    predictions = [sigmoid(dot(model['weights'], r['features'], model['bias'])) for r in valid]
    return {'rows': len(valid),
            'accuracy_at_0_5': sum((p >= .5) == bool(r['label']) for p, r in zip(predictions, valid)) / len(valid),
            'mean_log_loss': -sum(r['label'] * math.log(max(1e-9, p)) + (1-r['label']) * math.log(max(1e-9, 1-p))
                                for r, p in zip(valid, predictions)) / len(valid)}


def train(rows):
    # Split strictly by repository when possible; never claim a holdout with one repo.
    repos = sorted({r['repo'].lower() for r in rows})
    if len(repos) >= 5:
        test_repos = {r for r in repos if int(hashlib.sha256(r.encode()).hexdigest(), 16) % 5 == 0}
        if not test_repos:
            test_repos = {repos[-1]}
        train_rows = [r for r in rows if r['repo'].lower() not in test_repos]
        eval_rows = [r for r in rows if r['repo'].lower() in test_repos]
    else:
        train_rows, eval_rows, test_repos = rows, [], set()
    model = fit(train_rows)
    model['repository_holdout'] = sorted(test_repos)
    model['training_metrics'] = evaluate(model, train_rows)
    model['validation_metrics'] = evaluate(model, eval_rows) if eval_rows else None
    model['warning'] = 'Baseline classifier; evaluate end-to-end navigation on held-out issues and benchmark repos.'
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='navigation_steps.jsonl')
    p.add_argument('--output', required=True)
    p.add_argument('--stop-input', default=None, help='optional stop_training.jsonl')
    args = p.parse_args()
    edge_model = train(read_jsonl(args.input))
    model = {'edge_model': edge_model, 'stop_model': None}
    if args.stop_input:
        try:
            model['stop_model'] = train(read_jsonl(args.stop_input))
        except ValueError as error:
            model['stop_warning'] = str(error)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(model, indent=2) + '\n', encoding='utf-8')
    print('Trained:', edge_model['training_samples'], 'steps. Validation:', edge_model['validation_metrics'])


if __name__ == '__main__':
    main()