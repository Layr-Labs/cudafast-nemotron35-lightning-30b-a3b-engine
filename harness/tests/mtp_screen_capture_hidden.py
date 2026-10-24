#!/usr/bin/env python3
"""Unranked screen study: live native-MTP hidden rows from general proxy prompts.

The hook only copies live hidden rows to diagnostic storage. It does not change
proposals, target calculations, or benchmark measurement tools. Copy overhead
means this run must not be used as a performance measurement.
The saved data is not part of a submission and contains no golden outputs.
"""
import json
import os
from pathlib import Path
import sys
import torch


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vllm.model_executor.models import nemotron_h_mtp_screen as screen
    from vllm_resident.backends import VllmBackend, VllmConfig
    from transformers import AutoTokenizer

    rows = []
    original = screen.DraftHeadScreen.greedy

    def observe(self, hidden):
        rows.append(hidden.detach().clone())
        return original(self, hidden)

    screen.DraftHeadScreen.greedy = observe
    tokenizer = AutoTokenizer.from_pretrained(os.environ['RESIDENT_MODEL_DIR'], local_files_only=True)
    tasks = [
        'Write a Python binary search with tests for an empty list and repeated values.',
        'Implement a JavaScript debounce function and explain cancellation.',
        'Write a Rust iterator that parses comma-separated unsigned integers safely.',
        'Design a SQL query to compute a rolling seven-day active-user count.',
        'Implement Dijkstra shortest paths in Python with a priority queue.',
        'Explain and implement a stable merge sort in C without recursion.',
        'Create a Go HTTP handler with a bounded request body and validation.',
        'Show how to detect a linked-list cycle and identify its entry point.',
        'Implement a Python LRU cache with O(1) operations and unit tests.',
        'Write TypeScript code to group records by key with strict generic types.',
        'Describe how to test a transaction rollback using a database fixture.',
        'Implement a Java function that validates matching parentheses and brackets.',
    ]
    backend = VllmBackend(VllmConfig.from_env())
    rows.clear()  # Exclude dummy inputs used only to warm/capture startup graphs.
    counts = []
    for i, text in enumerate(tasks):
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False,
        )
        # Transformers versions differ on whether chat-template tokenization
        # returns a list or a mapping. Never pass mapping keys as token IDs.
        if hasattr(prompt, 'keys'):
            prompt = prompt['input_ids']
        assert isinstance(prompt, list) and all(isinstance(x, int) for x in prompt)
        start = len(rows)
        backend.start(prompt, max_tokens=512, logprobs=False)
        for _ in range(40):
            if not backend._engine.has_unfinished_requests():
                break
            backend.step()
        backend.abort()
        counts.append(dict(task=i, rows=len(rows)-start))
        print('CAPTURE_PROGRESS', i, counts[-1]['rows'], flush=True)
    assert rows and all(x.ndim == 2 and x.shape[1] == 2688 for x in rows)
    hidden = torch.cat(rows).cpu()
    assert hidden.dtype == torch.bfloat16 and hidden.shape[0] >= 600
    torch.save(hidden, os.environ['MTP_HIDDEN_RESULT'])
    result = dict(diagnostic_only=True, dtype=str(hidden.dtype), shape=list(hidden.shape), tasks=counts,
                  source='12 general programming proxies, not scored/golden prompts',
                  production_forward_unchanged=True)
    Path(os.environ['NEMO_RESULT']).write_text(json.dumps(result, indent=2)+'\n')
    print('CAPTURE_VERIFIER_PASS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
