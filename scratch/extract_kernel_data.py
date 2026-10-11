import os
import sys
from pathlib import Path
from jupyter_client import BlockingKernelClient

conn_file = "/run/user/1002/jupyter/runtime/kernel-v3af17432f3ab6694960853329d124ce9e0fb56d14.json"
if not os.path.exists(conn_file):
    print("Connection file not found:", conn_file)
    sys.exit(1)

kc = BlockingKernelClient()
kc.load_connection_file(conn_file)
kc.start_channels()

# Send code to inspect variables
code = """
import pickle
import torch

data_to_dump = {
    'results_t1': {},
    'results_t2': {},
}

for cond in ['I1', 'I2', 'I3', 'I4']:
    if cond in results_t1:
        data_to_dump['results_t1'][cond] = []
        for r in results_t1[cond]:
            # extract all non-model fields
            item = {k: v for k, v in r.items() if k != 'model'}
            # also save model state_dict
            item['state_dict'] = r['model'].state_dict()
            data_to_dump['results_t1'][cond].append(item)
    if cond in results_t2:
        data_to_dump['results_t2'][cond] = []
        for r in results_t2[cond]:
            item = {k: v for k, v in r.items() if k != 'model'}
            item['state_dict'] = r['model'].state_dict()
            data_to_dump['results_t2'][cond].append(item)

torch.save(data_to_dump, 'results/week5/content_allocator/extracted_experiment_runs.pt')
print('Extracted experiment runs saved successfully!')
"""

msg_id = kc.execute(code)
print("Executed code in running kernel, waiting for reply...")
reply = kc.get_shell_msg(timeout=30)
print("Reply received:", reply['content']['status'])
if reply['content']['status'] == 'error':
    print("Error:", reply['content']['ename'], reply['content']['evalue'])
    for tb in reply['content']['traceback']:
        print(tb)
kc.stop_channels()
