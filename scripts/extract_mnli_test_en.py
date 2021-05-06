import os
import json

if __name__ == "__main__":
    in_test_path = os.path.abspath('datasets/MNLI/xnli.test.jsonl')
    out_test_path = os.path.abspath('datasets/MNLI/multinli_1.0_test.jsonl')
    with open(in_test_path, 'r') as in_handle:
        with open(out_test_path, 'w+') as out_handle:
            for jline in in_handle.read().splitlines():
                line = json.loads(jline)
                if line.get('language') == 'en':
                    json.dump(line, out_handle)
                    out_handle.write('\n')
