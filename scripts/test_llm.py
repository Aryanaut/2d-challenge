#!/usr/bin/env python3
"""Send one camera frame to the AI server and print the result (or the exact error).

Uses the same client as the llm_client node (lite3_llm_tools), with the server and group from
src/lite3_llm_tools/config/llm_client.yaml unless overridden:

    GROUP_ID=<value> ./scripts/test_llm.py                 # value from the environment
    ./scripts/test_llm.py --group <value>                  # value from the command line
    ./scripts/test_llm.py                                  # value and server from llm_client.yaml
    ./scripts/test_llm.py --image photo.jpg --server http://host:8000

The group value is never printed, only its length. Exit code 0 = the server answered with a mission JSON.
"""
import argparse
import glob
import os
import sys

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'lite3_llm_tools')
sys.path.insert(0, ROOT)
from lite3_llm_tools.perception_client import PerceptionClient, PerceptionError, read_prompt  # noqa: E402
from lite3_llm_tools.prompts import prompt_path  # noqa: E402


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'config', 'llm_client.yaml')))['llm_client']['ros__parameters']
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--server', default=cfg['server_url'])
    ap.add_argument('--group', default=os.environ.get('GROUP_ID') or str(cfg['group_id']))
    ap.add_argument('--image', help='JPEG/PNG to send (default: newest camera frame in ~/lite3_survey_runs)')
    ap.add_argument('--prompt-file', default=prompt_path('describe_view'))
    ap.add_argument('--tries', type=int, default=1)
    a = ap.parse_args()

    image = a.image
    if not image:
        frames = sorted(glob.glob(os.path.expanduser('~/lite3_survey_runs/run_*/stop*_view*.jpg')), key=os.path.getmtime)
        if not frames:
            sys.exit('No image found: pass --image')
        image = frames[-1]
    mime = 'image/png' if image.lower().endswith('.png') else 'image/jpeg'
    print(f'server {a.server} | group: {len(a.group)} chars | image {os.path.basename(image)}')
    client = PerceptionClient(a.server, a.group, read_prompt(a.prompt_file),
                              read_timeout_s=cfg.get('read_timeout_s', 75.0),
                              connect_timeout_s=cfg.get('connect_timeout_s', 5.0))
    data = open(image, 'rb').read()
    ok = False
    for i in range(a.tries):
        try:
            r = client.analyze(data, os.path.basename(image), mime)
            print(f'{i + 1}: OK HTTP {r.http_status} in {r.elapsed_s:.1f}s')
            print(r.as_dict())
            ok = True
        except PerceptionError as e:
            print(f'{i + 1}: FAILED: {e}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
