"""python -m src.ingest WhatsAppExports/export.zip [-o messages.json]"""

import argparse
import json

from . import SOURCES, load, save_json

parser = argparse.ArgumentParser(description="Normalize an input file into common message records.")
parser.add_argument("path", help="input file or folder, e.g. a WhatsApp export .zip")
parser.add_argument("-o", "--output", help="write JSON here instead of printing it")
parser.add_argument("--source", choices=sorted(SOURCES), help="skip auto-detection")
args = parser.parse_args()

messages = load(args.path, source=args.source)
if args.output:
    save_json(messages, args.output)
    print(f"Wrote {len(messages)} messages to {args.output}")
else:
    print(json.dumps([m.to_dict() for m in messages], indent=2, ensure_ascii=False))
