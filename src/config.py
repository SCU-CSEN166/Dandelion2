"""Settings shared by the pipeline (main.py) and the evaluation scripts."""

from pathlib import Path

# Reviewed public places/organizations the anonymizer may leave unmasked, e.g. "Nakuru".
# Never derive this list from evaluation labels.
PUBLIC_ENTITIES: list[str] = []

OUTPUT_DIR = Path("output")

# The WhatsApp group chat with hand labels, and the labels themselves.
LABELED_CHAT = Path("WhatsAppExports/export2")
WHATSAPP_LABELS = Path("data/whatsapp_labels.json")
