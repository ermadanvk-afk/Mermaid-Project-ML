"""
Utility to download or check the official pretrained Q-Former checkpoint
used in EmoEdit (Yang et al., 2024b) and the MERMAID paper (Yang et al., 2025).

The authors distribute the pretrained checkpoint via Dropbox:
Folder link: https://www.dropbox.com/scl/fo/7tggp1e6n4zczq2yxgcz0/AP89ShFyi7tlzvMs9RvPy_w?rlkey=3b9bhojqhyqd4tao84rwo2p3b&st=buldfasb&dl=0
Target destination: checkpoint/Q-Former.pth
"""

import os
import sys

CHECKPOINT_DIR = "checkpoint"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "Q-Former.pth")
DROPBOX_FOLDER_URL = "https://www.dropbox.com/scl/fo/7tggp1e6n4zczq2yxgcz0/AP89ShFyi7tlzvMs9RvPy_w?rlkey=3b9bhojqhyqd4tao84rwo2p3b&st=buldfasb&dl=0"

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    if os.path.exists(CHECKPOINT_PATH):
        file_size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
        print(f"[✓] Q-Former checkpoint found at '{CHECKPOINT_PATH}' ({file_size_mb:.2f} MB).")
        return

    print("=" * 70)
    print("Pretrained Q-Former Checkpoint Setup")
    print("=" * 70)
    print(f"The MERMAID paper uses the 12-layer Q-Former pretrained by Yang et al. (EmoEdit).")
    print(f"Destination: {os.path.abspath(CHECKPOINT_PATH)}")
    print(f"Author's official Dropbox link:\n  {DROPBOX_FOLDER_URL}\n")
    print("Instructions:")
    print("1. Open the Dropbox link above in your browser.")
    print("2. Download 'Q-Former.pth' (or 'EmoAdapter').")
    print(f"3. Place the downloaded file at: {CHECKPOINT_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    main()
