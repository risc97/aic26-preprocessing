## psychology for developing pipeline

The dataset is big and the heavy steps take hours, so every step in this repo follows the same four rules. Keep them when you add a step:

- **Never hold all the results in RAM.** Write to disk as you go. Embeddings go out as shards, keyframes go out as JPEG files, and rows go into `metadata.db`.
- **Log what is happening.** One line per video, with the video id in it, so you can tell where a long run stopped.
- **Make it easy to smoke test.** Every script takes `--video L21_V001`, so you can try one video before you start the whole set.
- **Make it restartable.** A crash or an OOM should cost you one video, not the whole run. `extract_keyframes.py` and `embed_keyframes.py` check what is already done and skip it, and `--force` redoes a video on purpose.
  