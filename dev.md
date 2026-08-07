
```bash
nix-shell
chmod +x download_data.sh
./download_data.sh

python init_db.py

transnetv2_pytorch ./data/video/

python extract_keyframes.py
```
