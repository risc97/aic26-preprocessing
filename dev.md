
```bash
nix-shell
chmod +x download_data.sh
./download_data.sh

python init_db.py

transnetv2_pytorch ./data/video/

python extract_keyframes.py
```

psychology for developing pipeline
- never hold the entire results in the RAM, always writing down to hard drive
- logging carefully
- have the option to smoke test
- have the option to continue from crash/OOM/... (especially heavy task)