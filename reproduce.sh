python.exe -m scripts.train --config "configs\train_ppo.yaml"
python.exe -m scripts.train --config "configs\train_maskable_ppo.yaml"
python.exe -m scripts.eval --config "configs\eval.yaml"