# /bin/bash

python3 main.py --mode train --epochs 25
python3 main.py --mode test
python3 main.py --mode train_svm
python3 main.py --mode test_svm
