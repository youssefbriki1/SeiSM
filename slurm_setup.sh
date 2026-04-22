cd /project/60004/fauverick
git clone https://github.com/youssefbriki1/ift3710.git
cd ift3710

# create venv
module load StdEnv/2023 gcc/12.3 arrow/22.0.0 cuda/12.6
unset PYTHONPATH PYTHONHOME
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt


# run processing pipeline
cd src/data-processing/california
./run_preprocessing.sh

cd ../..

