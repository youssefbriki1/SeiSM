cd /scratch/brikiyou/
git clone https://github.com/youssefbriki1/ift3710.git
cd ift3710

# create venv
module load StdEnv/2023 gcc/12.3 python/3.10 arrow/14
unset PYTHONPATH PYTHONHOME
python3 -m venv env/py1013
source env/py1013/bin/activate

pip install -r requirements.txt


# run processing pipeline
cd src/data-processing/california
./run_preprocessing.sh

cd ../..
