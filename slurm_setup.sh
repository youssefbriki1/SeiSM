cd /project/60004/fauverick
git clone https://github.com/youssefbriki1/ift3710.git
cd ift3710

# create venv
module load gcc arrow/22.0.0
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt

# run processing pipeline
cd src/data-processing/california
./run_pre_processing.sh

cd ../..

