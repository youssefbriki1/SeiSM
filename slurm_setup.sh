cd /project/60004/fauverick
git clone https://github.com/youssefbriki1/ift3710.git
cd ift3710

# create venv
module load gcc arrow/22.0.0 cuda/12.6 
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip "setuptools<70.0.0" wheel "Cython<3"
pip install "pyproj<3.6.0" --no-build-isolation
pip install -r requirements.txt
pip install mamba-ssm --force-reinstall --no-binary mamba-ssm --no-build-isolation


# run processing pipeline
cd src/data-processing/california
./run_pre_processing.sh

cd ../..

