from pathlib import Path
from safenet_pipeline import SafeNetPipeline

DATA_DIR = Path(__file__).parent.parent / 'data' / 'california'
SCRIPT_DIR = Path(__file__).parent / 'data-processing' / 'california'

pipeline = SafeNetPipeline(
    data_dir             = DATA_DIR,
    preprocessing_script = SCRIPT_DIR / 'run_preprocessing.sh',
    num_patches          = 64
)

pipeline.smoke_test()
pipeline.train(skip_preprocessing=False)
# pipeline.run()