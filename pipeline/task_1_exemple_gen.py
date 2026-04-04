import os
import shutil
import time
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner

# PATH CONFIGURATION
# Absolute paths to ensure compatibility between WSL and the TFX runner
PIPELINE_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)
DATA_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, 'data', 'Dataset1_adult', 'source'))
OUTPUT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'tfx_pipeline_output'))
TRANSFORM_MODULE = os.path.abspath(os.path.join(PIPELINE_DIR, 'transform_module.py'))
METADATA_PATH = os.path.join(OUTPUT_DIR, 'metadata.db')

def create_pipeline(pipeline_name, pipeline_root, data_root, module_file, metadata_path):
  
    # Step 1: ExampleGen - Ingesting CSV and converting to TFRecord
    example_gen = CsvExampleGen(input_base=data_root)

    # Step 2: StatisticsGen - Computing per-feature statistics (min, max, distribution)
    stats_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # Step 3: SchemaGen - Inferring data types and value ranges
    schema_gen = SchemaGen(statistics=stats_gen.outputs['statistics'])

    # Step 4: ExampleValidator - Detecting anomalies (missing values, out-of-range)
    validator = ExampleValidator(
        statistics=stats_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # Step 5: Transform - Feature engineering using the preprocessing_fn
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=module_file
    )

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen, 
            stats_gen, 
            schema_gen, 
            validator, 
            transform
        ],
        enable_cache=False,  # Setting to False to ensure fresh execution for each run
        metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path)
    )

if __name__ == '__main__':
    # Removing previous artifacts to ensure a clean run
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning old output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Defining a unique name for the pipeline run
    unique_pipeline_name = f'adult_census_pipeline_{int(time.time())}'

    print(f"Starting the pipeline: {unique_pipeline_name}")

    try:
        LocalDagRunner().run(
            create_pipeline(
                pipeline_name=unique_pipeline_name,
                pipeline_root=OUTPUT_DIR,
                data_root=DATA_ROOT,
                module_file=TRANSFORM_MODULE,
                metadata_path=METADATA_PATH
            )
        )
        print(f"Pipeline execution finished.")
        print(f"Artifacts stored in: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
