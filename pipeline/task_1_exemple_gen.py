import os
import shutil
import tensorflow as tf
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.types import standard_artifacts, Channel 
from tfx.proto import trainer_pb2
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner

# Pipeline Path Configuration 
# Setting up base directories for data, modules, and pipeline outputs
PIPELINE_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)
DATA_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, 'data', 'Dataset1_adult', 'source'))
OUTPUT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'tfx_pipeline_output_v2'))
TRANSFORM_MODULE = os.path.abspath(os.path.join(PIPELINE_DIR, 'transform_module.py'))
TRAINER_MODULE = os.path.abspath(os.path.join(PIPELINE_DIR, 'trainer_module.py'))
METADATA_PATH = os.path.join(OUTPUT_DIR, 'metadata.db')

def create_pipeline(pipeline_name, pipeline_root, data_root, transform_module, trainer_module, metadata_path):
    """Initializes the TFX pipeline components."""

    # Step 1: ExampleGen - Ingesting CSV data and splitting into Train/Eval sets
    example_gen = CsvExampleGen(input_base=data_root)

    # Step 2: StatisticsGen - Computing descriptive statistics (mean, std, distributions)
    stats_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # Step 3: SchemaGen - Inferring data types and constraints from statistics
    schema_gen = SchemaGen(statistics=stats_gen.outputs['statistics'])

    # Step 4: ExampleValidator - Checking for data anomalies and schema skews
    validator = ExampleValidator(
        statistics=stats_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # Step 5: Transform - Applying feature engineering defined in transform_module.py
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module
    )

    # Step 6: Trainer - Training the Keras model using the trainer_module.py
    trainer = Trainer(
        module_file=trainer_module,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=100),
        eval_args=trainer_pb2.EvalArgs(num_steps=50)
    )

    # Step 7: Resolver - Retrieving the latest successful (blessed) model for evaluation
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=standard_artifacts.Model),
        model_blessing=Channel(type=standard_artifacts.ModelBlessing)
    ).with_id('latest_blessed_model_resolver')

    # Constructing the pipeline with the defined components
    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen, 
            stats_gen, 
            schema_gen, 
            validator, 
            transform,
            trainer,
            model_resolver
        ],
        enable_cache=False, # Setting this to False to ensure fresh runs for debugging (I spent a lot of time trying to understand why my code didnt run s I found this on the web)
        metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path)
    )

if __name__ == '__main__':
    # Removing old pipeline outputs to prevent database locks (they are really annoying to deal with and I am not )
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning old output directory: {OUTPUT_DIR}")
        try:
            shutil.rmtree(OUTPUT_DIR)
        except PermissionError:
            print("Warning: Access denied during cleanup. Ensure no other process is using the metadata DB.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Defining unique pipeline identity
    unique_pipeline_name = 'adult_census_pipeline_v2'
    print(f"Starting TFX Pipeline: {unique_pipeline_name}")

    try:
        # Executing the pipeline locally
        LocalDagRunner().run(
            create_pipeline(
                pipeline_name=unique_pipeline_name,
                pipeline_root=OUTPUT_DIR,
                data_root=DATA_ROOT,
                transform_module=TRANSFORM_MODULE,
                trainer_module=TRAINER_MODULE,
                metadata_path=METADATA_PATH
            )
        )
        print("Pipeline execution finished successfully.")
    except Exception as e:
        print(f"Pipeline execution failed: {e}")