import os
import shutil
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.proto import pusher_pb2
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Evaluator, Pusher
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
DATA_ROOT = '/home/samso/comp315_project/data/Dataset1_adult/source'
OUTPUT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'tfx_pipeline_output_v2'))
TRANSFORM_MODULE = os.path.abspath(os.path.join(PIPELINE_DIR, 'transform_module.py'))
TRAINER_MODULE = os.path.abspath(os.path.join(PIPELINE_DIR, 'trainer_module.py'))
METADATA_PATH = os.path.join(OUTPUT_DIR, 'metadata.db')
SERVING_MODEL_DIR = os.path.join(OUTPUT_DIR, 'serving_model')
 
def create_pipeline(pipeline_name, pipeline_root, data_root, transform_module, trainer_module, metadata_path, serving_model_dir):
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
    
   # Step 8: Evaluator - Evaluating the current model against baseline and slices
    eval_config = tfma.EvalConfig(
        model_specs=[
            # Указываем label_key, который соответствует названию в transform_module
            tfma.ModelSpec(label_key='target')
        ],
        slicing_specs=[
            tfma.SlicingSpec(),  
            tfma.SlicingSpec(feature_keys=['sex']),
            tfma.SlicingSpec(feature_keys=['race'])
        ],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='BinaryAccuracy'),
                tfma.MetricConfig(class_name='AUC')
            ])
        ]
    )
 
    evaluator = Evaluator(
        examples=transform.outputs['transformed_examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )
 
    # Step 9: Pusher - Pushing only blessed models to serving directory
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        )
    )
 
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
            model_resolver,
            evaluator,
            pusher
        ],
        enable_cache=False, # Setting this to False to ensure fresh runs for debugging
        metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path)
    )
 
if __name__ == '__main__':
    # Removing old pipeline outputs to prevent database locks
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
                metadata_path=METADATA_PATH,
                serving_model_dir=SERVING_MODEL_DIR  
            )
        )
        print("Pipeline execution finished successfully.")
    except Exception as e:
        print(f"Pipeline execution failed: {e}")