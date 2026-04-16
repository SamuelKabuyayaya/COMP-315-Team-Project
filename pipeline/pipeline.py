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
 
# pipeline path config
# keep key project paths in one place
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
 
    # step 1: read csv and split train and eval
    example_gen = CsvExampleGen(input_base=data_root)
 
    # step 2: compute dataset stats
    stats_gen = StatisticsGen(examples=example_gen.outputs['examples'])
 
    # step 3: infer schema from stats
    schema_gen = SchemaGen(statistics=stats_gen.outputs['statistics'])
 
    # step 4: validate examples against schema
    validator = ExampleValidator(
        statistics=stats_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )
 
    # step 5: apply feature transforms
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module
    )
 
    # step 6: train the keras model
    trainer = Trainer(
        module_file=trainer_module,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=100),
        eval_args=trainer_pb2.EvalArgs(num_steps=50)
    )
 
    # step 7: fetch latest blessed baseline model
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=standard_artifacts.Model),
        model_blessing=Channel(type=standard_artifacts.ModelBlessing)
    ).with_id('latest_blessed_model_resolver')
    
    # step 8: evaluate current model by overall and slices
    eval_config = tfma.EvalConfig(
        model_specs=[
            # label key follows transform module output
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
 
    # step 9: push model only if blessed
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        )
    )
 
    # build the final pipeline object
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
        enable_cache=False,  # keep false for fresh local runs
        metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path)
    )
 
if __name__ == '__main__':
    # clear old outputs to avoid metadata lock issues
    if os.path.exists(OUTPUT_DIR):
        print(f"cleaning old output directory: {OUTPUT_DIR}")
        try:
            shutil.rmtree(OUTPUT_DIR)
        except PermissionError:
            print("warning: access denied during cleanup. close anything using the metadata db and try again.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    # set a stable pipeline name
    unique_pipeline_name = 'adult_census_pipeline_v2'
    print(f"Starting TFX Pipeline: {unique_pipeline_name}")
 
    try:
        # run pipeline locally
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
        print("pipeline execution finished successfully.")
    except Exception as e:
        print(f"pipeline execution failed: {e}")
