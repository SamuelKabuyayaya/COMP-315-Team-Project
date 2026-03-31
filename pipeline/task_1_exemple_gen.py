import os
from tfx.components import CsvExampleGen
from tfx.proto import example_gen_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

DATA_ROOT = os.path.join('data', 'Dataset1_adult', 'adult')