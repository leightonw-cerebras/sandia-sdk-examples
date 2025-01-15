import argparse
import json
import logging

from cerebras.sdk.client import SdkCompiler

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("__main__")
logger.setLevel(logging.INFO)

# If True, produce compile artifacts on appliance for simfab run
# If False, produce compile artifacts on appliance for CS-3 run
SIMULATOR=False

if SIMULATOR:
   fabdims = "11,6"
else:
   fabdims = "757,996"

logger.info("Instantiating SDK Compiler")
compiler = SdkCompiler()

logger.info("Compiling executable artifact on appliance")
artifact_path = compiler.compile(
    # Path to source files
    "./src",
    # Top level layout file
    "layout.csl",
    # Compiler arguments
    f"--arch=wse2 -o out --fabric-dims={fabdims} --fabric-offsets=4,1 " \
    f"--params=kernel_x_dim:4,kernel_y_dim:4,M:32,N:16 " \
    f"--params=MEMCPYH2D_DATA_1_ID:0,MEMCPYH2D_DATA_2_ID:1,MEMCPYD2H_DATA_1_ID:2 " \
    f" --memcpy --channels=1",
    # Output directory
    "."
)

logger.info("Write artifact_path to a JSON file")
# write the artifact_path to a json file
with open("artifact_path.json", "w", encoding="utf8") as f:
    json.dump({"artifact_path": artifact_path,}, f)

logger.info(f"Compile artifact path: {artifact_path}")
