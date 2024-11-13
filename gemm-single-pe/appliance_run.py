import argparse
import json
import logging
import os

import numpy as np

from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder
from cerebras.sdk.client import SdkRuntime

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("__main__")
logger.setLevel(logging.INFO)

logger.info("Read artifact_path for compile artifacts")
if os.path.exists("artifact_path.json"):
    # read the artifact_path from the json file
    with open("artifact_path.json", "r", encoding="utf8") as f:
        data = json.load(f)
        artifact_path = data["artifact_path"]
else:
    raise RuntimeError(
        """The artifact_path.json file could not be found.
            Please compile the code first"""
    )

# If True, run simulator on appliance worker node
# If False, run on CS-3 within appliance
SIMULATOR=False

# parameters from compile artifacts
M=4
K=4
N=6

logger.info("Instantiate SDK Runtime object")
with SdkRuntime(
    artifact_path, simulator=SIMULATOR
) as runner:

    logger.info("Construct input A, B, C and calculate expected C")
    # Construct A, B, C
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.random.rand(M, N).astype(np.float32)
    
    # Calculate expected C
    C_expected = A @ B + C
    
    # Get symbols for A, B, C on device
    A_symbol = runner.get_id('A')
    B_symbol = runner.get_id('B')
    C_symbol = runner.get_id('C')
    
    logger.info("Copy A, B, C to device")
    # Copy A, B, C to device
    runner.memcpy_h2d(A_symbol, A.flatten(order='F'), 0, 0, 1, 1, M*K, streaming=False,
      order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    runner.memcpy_h2d(B_symbol, B.flatten(order='F'), 0, 0, 1, 1, K*N, streaming=False,
      order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    runner.memcpy_h2d(C_symbol, C.flatten(order='F'), 0, 0, 1, 1, M*N, streaming=False,
      order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    
    logger.info("Launch compute_gemm function on device")
    # Launch the compute_gemm function on device
    runner.launch('compute_gemm', nonblock=False)
    
    # Copy C back from device
    C_result = np.zeros([M*N], dtype=np.float32)
    logger.info("Copy back C from device")
    runner.memcpy_d2h(C_result, C_symbol, 0, 0, 1, 1, M*N, streaming=False,
      order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# End of SdkRuntime context manager
logger.info("Stopping program")

# Ensure that the result matches our expectation
logger.info("Check result...")
np.testing.assert_allclose(C_result, C_expected.flatten(order='F'), atol=0.01, rtol=0)
logger.info("SUCCESS!")
