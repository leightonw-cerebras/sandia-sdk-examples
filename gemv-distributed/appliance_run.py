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
kernel_x_dim=4
kernel_y_dim=4

M=32
N=16

MEMCPYH2D_DATA_1=0
MEMCPYH2D_DATA_2=1
MEMCPYD2H_DATA_1=2

logger.info("Instantiate SDK Runtime object")
with SdkRuntime(
    artifact_path, simulator=SIMULATOR
) as runner:
    logger.info("Construct input A, x, b and calculate expected y")
    # Construct A, x, b
    A = np.arange(M*N, dtype=np.float32).reshape(M,N)
    x = np.full(shape=N, fill_value=1.0, dtype=np.float32)
    b = np.full(shape=M, fill_value=2.0, dtype=np.float32)

    # Calculate expected y
    y_expected = A@x + b

    # Size of N dimension on each PE
    N_per_PE = N // kernel_x_dim
    M_per_PE = M // kernel_y_dim

    # Get symbol for A on device
    A_symbol = runner.get_id('A')

    # Copy chunks of A into all PEs
    # Each chunk on each PE is stored column major
    A_prepared = A.reshape(kernel_y_dim, M_per_PE, kernel_x_dim, N_per_PE).transpose(0, 2, 3, 1).ravel()
    logger.info("Send A to device")
    runner.memcpy_h2d(A_symbol, A_prepared, 0, 0, kernel_x_dim, kernel_y_dim, M_per_PE*N_per_PE,
      streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
      nonblock=False)

    # Stream x into PEs (0, 0) and (kernel_x_dim-1, 0)
    # PE (0, 0) gets first N/2 elements; PE (1, 0) gets last N/2 elements
    logger.info("Send x to device")
    runner.memcpy_h2d(MEMCPYH2D_DATA_1, x, 0, 0, kernel_x_dim, 1, N_per_PE, streaming=True,
      order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

    # Stream b into PEs (0, 0) to (0, kernel_y_dim-1)
    logger.info("Send b to device")
    runner.memcpy_h2d(MEMCPYH2D_DATA_2, b, 0, 0, 1, kernel_y_dim, M_per_PE, streaming=True,
      order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

    # Stream y back from PEs (kernel_x_dim-1, 0) and (kernel_x_dim-1, kernel_y_dim-1)
    y_result = np.zeros([M], dtype=np.float32)
    logger.info("Receive y from device")
    runner.memcpy_d2h(y_result, MEMCPYD2H_DATA_1, kernel_x_dim-1, 0, 1, kernel_y_dim, M_per_PE,
      streaming=True, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
      nonblock=False)

# End of SdkRuntime context manager
logger.info("Stopping program")

# Ensure that the result matches our expectation
logger.info("Check result...")
np.testing.assert_allclose(y_result, y_expected)
logger.info("SUCCESS!")
