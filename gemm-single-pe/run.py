#!/usr/bin/env cs_python

import argparse
import json
import logging
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("__main__")
logger.setLevel(logging.INFO)

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

# Get matrix dimensions from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

M = int(compile_data['params']['M'])
K = int(compile_data['params']['K'])
N = int(compile_data['params']['N'])

logger.info("Construct input A, B, C and calculate expected C")
# Construct A, B, C
A = np.random.rand(M, K).astype(np.float32)
B = np.random.rand(K, N).astype(np.float32)
C = np.random.rand(M, N).astype(np.float32)

# Calculate expected C
C_expected = A @ B + C

logger.info("Instantiate SDK Runtime object")
# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

# Get symbols for A, B, C on device
A_symbol = runner.get_id('A')
B_symbol = runner.get_id('B')
C_symbol = runner.get_id('C')

logger.info("Load and run program")
# Load and run the program
runner.load()
runner.run()

logger.info("Copy A, B, C to device")
# Copy A, B, C to device
runner.memcpy_h2d(A_symbol, A.flatten(order='F'), 0, 0, 1, 1, M*K, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.memcpy_h2d(B_symbol, B.flatten(order='F'), 0, 0, 1, 1, K*N, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.memcpy_h2d(C_symbol, C.flatten(order='F'), 0, 0, 1, 1, M*N, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Launch the compute_gemm function on device
logger.info("Launch compute_gemm function on device")
runner.launch('compute_gemm', nonblock=False)

# Copy C back from device
C_result = np.zeros([M*N], dtype=np.float32)
logger.info("Copy back C from device")
runner.memcpy_d2h(C_result, C_symbol, 0, 0, 1, 1, M*N, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Stop the program
logger.info("Stopping program")
runner.stop()

# Ensure that the result matches our expectation
logger.info("Check result...")
np.testing.assert_allclose(C_result, C_expected.flatten(order='F'), atol=0.01, rtol=0)
logger.info("SUCCESS!")
