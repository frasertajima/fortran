# v28d Streaming Experiments

Standalone tests to validate async streaming and double buffering concepts before integration.

## Key Concepts to Test

### 1. Double Buffering
- Ping-pong buffers for overlapping data transfer with computation
- Test: Transfer batch N+1 while computing on batch N

### 2. CUDA Streams
- Async operations on separate streams
- Test: cudaMemcpyAsync vs cudaMemcpy performance

### 3. Pinned (Page-Locked) Memory
- Required for truly async transfers
- Test: cudaMallocHost vs regular malloc for H2D transfers

### 4. Stream Synchronization
- cudaStreamSynchronize vs cudaDeviceSynchronize
- Event-based synchronization for fine-grained control

## Test Files (To Create)

1. `test_pinned_memory.cuf` - Compare pinned vs pageable memory transfer speeds
2. `test_double_buffer.cuf` - Validate ping-pong buffer pattern
3. `test_stream_overlap.cuf` - Measure overlap of transfer + compute
4. `test_event_timing.cuf` - Accurate GPU timing with CUDA events

## Expected Benefits

- Hide data transfer latency behind computation
- Potentially 10-20% speedup on memory-bound operations
- Better GPU utilization during training loops
