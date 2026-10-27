// Copyright 2026 Eagle AI Corp.
// SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "tvm_ffi_utils.h"

namespace {

// execs (int64 [last_depth + 3]): column d holds depth d's graph exec
// (2 <= d <= last_depth; 0 where the batch did not capture it), then the
// greedy and sampled finalizers. go (int32 [>= last_depth + 1]):
// whether depth d drafts this round, written by depth d - 1's decision.
__global__ void chain_kernel(const int64_t* __restrict__ execs, const int32_t* __restrict__ go,
                             const bool* __restrict__ sampled, int depth, int last_depth) {
  const bool deeper = depth < last_depth && go[depth + 1] != 0;
  const int64_t next = execs[deeper ? depth + 1 : last_depth + 1 + static_cast<int>(*sampled)];
  // A round reaching a graph its batch did not capture would leave its tree
  // unwritten: fail loudly instead.
  if (next == 0 || cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(next),
                                   cudaStreamGraphTailLaunch) != cudaSuccess) {
    __trap();
  }
}

void check_cuda(cudaError_t status, const char* what) {
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "eagle_harness_chain: " << what << ": " << cudaGetErrorString(status);
}

// What a graph holds that device launch may refuse: its node types (device
// graphs allow kernel, memcpy, memset and child graph nodes only), memcpy
// kinds, kernels launched with attributes, and edges that are not plain
// dependencies. Reported with an instantiate failure, which CUDA otherwise
// explains as "invalid argument" alone.
std::string describe_graph(cudaGraph_t graph) {
  static const char* const kTypes[] = {"kernel", "memcpy", "memset", "host", "child graph", "empty",
                                       "event wait", "event record", "semaphore signal",
                                       "semaphore wait", "mem alloc", "mem free", "batch mem op",
                                       "conditional"};
  size_t count = 0;
  if (cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess) return "";
  std::vector<cudaGraphNode_t> nodes(count);
  if (count && cudaGraphGetNodes(graph, nodes.data(), &count) != cudaSuccess) return "";
  std::map<std::string, int> types;
  std::map<std::string, int> attributes;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type;
    if (cudaGraphNodeGetType(node, &type) != cudaSuccess) continue;
    std::string name = type < sizeof(kTypes) / sizeof(kTypes[0]) ? kTypes[type]
                                                                 : "type " + std::to_string(type);
    if (type == cudaGraphNodeTypeMemcpy) {
      cudaMemcpy3DParms params{};
      if (cudaGraphMemcpyNodeGetParams(node, &params) == cudaSuccess) {
        static const char* const kKinds[] = {"host to host", "host to device", "device to host",
                                             "device to device", "default"};
        name += std::string(" ") +
                (params.kind < 5 ? kKinds[params.kind] : std::to_string(params.kind));
      }
    } else if (type == cudaGraphNodeTypeKernel) {
      cudaLaunchAttributeValue value{};
      if (cudaGraphKernelNodeGetAttribute(node, cudaLaunchAttributeCooperative, &value) ==
              cudaSuccess &&
          value.cooperative)
        ++attributes["cooperative"];
      value = {};
      if (cudaGraphKernelNodeGetAttribute(node, cudaLaunchAttributeProgrammaticStreamSerialization,
                                          &value) == cudaSuccess &&
          value.programmaticStreamSerializationAllowed)
        ++attributes["programmatic stream serialization"];
      value = {};
      if (cudaGraphKernelNodeGetAttribute(node, cudaLaunchAttributeMemSyncDomain, &value) ==
              cudaSuccess &&
          value.memSyncDomain != cudaLaunchMemSyncDomainDefault)
        ++attributes["remote mem sync domain"];
    }
    ++types[name];
  }
  size_t edges = 0;
  int typed_edges = 0;
#if CUDART_VERSION >= 13000
  auto get_edges = cudaGraphGetEdges;
#else
  auto get_edges = cudaGraphGetEdges_v2;
#endif
  if (get_edges(graph, nullptr, nullptr, nullptr, &edges) == cudaSuccess && edges) {
    std::vector<cudaGraphNode_t> from(edges), to(edges);
    std::vector<cudaGraphEdgeData> data(edges);
    if (get_edges(graph, from.data(), to.data(), data.data(), &edges) == cudaSuccess)
      for (const cudaGraphEdgeData& edge : data)
        if (edge.type != cudaGraphDependencyTypeDefault || edge.from_port || edge.to_port)
          ++typed_edges;
  }
  std::ostringstream out;
  out << " (graph of " << count << " nodes:";
  for (const auto& [name, n] : types) out << " " << n << " " << name << ",";
  for (const auto& [name, n] : attributes) out << " " << n << " kernels " << name << ",";
  out << " " << edges << " edges, " << typed_edges << " programmatic/ported)";
  return out.str();
}

}  // namespace

int64_t eagle_harness_chain_instantiate(int64_t graph, int64_t stream) {
  cudaGraphExec_t exec = nullptr;
  const cudaError_t instantiated = cudaGraphInstantiateWithFlags(
      &exec, reinterpret_cast<cudaGraph_t>(graph), cudaGraphInstantiateFlagDeviceLaunch);
  TVM_FFI_ICHECK(instantiated == cudaSuccess)
      << "eagle_harness_chain: instantiate for device launch: " << cudaGetErrorString(instantiated)
      << describe_graph(reinterpret_cast<cudaGraph_t>(graph));
  const cudaError_t uploaded = cudaGraphUpload(exec, reinterpret_cast<cudaStream_t>(stream));
  if (uploaded != cudaSuccess) cudaGraphExecDestroy(exec);
  check_cuda(uploaded, "upload");
  return reinterpret_cast<int64_t>(exec);
}

void eagle_harness_chain_destroy(int64_t exec) {
  check_cuda(cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(exec)), "destroy");
}

void eagle_harness_chain_next(TensorView execs, TensorView go, TensorView sampled, int64_t depth,
                              int64_t last_depth) {
  check_tensor(execs, dl_int64, execs.device());
  check_tensor(go, dl_int32, execs.device());
  check_scalar(sampled, dl_bool, execs.device());
  TVM_FFI_ICHECK(last_depth >= 1 && last_depth <= 31);
  TVM_FFI_ICHECK(execs.dtype() == dl_int64 && execs.ndim() == 1 && execs.stride(0) == 1 &&
                 execs.size(0) == last_depth + 3)
      << "eagle_harness_chain: exec table";
  TVM_FFI_ICHECK(go.dtype() == dl_int32 && go.ndim() == 1 && go.stride(0) == 1 &&
                 go.size(0) > last_depth)
      << "eagle_harness_chain: go flags";
  TVM_FFI_ICHECK(sampled.dtype() == dl_bool && sampled.numel() >= 1)
      << "eagle_harness_chain: sampled flag";
  TVM_FFI_ICHECK(depth >= 1 && depth <= last_depth) << "eagle_harness_chain: depth";
  chain_kernel<<<1, 1, 0, get_stream(execs.device())>>>(
      static_cast<const int64_t*>(execs.data_ptr()), static_cast<const int32_t*>(go.data_ptr()),
      static_cast<const bool*>(sampled.data_ptr()), static_cast<int>(depth),
      static_cast<int>(last_depth));
  check_cuda(cudaGetLastError(), "chain kernel launch");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_chain_instantiate, eagle_harness_chain_instantiate);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_chain_destroy, eagle_harness_chain_destroy);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_chain_next, eagle_harness_chain_next);
