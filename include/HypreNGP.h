// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NALU_HYPRENGP_H
#define NALU_HYPRENGP_H

#ifdef NALU_USES_NALU_HYPRE
#include "NALU_HYPRE_config.h"
#endif

#ifdef NALU_HYPRE_USING_GPU
#include "NALU_HYPRE_utilities.h"
#include "krylov.h"
#include "NALU_HYPRE.h"
#include "_nalu_hypre_utilities.h"
#include "_nalu_hypre_utilities.hpp"
#endif

#include "NaluParsing.h"
#include <yaml-cpp/yaml.h>

namespace nalu_hypre {

#ifdef NALU_HYPRE_USING_GPU

inline void
nalu_hypre_initialize()
{
  NALU_HYPRE_Init();
}

inline void
nalu_hypre_set_params(YAML::Node nodes)
{
#ifdef NALU_HYPRE_USING_DEVICE_POOL
  /* device pool allocator */
  nalu_hypre_uint mempool_bin_growth = 8, mempool_min_bin = 3, mempool_max_bin = 9;
  size_t mempool_max_cached_bytes = 2000LL * 1024 * 1024;
#endif
#if defined(NALU_HYPRE_USING_UMPIRE_DEVICE)
  long long device_pool_size = 4096LL * 1024 * 1024;
#endif
  bool use_vendor_spgemm = false;
  bool use_vendor_spmv = false;
  bool use_vendor_sptrans = false;

  const YAML::Node node = nodes["nalu_hypre_config"];
  if (node) {
#ifdef NALU_HYPRE_USING_DEVICE_POOL
    int memory_pool_mbs = 2000;
    sierra::nalu::get_if_present(
      node, "memory_pool_mbs", memory_pool_mbs, memory_pool_mbs);
    mempool_max_cached_bytes = ((size_t)memory_pool_mbs) * 1024 * 1024;
#endif
#if defined(NALU_HYPRE_USING_UMPIRE_DEVICE)
    int memory_pool_mbs = 4096;
    sierra::nalu::get_if_present(
      node, "umpire_device_pool_mbs", memory_pool_mbs, memory_pool_mbs);
    device_pool_size = ((long long)memory_pool_mbs) * 1024 * 1024;
#endif

    sierra::nalu::get_if_present(
      node, "use_vendor_spgemm", use_vendor_spgemm, use_vendor_spgemm);
    sierra::nalu::get_if_present(
      node, "use_vendor_spmv", use_vendor_spmv, use_vendor_spmv);
    sierra::nalu::get_if_present(
      node, "use_vendor_sptrans", use_vendor_sptrans, use_vendor_sptrans);
  }

#ifdef NALU_HYPRE_USING_DEVICE_POOL
  /* To be effective, nalu_hypre_SetCubMemPoolSize must immediately follow NALU_HYPRE_Init
   */
  NALU_HYPRE_SetGPUMemoryPoolSize(
    mempool_bin_growth, mempool_min_bin, mempool_max_bin,
    mempool_max_cached_bytes);
#endif
#if defined(NALU_HYPRE_USING_UMPIRE_DEVICE)
  if (device_pool_size) {
    NALU_HYPRE_SetUmpireDevicePoolName("NALU_HYPRE_DEVICE_POOL");
    NALU_HYPRE_SetUmpireDevicePoolSize(device_pool_size);
  }
#endif
  NALU_HYPRE_SetSpGemmUseVendor(use_vendor_spgemm);
  NALU_HYPRE_SetSpMVUseVendor(use_vendor_spmv);
  NALU_HYPRE_SetSpTransUseVendor(use_vendor_sptrans);
  NALU_HYPRE_SetMemoryLocation(NALU_HYPRE_MEMORY_DEVICE);
  NALU_HYPRE_SetExecutionPolicy(NALU_HYPRE_EXEC_DEVICE);
  NALU_HYPRE_SetUseGpuRand(true);
}

inline void
nalu_hypre_set_params()
{
#ifdef NALU_HYPRE_USING_DEVICE_POOL
  /* device pool allocator */
  nalu_hypre_uint mempool_bin_growth = 8, mempool_min_bin = 3, mempool_max_bin = 9;
  size_t mempool_max_cached_bytes = 2000LL * 1024 * 1024;

  /* To be effective, nalu_hypre_SetCubMemPoolSize must immediately follow NALU_HYPRE_Init
   */
  NALU_HYPRE_SetGPUMemoryPoolSize(
    mempool_bin_growth, mempool_min_bin, mempool_max_bin,
    mempool_max_cached_bytes);
#endif
#if defined(NALU_HYPRE_USING_UMPIRE_DEVICE)
  long long device_pool_size = 4096LL * 1024 * 1024;
  NALU_HYPRE_SetUmpireDevicePoolName("NALU_HYPRE_DEVICE_POOL");
  NALU_HYPRE_SetUmpireDevicePoolSize(device_pool_size);
#endif

  NALU_HYPRE_SetSpGemmUseVendor(false);
  NALU_HYPRE_SetMemoryLocation(NALU_HYPRE_MEMORY_DEVICE);
  NALU_HYPRE_SetExecutionPolicy(NALU_HYPRE_EXEC_DEVICE);
  NALU_HYPRE_SetUseGpuRand(true);
}

inline void
nalu_hypre_finalize()
{
  NALU_HYPRE_Finalize();
}

#else

inline void
nalu_hypre_initialize()
{
}

inline void
nalu_hypre_set_params(YAML::Node nodes)
{
}

inline void
nalu_hypre_set_params()
{
}

inline void
nalu_hypre_finalize()
{
}

#endif
} // namespace nalu_hypre

#endif /* NALU_HYPRENGP_H */
