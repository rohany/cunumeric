/* Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "cunumeric/sort/sort.h"
#include "cunumeric/sort/sort_template.inl"

#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;

template <LegateTypeCode CODE, int32_t DIM>
struct SortImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(VAL* inptr,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume,
                  bool is_index_space,
                  Legion::DomainPoint index_point,
                  Legion::Domain domain)
  {
    thrust::sort(inptr, inptr + volume);

    // in case of distributed data we need to switch to sample sort
    if (is_index_space) {
      // not implemented yet
      assert(false);
    }
  }
};

/*static*/ void SortTask::gpu_variant(TaskContext& context)
{
  sort_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
