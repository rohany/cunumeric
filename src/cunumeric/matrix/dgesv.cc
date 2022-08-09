/* Copyright 2021-2022 NVIDIA Corporation
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

#include "cunumeric/matrix/dgesv.h"
#include <lapack.h>

namespace cunumeric {

using namespace Legion;
using namespace legate;

void DgesvTask::cpu_variant(TaskContext& ctx)
{
  assert(ctx.is_single_task());
  auto& a = ctx.inputs()[0];
  auto& b = ctx.inputs()[1];
  auto& x = ctx.outputs()[0];

  // TODO (rohany): Add the code to instantiate the right versions
  //  for this and the types etc.

  auto a_acc = a.read_accessor<double, 2>();
  auto b_acc = b.read_accessor<double, 1>();
  auto x_acc = x.read_write_accessor<double, 1>();

  // The DGESV BLAS call takes in the b vector and overwrites it
  // with the solution. So, we'll first copy b into x and then pass
  // x to the DGESV call.
  for (PointInDomainIterator<1> itr(b.domain()); itr(); itr++) { x_acc[*itr] = b_acc[*itr]; }
  // It also writes into the input matrix a, so we make a copy of it
  // here as well to avoid writing into the input.
  DeferredBuffer<double, 2> a_copy(a.domain(), Memory::SYSTEM_MEM);
  for (PointInDomainIterator<2> itr(a.domain()); itr(); itr++) { a_copy[*itr] = a_acc[*itr]; }

  int n     = a.domain().hi()[0] - a.domain().lo()[0] + 1;
  auto ipiv = create_buffer<int>(n, Memory::SYSTEM_MEM);

#ifdef LEGATE_USE_OPENMP
  openblas_set_num_threads(1);  // make sure this isn't overzealous
#endif
  int info;
  int one = 1;
  LAPACK_dgesv(&n, &one, a_copy.ptr({0, 0}), &n, ipiv.ptr(0), x_acc.ptr(0), &n, &info);
}

}  // namespace cunumeric

namespace {
static void __attribute__((constructor)) register_tasks(void)
{
  cunumeric::DgesvTask::register_variants();
}
}  // namespace