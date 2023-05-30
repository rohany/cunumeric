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

#include "cunumeric/ternary/where.h"
#include "cunumeric/ternary/where_template.inl"

#include "core/runtime/mlir.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cunumeric {

using namespace legate;

template <Type::Code CODE, int DIM>
struct WhereImplBody<VariantKind::CPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<bool, DIM> mask,
                  AccessorRO<VAL, DIM> in1,
                  AccessorRO<VAL, DIM> in2,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      size_t volume = rect.volume();
      auto outptr   = out.ptr(rect);
      auto maskptr  = mask.ptr(rect);
      auto in1ptr   = in1.ptr(rect);
      auto in2ptr   = in2.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx)
        outptr[idx] = maskptr[idx] ? in1ptr[idx] : in2ptr[idx];
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto point = pitches.unflatten(idx, rect.lo);
        out[point] = mask[point] ? in1[point] : in2[point];
      }
    }
  }
};

/*static*/ void WhereTask::cpu_variant(TaskContext& context)
{
  where_template<VariantKind::CPU>(context);
}

class WhereGenerator : public MLIRTaskBodyGenerator {
 public:
  std::unique_ptr<MLIRModule> generate_body(
     MLIRRuntime* runtime,
     const std::string& kernelName,
     const std::vector<CompileTimeStoreDescriptor>& inputs,
     const std::vector<CompileTimeStoreDescriptor>& outputs,
     const std::vector<CompileTimeStoreDescriptor>& reducs,
     char* buffer,
     int32_t buflen
  ) {
    assert(inputs.size() == 3);
    assert(outputs.size() == 1);
    assert(reducs.size() == 0);

    auto ctx = runtime->getContext().get();
    mlir::OpBuilder builder(ctx);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());
    auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "where"));

    auto& mask = inputs[0];
    auto& in1 = inputs[1];
    auto& in2 = inputs[2];
    auto& out = outputs[0];

    auto maskType = buildMemRefType(ctx, mask);
    auto in1Type = buildMemRefType(ctx, in1);
    auto in2Type = buildMemRefType(ctx, in2);
    auto outType = buildMemRefType(ctx, out);
    auto funcType = builder.getFunctionType({maskType, in1Type, in2Type, outType}, std::nullopt);

    mlir::NamedAttribute namedAttr(mlir::StringAttr::get(ctx, "llvm.emit_c_interface"), mlir::UnitAttr::get(ctx));
    auto func = builder.create<mlir::func::FuncOp>(loc, kernelName, funcType, std::vector<mlir::NamedAttribute>{namedAttr});
    auto block = func.addEntryBlock();
    auto maskVar = block->getArgument(0);
    auto in1Var = block->getArgument(1);
    auto in2Var = block->getArgument(2);
    auto outVar = block->getArgument(3);
    builder.setInsertionPointToStart(block);

    auto [loopLBs, loopUBs] = loopBoundsFromVar(builder, loc, maskVar, mask.ndim);

    mlir::affine::buildAffineLoopNest(
        builder,
        loc,
        loopLBs,
        loopUBs,
        std::vector<int64_t>(mask.ndim, 1),
        [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange lvs) {
          auto maskLoad = builder.create<mlir::affine::AffineLoadOp>(loc, maskVar, lvs);
	  auto in1load = builder.create<mlir::affine::AffineLoadOp>(loc, in1Var, lvs);
	  auto in2load = builder.create<mlir::affine::AffineLoadOp>(loc, in2Var, lvs);
	  auto select = builder.create<mlir::arith::SelectOp>(loc, maskLoad, in1load, in2load);
	  builder.create<mlir::affine::AffineStoreOp>(loc, select, outVar, lvs);
        });
    builder.create<mlir::func::ReturnOp>(loc);

    return std::make_unique<MLIRModule>(std::move(module), kernelName, inputs, outputs, reducs);
  }
  ~WhereGenerator() {}
};

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  auto generator = std::make_unique<WhereGenerator>();
  WhereTask::register_variants(std::move(generator));
}
}  // namespace

}  // namespace cunumeric
