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

#include "cunumeric/nullary/fill.h"
#include "cunumeric/nullary/fill_template.inl"

#include "core/runtime/mlir.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace cunumeric {

using namespace legate;

template <typename VAL, int32_t DIM>
struct FillImplBody<VariantKind::CPU, VAL, DIM> {
  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, 1> in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    auto fill_value = in[0];
    size_t volume   = rect.volume();
    if (dense) {
      auto outptr = out.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) outptr[idx] = fill_value;
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        const auto point = pitches.unflatten(idx, rect.lo);
        out[point]       = fill_value;
      }
    }
  }
};

/*static*/ void FillTask::cpu_variant(TaskContext& context)
{
  fill_template<VariantKind::CPU>(context);
}

class FillGenerator : public MLIRTaskBodyGenerator {
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
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    assert(reducs.size() == 0);

    // TODO (rohany): Better story for unpacking task / compile-time arguments...
    SimpleDeserializer dez(reinterpret_cast<int8_t*>(buffer), buflen);
    auto argval = dez.unpack<Scalar>().value<bool>();
    // TODO (rohany): I don't understand what this parameter does, not supporting for now.
    assert(!argval);

    auto ctx = runtime->getContext().get();
    mlir::OpBuilder builder(ctx);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());
    auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "fill"));

    auto& in = inputs[0];
    // TODO (rohany): I'm not sure why the ndim here isn't always 0?
    assert(in.ndim == 1);
    auto& out = outputs[0];

    auto inType = buildMemRefType(ctx, in);
    auto outType = buildMemRefType(ctx, out);
    auto funcType = builder.getFunctionType({inType, outType}, std::nullopt);
    mlir::NamedAttribute namedAttr(mlir::StringAttr::get(ctx, "llvm.emit_c_interface"), mlir::UnitAttr::get(ctx));
    auto func = builder.create<mlir::func::FuncOp>(loc, kernelName, funcType, std::vector<mlir::NamedAttribute>{namedAttr});
    auto block = func.addEntryBlock();
    auto inVar = block->getArgument(0);
    auto outVar = block->getArgument(1);
    builder.setInsertionPointToStart(block);

    auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto fillVal = builder.create<mlir::affine::AffineLoadOp>(loc, inVar, llvm::SmallVector<mlir::Value, 1>{zeroIdx});
    // If we're just writing to a future, no need to emit a loop.
    if (out.ndim == 0) {
      builder.create<mlir::affine::AffineStoreOp>(loc, fillVal, outVar, llvm::SmallVector<mlir::Value, 1>());
    } else {
      auto [loopLBs, loopUBs] = loopBoundsFromVar(builder, loc, outVar, out.ndim);
      mlir::affine::buildAffineLoopNest(
        builder,
        loc,
        loopLBs,
        loopUBs,
        std::vector<int64_t>(out.ndim, 1),
        [&outVar, &fillVal](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange lvs) {
          builder.create<mlir::affine::AffineStoreOp>(loc, fillVal, outVar, lvs);
        }
      );
    }

    builder.create<mlir::func::ReturnOp>(loc);
    return std::make_unique<MLIRModule>(std::move(module), kernelName, inputs, outputs, reducs);
  }
};

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  FillTask::register_variants(std::make_unique<FillGenerator>());
}
}  // namespace

}  // namespace cunumeric
