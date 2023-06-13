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

#include "cunumeric/unary/scalar_unary_red.h"
#include "cunumeric/unary/scalar_unary_red_template.inl"

#include "core/runtime/mlir.h"
#include "core/utilities/deserializer.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace cunumeric {

/*static*/ void ScalarUnaryRedTask::cpu_variant(TaskContext& context)
{
  scalar_unary_red_template<VariantKind::CPU>(context);
}

class ScalarUnaryRedGenerator : public MLIRTaskBodyGenerator {
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
    assert(outputs.size() == 0);
    assert(reducs.size() == 1);

    // TODO (rohany): The existing code right now packs two arguments, where the second
    // is a shape (which I don't know what the point of that is.
    SimpleDeserializer dez(reinterpret_cast<int8_t*>(buffer), buflen);
    auto code = dez.unpack<Scalar>().value<UnaryRedCode>();
    assert(code == UnaryRedCode::SUM);

    auto ctx = runtime->getContext().get();
    mlir::OpBuilder builder(ctx);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());
    mlir::Location loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "scalar_unary_reduction"));

    auto& in = inputs[0];
    auto& out = reducs[0];
    assert(out.ndim == 0);

    auto inType = buildMemRefType(ctx, in);
    auto outType = buildMemRefType(ctx, out);
    auto funcType = builder.getFunctionType({inType, outType}, std::nullopt);
    mlir::NamedAttribute namedAttr(mlir::StringAttr::get(ctx, "llvm.emit_c_interface"), mlir::UnitAttr::get(ctx));
    auto func = builder.create<mlir::func::FuncOp>(loc, kernelName, funcType, std::vector<mlir::NamedAttribute>{namedAttr});
    auto block = func.addEntryBlock();
    auto inVar = block->getArgument(0);
    auto outVar = block->getArgument(1);
    builder.setInsertionPointToStart(block);

    auto existing = builder.create<mlir::affine::AffineLoadOp>(loc, outVar, llvm::SmallVector<mlir::Value, 1>());
    auto initVal = builder.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(0.0), builder.getF64Type());

    if (in.ndim == 0) {
      auto inVal = builder.create<mlir::affine::AffineLoadOp>(loc, inVar, llvm::SmallVector<mlir::Value, 1>());
      auto add = builder.create<mlir::arith::AddFOp>(loc, existing.getType(), existing, inVal);
      builder.create<mlir::affine::AffineStoreOp>(loc, add, outVar, llvm::SmallVector<mlir::Value, 1>());
    } else {
      // Reset the builder after all changes to it in this branch.
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto [loopLBs, loopUBs] = loopBoundsFromVar(builder, loc, inVar, in.ndim);
      llvm::SmallVector<mlir::Value, 4> lvs;
      llvm::SmallVector<mlir::affine::AffineForOp, 4> fors;
      for (int i = 0; i < in.ndim; i++) {
        // TODO (rohany): Comment up this pattern when it works...
        auto loopBody = [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::Value lv, mlir::ValueRange iterArgs) {
          lvs.push_back(lv);
          // If we're in the innermost loop, actually perform the read of the input store.
          if (i == in.ndim - 1) {
            auto inVal = nestedBuilder.create<mlir::affine::AffineLoadOp>(loc, inVar, lvs);
            auto redVal = nestedBuilder.create<mlir::arith::AddFOp>(loc, iterArgs[0].getType(), iterArgs[0], inVal);
            nestedBuilder.create<mlir::affine::AffineYieldOp>(loc, llvm::SmallVector<mlir::Value, 1>{redVal});
          }
        };
        auto loop = builder.create<mlir::affine::AffineForOp>(
          loc,
          llvm::SmallVector<mlir::Value, 1>{loopLBs[i]},
          mlir::AffineMap::getMultiDimIdentityMap(1, ctx),
          llvm::SmallVector<mlir::Value, 1>{loopUBs[i]},
          mlir::AffineMap::getMultiDimIdentityMap(1, ctx),
          1 /* step */,
          llvm::SmallVector<mlir::Value, 1>{initVal},
          loopBody
        );
        builder.setInsertionPointToStart(loop.getBody());
        fors.push_back(loop);
      }
      for (int i = in.ndim - 2; i >= 0; i--) {
        auto curFor = fors[i];
        auto nestedFor = fors[i + 1];
        // As the final operation of the current for, accumulate the result of the
        // nested into the result of the current for.
        builder.setInsertionPointAfter(nestedFor);
        auto result = nestedFor.getResults()[0];
        auto iterArg = curFor.getRegionIterArgs()[0];
        auto redVal = builder.create<mlir::arith::AddFOp>(loc, result.getType(), iterArg, result);
        builder.create<mlir::affine::AffineYieldOp>(loc, llvm::SmallVector<mlir::Value, 1>{redVal});
      }
      auto topLevelFor = fors[0];
      builder.setInsertionPointAfter(topLevelFor);
      auto result = topLevelFor.getResults()[0];
      auto add = builder.create<mlir::arith::AddFOp>(loc, existing.getType(), existing, result);
      builder.create<mlir::affine::AffineStoreOp>(loc, add, outVar, llvm::SmallVector<mlir::Value, 1>());
    }

    builder.create<mlir::func::ReturnOp>(loc);

    return std::make_unique<MLIRModule>(std::move(module), kernelName, inputs, outputs, reducs);
  }
  ~ScalarUnaryRedGenerator() {};
};

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  ScalarUnaryRedTask::register_variants(std::make_unique<ScalarUnaryRedGenerator>());
}
}  // namespace

}  // namespace cunumeric
