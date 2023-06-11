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

#include "cunumeric/matrix/dot.h"
#include "cunumeric/matrix/dot_template.inl"

#include "core/runtime/mlir.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace cunumeric {

using namespace legate;

template <Type::Code CODE>
struct DotImplBody<VariantKind::CPU, CODE> {
  using VAL = legate_type_of<CODE>;
  using ACC = acc_type_of<VAL>;

  template <typename AccessorRD>
  void operator()(AccessorRD out,
                  const AccessorRO<VAL, 1>& rhs1,
                  const AccessorRO<VAL, 1>& rhs2,
                  const Rect<1>& rect,
                  bool dense)
  {
    const auto volume = rect.volume();
    if (dense) {
      auto rhs1ptr = rhs1.ptr(rect);
      auto rhs2ptr = rhs2.ptr(rect);
      for (coord_t idx = 0; idx < volume; ++idx) {
        const auto prod = static_cast<ACC>(rhs1ptr[idx]) * static_cast<ACC>(rhs2ptr[idx]);
        out.reduce(0, prod);
      }
    } else {
      for (coord_t idx = rect.lo[0]; idx <= rect.hi[0]; ++idx) {
        const auto prod = static_cast<ACC>(rhs1[idx]) * static_cast<ACC>(rhs2[idx]);
        out.reduce(0, prod);
      }
    }
  }
};

/*static*/ void DotTask::cpu_variant(TaskContext& context)
{
  dot_template<VariantKind::CPU>(context);
}

class DotGenerator : public MLIRTaskBodyGenerator {
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
    assert(inputs.size() == 2);
    assert(outputs.size() == 0);
    assert(reducs.size() == 1);
    assert(buflen == 0);

    auto ctx = runtime->getContext().get();
    mlir::OpBuilder builder(ctx);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());
    auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "dot"));

    auto& in1 = inputs[0];
    auto& in2 = inputs[1];
    auto& out = reducs[0];
    assert(in1.ndim == 1 && in1.typ == Type::Code::FLOAT64);
    assert(in2.ndim == 1 && in2.typ == Type::Code::FLOAT64);

    auto in1Type = buildMemRefType(ctx, in1);
    auto in2Type = buildMemRefType(ctx, in2);
    auto outType = buildMemRefType(ctx, out);
    auto funcType = builder.getFunctionType({in1Type, in2Type, outType}, std::nullopt);
    mlir::NamedAttribute namedAttr(mlir::StringAttr::get(ctx, "llvm.emit_c_interface"), mlir::UnitAttr::get(ctx));
    auto func = builder.create<mlir::func::FuncOp>(loc, kernelName, funcType, std::vector<mlir::NamedAttribute>{namedAttr});
    auto block = func.addEntryBlock();
    auto in1Var = block->getArgument(0);
    auto in2Var = block->getArgument(1);
    auto outVar = block->getArgument(2);
    builder.setInsertionPointToStart(block);

    auto [loopLBs, loopUBs] = loopBoundsFromVar(builder, loc, in1Var, in1.ndim);
    assert(loopLBs.size() == 1 && loopUBs.size() == 1);

    auto forOp = builder.create<mlir::affine::AffineForOp>(
      loc,
      loopLBs,
      mlir::AffineMap::getMultiDimIdentityMap(loopLBs.size(), ctx),
      loopUBs,
      mlir::AffineMap::getMultiDimIdentityMap(loopUBs.size(), ctx),
      1 /* step */,
      llvm::SmallVector<mlir::Value, 1>{builder.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(0.0), builder.getF64Type())},
      [&in1Var, &in2Var](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value iVar, mlir::ValueRange iterArgs) {
        auto in1Load = builder.create<mlir::affine::AffineLoadOp>(loc, in1Var, llvm::SmallVector<mlir::Value, 1>{iVar});
        auto in2Load = builder.create<mlir::affine::AffineLoadOp>(loc, in2Var, llvm::SmallVector<mlir::Value, 1>{iVar});
        // TODO (rohany): Dispatch to the right kind of binary operations here.
        auto mul = builder.create<mlir::arith::MulFOp>(loc, in1Load.getType(), in1Load, in2Load);
        auto add = builder.create<mlir::arith::AddFOp>(loc, mul.getType(), mul, iterArgs[0]);
        builder.create<mlir::affine::AffineYieldOp>(loc, llvm::SmallVector<mlir::Value, 1>{add});
      }
    );
    auto existing = builder.create<mlir::affine::AffineLoadOp>(loc, outVar, llvm::SmallVector<mlir::Value, 1>());
    auto add = builder.create<mlir::arith::AddFOp>(loc, existing.getType(), existing, forOp.getResults()[0]);
    builder.create<mlir::affine::AffineStoreOp>(loc, add, outVar, llvm::SmallVector<mlir::Value, 1>());
    builder.create<mlir::func::ReturnOp>(loc);

    return std::make_unique<MLIRModule>(std::move(module), kernelName, inputs, outputs, reducs);
  }
  ~DotGenerator() {}
};

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  auto generator = std::make_unique<DotGenerator>();
  DotTask::register_variants(std::move(generator));
}
}  // namespace

}  // namespace cunumeric
