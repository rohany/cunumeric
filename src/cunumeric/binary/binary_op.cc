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

#include "cunumeric/binary/binary_op.h"
#include "cunumeric/binary/binary_op_template.inl"

#include "core/runtime/mlir.h"
#include "core/utilities/deserializer.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace cunumeric {

using namespace legate;

template <BinaryOpCode OP_CODE, Type::Code CODE, int DIM>
struct BinaryOpImplBody<VariantKind::CPU, OP_CODE, CODE, DIM> {
  using OP   = BinaryOp<OP_CODE, CODE>;
  using RHS1 = legate_type_of<CODE>;
  using RHS2 = rhs2_of_binary_op<OP_CODE, CODE>;
  using LHS  = std::result_of_t<OP(RHS1, RHS2)>;

  void operator()(OP func,
                  AccessorWO<LHS, DIM> out,
                  AccessorRO<RHS1, DIM> in1,
                  AccessorRO<RHS2, DIM> in2,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      auto outptr = out.ptr(rect);
      auto in1ptr = in1.ptr(rect);
      auto in2ptr = in2.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) outptr[idx] = func(in1ptr[idx], in2ptr[idx]);
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        out[p] = func(in1[p], in2[p]);
      }
    }
  }
};

/*static*/ void BinaryOpTask::cpu_variant(TaskContext& context)
{
  binary_op_template<VariantKind::CPU>(context);
}

mlir::Value buildBinop(mlir::OpBuilder& builder, mlir::Location& loc, mlir::Value lhs, mlir::Value rhs, BinaryOpCode code) {
  // TODO (rohany): Also do a switch on the types.
  switch (code) {
    case BinaryOpCode::ADD:
      return builder.create<mlir::arith::AddFOp>(loc, lhs.getType(), lhs, rhs);
    case BinaryOpCode::SUBTRACT:
      return builder.create<mlir::arith::SubFOp>(loc, lhs.getType(), lhs, rhs);
    case BinaryOpCode::MULTIPLY:
      return builder.create<mlir::arith::MulFOp>(loc, lhs.getType(), lhs, rhs);
    case BinaryOpCode::DIVIDE:
      return builder.create<mlir::arith::DivFOp>(loc, lhs.getType(), lhs, rhs);
    case BinaryOpCode::GREATER:
      // TODO (rohany): Using UNORDERED comparisons for now, can revisit this if I can
      //  find a specification on the numpy side somewhere.
      return builder.create<mlir::arith::CmpFOp>(
          loc, coreTypeToMLIRType(builder.getContext(), LegateTypeCode::BOOL_LT), mlir::arith::CmpFPredicate::UGT, lhs, rhs);
    case BinaryOpCode::GREATER_EQUAL:
      return builder.create<mlir::arith::CmpFOp>(
          loc, coreTypeToMLIRType(builder.getContext(), LegateTypeCode::BOOL_LT), mlir::arith::CmpFPredicate::UGE, lhs, rhs);
    case BinaryOpCode::LESS:
      return builder.create<mlir::arith::CmpFOp>(
          loc, coreTypeToMLIRType(builder.getContext(), LegateTypeCode::BOOL_LT), mlir::arith::CmpFPredicate::ULT, lhs, rhs);
    case BinaryOpCode::LESS_EQUAL:
      return builder.create<mlir::arith::CmpFOp>(
          loc, coreTypeToMLIRType(builder.getContext(), LegateTypeCode::BOOL_LT), mlir::arith::CmpFPredicate::ULE, lhs, rhs);
    default:
      assert(false);
      return lhs;
  }
}

class BinaryOpGenerator : public MLIRTaskBodyGenerator {
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
    // TODO (rohany): We'll also not worry about deduplication of stores
    //  with the same ID, or we can even just let the core passes worry about
    //  that, rather than having definitions think about it. It seems like
    //  something the core should do, which is a programmatic transformation.

    assert(inputs.size() == 2);
    assert(outputs.size() == 1);
    assert(reducs.size() == 0);

    // TODO (rohany): Better story for unpacking task / compile-time arguments...
    SimpleDeserializer dez(reinterpret_cast<int8_t*>(buffer), buflen);
    auto code = dez.unpack<Scalar>().value<BinaryOpCode>();

    auto ctx = runtime->getContext().get();
    mlir::OpBuilder builder(ctx);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());
    auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "binary_op"));

    auto& a = inputs[0];
    auto& b = inputs[1];
    auto& c = outputs[0];

    auto aType = buildMemRefType(ctx, a);
    auto bType = buildMemRefType(ctx, b);
    auto cType = buildMemRefType(ctx, c);
    auto funcType = builder.getFunctionType({aType, bType, cType}, std::nullopt);
    mlir::NamedAttribute namedAttr(mlir::StringAttr::get(ctx, "llvm.emit_c_interface"), mlir::UnitAttr::get(ctx));
    auto func = builder.create<mlir::func::FuncOp>(loc, kernelName, funcType, std::vector<mlir::NamedAttribute>{namedAttr});
    auto block = func.addEntryBlock();
    auto aVar = block->getArgument(0);
    auto bVar = block->getArgument(1);
    auto cVar = block->getArgument(2);
    builder.setInsertionPointToStart(block);

    auto [loopLBs, loopUBs] = loopBoundsFromVar(builder, loc, aVar, a.ndim);

    mlir::buildAffineLoopNest(
        builder,
        loc,
        loopLBs,
        loopUBs,
        std::vector<int64_t>(a.ndim, 1),
        [&aVar, &bVar, &cVar, &code](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange lvs) {
          auto aLoad = builder.create<mlir::AffineLoadOp>(loc, aVar, lvs);
          auto bLoad = builder.create<mlir::AffineLoadOp>(loc, bVar, lvs);
          auto binop = buildBinop(builder, loc, aLoad, bLoad, code);
          auto cStore = builder.create<mlir::AffineStoreOp>(loc, binop, cVar, lvs);
        });
    builder.create<mlir::func::ReturnOp>(loc);

    return std::make_unique<MLIRModule>(std::move(module), kernelName, inputs, outputs, reducs);
  }
  ~BinaryOpGenerator() {}
};

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  auto generator = std::make_unique<BinaryOpGenerator>();
  BinaryOpTask::register_variants(std::move(generator));
}
}  // namespace

}  // namespace cunumeric
