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

#include "cunumeric/unary/unary_op.h"
#include "cunumeric/unary/unary_op_template.inl"

#include "core/runtime/mlir.h"
#include "core/utilities/deserializer.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace cunumeric {

using namespace legate;

template <UnaryOpCode OP_CODE, Type::Code CODE, int DIM>
struct UnaryOpImplBody<VariantKind::CPU, OP_CODE, CODE, DIM> {
  using OP  = UnaryOp<OP_CODE, CODE>;
  using ARG = typename OP::T;
  using RES = std::result_of_t<OP(ARG)>;

  void operator()(OP func,
                  AccessorWO<RES, DIM> out,
                  AccessorRO<ARG, DIM> in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      auto outptr = out.ptr(rect);
      auto inptr  = in.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) outptr[idx] = func(inptr[idx]);
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        out[p] = func(in[p]);
      }
    }
  }
};

template <typename VAL, int DIM>
struct PointCopyImplBody<VariantKind::CPU, VAL, DIM> {
  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, DIM> in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      auto outptr = out.ptr(rect);
      auto inptr  = in.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) outptr[idx] = inptr[idx];
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        out[p] = in[p];
      }
    }
  }
};

template <UnaryOpCode OP_CODE, Type::Code CODE, int DIM>
struct MultiOutUnaryOpImplBody<VariantKind::CPU, OP_CODE, CODE, DIM> {
  using OP   = MultiOutUnaryOp<OP_CODE, CODE>;
  using RHS1 = typename OP::RHS1;
  using RHS2 = typename OP::RHS2;
  using LHS  = std::result_of_t<OP(RHS1, RHS2*)>;

  void operator()(OP func,
                  AccessorWO<LHS, DIM> lhs,
                  AccessorRO<RHS1, DIM> rhs1,
                  AccessorWO<RHS2, DIM> rhs2,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      auto lhsptr  = lhs.ptr(rect);
      auto rhs1ptr = rhs1.ptr(rect);
      auto rhs2ptr = rhs2.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) lhsptr[idx] = func(rhs1ptr[idx], &rhs2ptr[idx]);
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        lhs[p] = func(rhs1[p], rhs2.ptr(p));
      }
    }
  }
};

/*static*/ void UnaryOpTask::cpu_variant(TaskContext& context)
{
  unary_op_template<VariantKind::CPU>(context);
}

mlir::Value buildUnop(mlir::OpBuilder& builder, mlir::Location& loc, mlir::Value in, UnaryOpCode code) {
  // TODO (rohany): Also do a switch on the types.
  switch (code) {
    case UnaryOpCode::ABSOLUTE:
      return builder.create<mlir::math::AbsFOp>(loc, in.getType(), in);
    case UnaryOpCode::COPY:
    case UnaryOpCode::POSITIVE:
      return in;
    case UnaryOpCode::NEGATIVE:
      return builder.create<mlir::arith::NegFOp>(loc, in.getType(), in);
    case UnaryOpCode::SQRT:
      return builder.create<mlir::math::SqrtOp>(loc, in.getType(), in);
    case UnaryOpCode::EXP:
      return builder.create<mlir::math::ExpOp>(loc, in.getType(), in);
    case UnaryOpCode::LOG:
      return builder.create<mlir::math::LogOp>(loc, in.getType(), in);
    case UnaryOpCode::LOG10:
      return builder.create<mlir::math::Log10Op>(loc, in.getType(), in);
    case UnaryOpCode::LOG1P:
      return builder.create<mlir::math::Log1pOp>(loc, in.getType(), in);
    case UnaryOpCode::LOG2:
      return builder.create<mlir::math::Log2Op>(loc, in.getType(), in);
    default:
      assert(false);
      return in;
  }
}

class UnaryOpGenerator : public MLIRTaskBodyGenerator {
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

    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    assert(reducs.size() == 0);

    // TODO (rohany): Better story for unpacking task / compile-time arguments...
    SimpleDeserializer dez(reinterpret_cast<int8_t*>(buffer), buflen);
    auto code = dez.unpack<Scalar>().value<UnaryOpCode>();

    auto ctx = runtime->getContext().get();
    mlir::OpBuilder builder(ctx);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());
    auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "unary_op"));

    auto& in = inputs[0];
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

    auto [loopLBs, loopUBs] = loopBoundsFromVar(builder, loc, outVar, out.ndim);

    mlir::affine::buildAffineLoopNest(
        builder,
        loc,
        loopLBs,
        loopUBs,
        std::vector<int64_t>(in.ndim, 1),
        [&inVar, &outVar, &code](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange lvs) {
          auto inLoad = builder.create<mlir::affine::AffineLoadOp>(loc, inVar, lvs);
          auto unop = buildUnop(builder, loc, inLoad, code);
          builder.create<mlir::affine::AffineStoreOp>(loc, unop, outVar, lvs);
        });
    builder.create<mlir::func::ReturnOp>(loc);

    return std::make_unique<MLIRModule>(std::move(module), kernelName, inputs, outputs, reducs);
  }
  ~UnaryOpGenerator() {}
};

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  auto generator = std::make_unique<UnaryOpGenerator>();
  UnaryOpTask::register_variants(std::move(generator));
}
}  // namespace

}  // namespace cunumeric
