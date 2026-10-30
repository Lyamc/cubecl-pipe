//! Single-instruction PTX for cube ops that LLVM has no intrinsic for: `dp4a` and the generic
//! one-result `gpu_asm!` statement (e.g. `prmt.b32`), as inline PTX with 32-bit register operands.

use crate::{prelude::*, shared::to_llvm::insert::insert};
use cubecl_core::ir::dialect::{InlineAsmOp, math::Dp4aOp};

#[op_interface_impl]
impl ToLLVMDialect for Dp4aOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        info: &OperandsInfo,
    ) -> Result<()> {
        let old = self.get_operation();
        let (a, b, c) = (self.a(ctx), self.b(ctx), self.c(ctx));
        let ty = IntegerType::get(ctx, 32, Signedness::Signless).into();
        let asm = llvm::InlineAsmOp::new(ctx, ty, vec![a, b, c], "dp4a.s32.s32 $0, $1, $2, $3;", "=r,r,r,r", false);
        let v = insert(ctx, rw, &asm);
        rw.replace_operation_with_values(ctx, old, vec![v]);
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for InlineAsmOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        info: &OperandsInfo,
    ) -> Result<()> {
        let old = self.get_operation();
        let template = self.asm(ctx).as_str().to_owned();
        let inputs: Vec<Value> = self.inputs(ctx).into_iter().collect();
        let pure = self.pure(ctx);
        let ty = IntegerType::get(ctx, 32, Signedness::Signless).into();
        let mut cons = vec![String::from("=r")];
        cons.extend(inputs.iter().map(|_| String::from("r")));
        let asm = llvm::InlineAsmOp::new(ctx, ty, inputs, &template, &cons.join(","), !pure);
        let v = insert(ctx, rw, &asm);
        rw.replace_operation_with_values(ctx, old, vec![v]);
        Ok(())
    }
}
