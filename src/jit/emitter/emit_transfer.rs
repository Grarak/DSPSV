use crate::hle::hle::{get_regs, get_regs_mut};
use crate::hle::CpuType;
use crate::jit::assembler::arm::alu_assembler::{AluImm, AluReg, AluShiftImm};
use crate::jit::assembler::arm::transfer_assembler::LdrStrImm;
use crate::jit::inst_info::{Operand, Shift, ShiftValue};
use crate::jit::inst_mem_handler::{
    inst_mem_handler, inst_mem_handler_multiple, inst_mem_handler_swp,
};
use crate::jit::jit_asm::JitAsm;
use crate::jit::reg::{Reg, RegReserve};
use crate::jit::{Cond, MemoryAmount, Op, ShiftType};

impl<'a, const CPU: CpuType> JitAsm<'a, CPU> {
    pub fn emit_single_transfer<const THUMB: bool, const WRITE: bool>(
        &mut self,
        buf_index: usize,
        pc: u32,
        pre: bool,
        write_back: bool,
        amount: MemoryAmount,
    ) {
        let jit_asm_addr = self as *mut _ as _;

        let after_host_restore = |asm: &mut Self| {
            let inst_info = &asm.jit_buf.instructions[buf_index];
            let opcodes = &mut asm.jit_buf.emit_opcodes;

            let operands = inst_info.operands();
            let op0 = *operands[0].as_reg_no_shift().unwrap();
            let og_op1 = operands[1].as_reg_no_shift().unwrap();
            let op2 = &operands[2];

            let mut reg_reserve = RegReserve::gp();
            reg_reserve -= *og_op1;
            if let Operand::Reg { reg, shift } = op2 {
                reg_reserve -= *reg;
                if let Some(shift) = shift {
                    let mut handle_shift = |value| {
                        if let ShiftValue::Reg(reg) = value {
                            reg_reserve -= reg;
                        }
                    };
                    match shift {
                        Shift::Lsl(v) => handle_shift(*v),
                        Shift::Lsr(v) => handle_shift(*v),
                        Shift::Asr(v) => handle_shift(*v),
                        Shift::Ror(v) => handle_shift(*v),
                    }
                }
            }

            let handle_emulated =
                |reg: Reg, reg_reserve: &mut RegReserve, opcodes: &mut Vec<u32>| {
                    if reg.is_emulated() || reg == Reg::SP {
                        let tmp_reg = reg_reserve.pop().unwrap();
                        if reg == Reg::PC {
                            opcodes.extend(AluImm::mov32(tmp_reg, pc + if THUMB { 4 } else { 8 }));
                        } else {
                            opcodes.extend(get_regs!(asm.hle, CPU).emit_get_reg(tmp_reg, reg));
                        }
                        tmp_reg
                    } else {
                        reg
                    }
                };

            let op1 = handle_emulated(*og_op1, &mut reg_reserve, opcodes);
            let (op2, op2_shift) = match op2 {
                Operand::Reg { reg, shift } => {
                    let reg = handle_emulated(*reg, &mut reg_reserve, opcodes);
                    match shift {
                        None => (Some(reg), None),
                        Some(shift) => {
                            let mut handle_shift =
                                |shift_type: ShiftType, value: ShiftValue| match value {
                                    ShiftValue::Reg(shift_reg) => {
                                        let reg =
                                            handle_emulated(shift_reg, &mut reg_reserve, opcodes);
                                        (Operand::reg(reg), shift_type)
                                    }
                                    ShiftValue::Imm(imm) => (Operand::imm(imm as u32), shift_type),
                                };
                            (
                                Some(reg),
                                Some(match shift {
                                    Shift::Lsl(v) => handle_shift(ShiftType::Lsl, *v),
                                    Shift::Lsr(v) => handle_shift(ShiftType::Lsr, *v),
                                    Shift::Asr(v) => handle_shift(ShiftType::Asr, *v),
                                    Shift::Ror(v) => handle_shift(ShiftType::Ror, *v),
                                }),
                            )
                        }
                    }
                }
                Operand::Imm(imm) => {
                    let tmp_reg = reg_reserve.pop().unwrap();
                    opcodes.extend(AluImm::mov32(tmp_reg, *imm));
                    (Some(tmp_reg), None)
                }
                _ => (None, None),
            };

            let addr_reg = reg_reserve.pop_call_reserved().unwrap();

            if let Some(op2) = op2 {
                match op2_shift {
                    Some((reg, shift_type)) => match reg {
                        Operand::Reg { reg, .. } => {
                            opcodes.push(if inst_info.op.mem_transfer_single_sub() {
                                AluReg::sub(addr_reg, op1, op2, shift_type, reg, Cond::AL)
                            } else {
                                AluReg::add(addr_reg, op1, op2, shift_type, reg, Cond::AL)
                            });
                        }
                        Operand::Imm(imm) => {
                            opcodes.push(if inst_info.op.mem_transfer_single_sub() {
                                AluShiftImm::sub(
                                    addr_reg,
                                    op1,
                                    op2,
                                    shift_type,
                                    imm as u8,
                                    Cond::AL,
                                )
                            } else {
                                AluShiftImm::add(
                                    addr_reg,
                                    op1,
                                    op2,
                                    shift_type,
                                    imm as u8,
                                    Cond::AL,
                                )
                            });
                        }
                        Operand::None => {
                            unreachable!()
                        }
                    },
                    None => {
                        opcodes.push(if inst_info.op.mem_transfer_single_sub() {
                            AluShiftImm::sub_al(addr_reg, op1, op2)
                        } else {
                            AluShiftImm::add_al(addr_reg, op1, op2)
                        });
                    }
                }
            }

            if inst_info.op == Op::LdrPcT {
                opcodes.push(AluImm::bic_al(addr_reg, addr_reg, 3));
            }

            if pre {
                opcodes.push(AluShiftImm::mov_al(Reg::R0, addr_reg));
            } else if op1 != Reg::R0 {
                opcodes.push(AluShiftImm::mov_al(Reg::R0, op1));
            }
            opcodes.extend(AluImm::mov32(
                Reg::R1,
                get_regs_mut!(asm.hle, CPU).get_reg_mut(op0) as *mut _ as _,
            ));
            if WRITE && op0 == Reg::PC {
                opcodes.extend(AluImm::mov32(Reg::R3, pc + if THUMB { 4 } else { 8 }));
                opcodes.push(LdrStrImm::str_al(Reg::R3, Reg::R1));
            }

            if write_back && (WRITE || op0 != *og_op1) {
                Some((*og_op1, addr_reg))
            } else {
                None
            }
        };

        let before_guest_restore = |asm: &mut Self, write_back_regs: Option<(Reg, Reg)>| {
            if let Some((op1, write_back)) = write_back_regs {
                asm.jit_buf
                    .emit_opcodes
                    .extend(get_regs!(asm.hle, CPU).emit_set_reg(op1, write_back, Reg::R0));
            }
        };

        let func_addr = match amount {
            MemoryAmount::Byte => {
                if self.jit_buf.instructions[buf_index]
                    .op
                    .mem_transfer_single_signed()
                {
                    inst_mem_handler::<CPU, THUMB, WRITE, { MemoryAmount::Byte }, true> as *const _
                } else {
                    inst_mem_handler::<CPU, THUMB, WRITE, { MemoryAmount::Byte }, false> as *const _
                }
            }
            MemoryAmount::Half => {
                if self.jit_buf.instructions[buf_index]
                    .op
                    .mem_transfer_single_signed()
                {
                    inst_mem_handler::<CPU, THUMB, WRITE, { MemoryAmount::Half }, true> as *const _
                } else {
                    inst_mem_handler::<CPU, THUMB, WRITE, { MemoryAmount::Half }, false> as *const _
                }
            }
            MemoryAmount::Word => {
                inst_mem_handler::<CPU, THUMB, WRITE, { MemoryAmount::Word }, false> as *const _
            }
            MemoryAmount::Double => {
                inst_mem_handler::<CPU, THUMB, WRITE, { MemoryAmount::Double }, false> as *const _
            }
        };

        self.emit_call_host_func(
            after_host_restore,
            before_guest_restore,
            &[None, None, Some(pc), Some(jit_asm_addr)],
            func_addr,
        );
    }

    pub fn emit_multiple_transfer<const THUMB: bool>(&mut self, buf_index: usize, pc: u32) {
        let hle_addr = self.hle as *mut _ as _;
        let inst_info = &self.jit_buf.instructions[buf_index];

        let mut rlist = (inst_info.opcode & if THUMB { 0xFF } else { 0xFFFF }) as u16;
        if inst_info.op == Op::PushLrT {
            rlist |= 1 << Reg::LR as u8;
        } else if inst_info.op == Op::PopPcT {
            rlist |= 1 << Reg::PC as u8;
        }

        let mut pre = inst_info.op.mem_transfer_pre();
        let decrement = inst_info.op.mem_transfer_decrement();
        if decrement {
            pre = !pre;
        }
        let write_back = inst_info.op.mem_transfer_write_back();

        let op0 = *inst_info.operands()[0].as_reg_no_shift().unwrap();

        #[rustfmt::skip]
        let func_addr = match (
            inst_info.op.mem_is_write(),
            inst_info.op.mem_transfer_user(),
            pre,
            write_back,
            decrement,
        ) {
            (false, false, false, false, false) => inst_mem_handler_multiple::<CPU, THUMB, false, false, false, false, false> as _,
            (true, false, false, false, false) => inst_mem_handler_multiple::<CPU, THUMB, true, false, false, false, false> as _,
            (false, true, false, false, false) => inst_mem_handler_multiple::<CPU, THUMB, false, true, false, false, false> as _,
            (true, true, false, false, false) => inst_mem_handler_multiple::<CPU, THUMB, true, true, false, false, false> as _,
            (false, false, true, false, false) => inst_mem_handler_multiple::<CPU, THUMB, false, false, true, false, false> as _,
            (true, false, true, false, false) => inst_mem_handler_multiple::<CPU, THUMB, true, false, true, false, false> as _,
            (false, true, true, false, false) => inst_mem_handler_multiple::<CPU, THUMB, false, true, true, false, false> as _,
            (true, true, true, false, false) => inst_mem_handler_multiple::<CPU, THUMB, true, true, true, false, false> as _,
            (false, false, false, true, false) => inst_mem_handler_multiple::<CPU, THUMB, false, false, false, true, false> as _,
            (true, false, false, true, false) => inst_mem_handler_multiple::<CPU, THUMB, true, false, false, true, false> as _,
            (false, true, false, true, false) => inst_mem_handler_multiple::<CPU, THUMB, false, true, false, true, false> as _,
            (true, true, false, true, false) => inst_mem_handler_multiple::<CPU, THUMB, true, true, false, true, false> as _,
            (false, false, true, true, false) => inst_mem_handler_multiple::<CPU, THUMB, false, false, true, true, false> as _,
            (true, false, true, true, false) => inst_mem_handler_multiple::<CPU, THUMB, true, false, true, true, false> as _,
            (false, true, true, true, false) => inst_mem_handler_multiple::<CPU, THUMB, false, true, true, true, false> as _,
            (true, true, true, true, false) => inst_mem_handler_multiple::<CPU, THUMB, true, true, true, true, false> as _,
            (false, false, false, false, true) => inst_mem_handler_multiple::<CPU, THUMB, false, false, false, false, true> as _,
            (true, false, false, false, true) => inst_mem_handler_multiple::<CPU, THUMB, true, false, false, false, true> as _,
            (false, true, false, false, true) => inst_mem_handler_multiple::<CPU, THUMB, false, true, false, false, true> as _,
            (true, true, false, false, true) => inst_mem_handler_multiple::<CPU, THUMB, true, true, false, false, true> as _,
            (false, false, true, false, true) => inst_mem_handler_multiple::<CPU, THUMB, false, false, true, false, true> as _,
            (true, false, true, false, true) => inst_mem_handler_multiple::<CPU, THUMB, true, false, true, false, true> as _,
            (false, true, true, false, true) => inst_mem_handler_multiple::<CPU, THUMB, false, true, true, false, true> as _,
            (true, true, true, false, true) => inst_mem_handler_multiple::<CPU, THUMB, true, true, true, false, true> as _,
            (false, false, false, true, true) => inst_mem_handler_multiple::<CPU, THUMB, false, false, false, true, true> as _,
            (true, false, false, true, true) => inst_mem_handler_multiple::<CPU, THUMB, true, false, false, true, true> as _,
            (false, true, false, true, true) => inst_mem_handler_multiple::<CPU, THUMB, false, true, false, true, true> as _,
            (true, true, false, true, true) => inst_mem_handler_multiple::<CPU, THUMB, true, true, false, true, true> as _,
            (false, false, true, true, true) => inst_mem_handler_multiple::<CPU, THUMB, false, false, true, true, true> as _,
            (true, false, true, true, true) => inst_mem_handler_multiple::<CPU, THUMB, true, false, true, true, true> as _,
            (false, true, true, true, true) => inst_mem_handler_multiple::<CPU, THUMB, false, true, true, true, true> as _,
            (true, true, true, true, true) => inst_mem_handler_multiple::<CPU, THUMB, true, true, true, true, true> as _,
        };

        self.emit_call_host_func(
            |_| {},
            |_, _| {},
            &[
                Some(pc),
                Some(rlist as u32),
                Some(op0 as u32),
                Some(hle_addr),
            ],
            func_addr,
        );
    }

    pub fn emit_str(&mut self, buf_index: usize, pc: u32) {
        let op = self.jit_buf.instructions[buf_index].op;
        self.emit_single_transfer::<false, true>(
            buf_index,
            pc,
            op.mem_transfer_pre(),
            op.mem_transfer_write_back(),
            MemoryAmount::from(op),
        );
    }

    pub fn emit_ldr(&mut self, buf_index: usize, pc: u32) {
        let op = self.jit_buf.instructions[buf_index].op;
        self.emit_single_transfer::<false, false>(
            buf_index,
            pc,
            op.mem_transfer_pre(),
            op.mem_transfer_write_back(),
            MemoryAmount::from(op),
        );
    }

    pub fn emit_swp(&mut self, buf_index: usize, pc: u32) {
        let hle_addr = self.hle as *mut _ as _;
        let inst_info = &self.jit_buf.instructions[buf_index];
        let operands = inst_info.operands();
        let op0 = *operands[0].as_reg_no_shift().unwrap();
        let op1 = *operands[1].as_reg_no_shift().unwrap();
        let op2 = *operands[2].as_reg_no_shift().unwrap();

        let reg_arg = ((op2 as u32) << 16) | ((op1 as u32) << 8) | (op0 as u32);

        let func_addr = if inst_info.op == Op::Swpb {
            inst_mem_handler_swp::<CPU, { MemoryAmount::Byte }> as *const ()
        } else {
            inst_mem_handler_swp::<CPU, { MemoryAmount::Word }> as *const ()
        };

        self.emit_call_host_func(
            |_| {},
            |_, _| {},
            &[Some(reg_arg), Some(pc), Some(hle_addr)],
            func_addr,
        );
    }
}
