use crate::hle::hle::{get_jit, get_jit_mut, get_regs, get_regs_mut, Hle};
use crate::hle::thread_regs::ThreadRegs;
use crate::hle::CpuType;
use crate::jit::assembler::arm::alu_assembler::{AluImm, AluShiftImm};
use crate::jit::assembler::arm::transfer_assembler::LdrStrImm;
use crate::jit::assembler::arm::transfer_assembler::{LdmStm, Msr};
use crate::jit::disassembler::lookup_table::lookup_opcode;
use crate::jit::disassembler::thumb::lookup_table_thumb::lookup_thumb_opcode;
use crate::jit::inst_info::InstInfo;
use crate::jit::jit_memory::{JitInsertArgs, JIT_BLOCK_SIZE};
use crate::jit::reg::Reg;
use crate::jit::reg::{reg_reserve, RegReserve};
use crate::jit::Cond;
use crate::logging::debug_println;
use crate::DEBUG_LOG;
use std::arch::asm;
use std::intrinsics::likely;
use std::ptr;

#[derive(Default)]
struct DebugRegs {
    gp: [u32; 13],
    sp: u32,
}

impl DebugRegs {
    fn emit_save_regs(&self) -> Vec<u32> {
        let last_gp_reg_addr = ptr::addr_of!(self.gp[self.gp.len() - 1]) as u32;
        let mut opcodes = Vec::new();
        opcodes.extend(AluImm::mov32(Reg::LR, last_gp_reg_addr));
        opcodes.extend([
            LdrStrImm::str_offset_al(Reg::SP, Reg::LR, 4),
            LdmStm::push_post(RegReserve::gp(), Reg::LR, Cond::AL),
        ]);
        opcodes
    }

    fn emit_restore_regs(&self, thread_regs: &ThreadRegs) -> Vec<u32> {
        let gp_reg_addr = ptr::addr_of!(self.gp[0]) as u32;
        let cpsr_addr = thread_regs.get_reg(Reg::CPSR) as *const _ as u32;
        let mut opcodes = Vec::new();

        opcodes.extend(AluImm::mov32(Reg::SP, cpsr_addr));
        opcodes.push(LdrStrImm::ldr_al(Reg::R0, Reg::SP));
        opcodes.push(Msr::cpsr_flags(Reg::R0, Cond::AL));

        opcodes.extend(AluImm::mov32(Reg::SP, gp_reg_addr));
        opcodes.extend([
            LdmStm::pop_post_al(RegReserve::gp()),
            LdrStrImm::ldr_al(Reg::SP, Reg::SP),
        ]);
        opcodes
    }
}

#[derive(Default)]
#[repr(C)]
pub struct HostRegs {
    sp: u32,
    lr: u32,
}

impl HostRegs {
    fn get_sp_addr(&self) -> u32 {
        ptr::addr_of!(self.sp) as _
    }

    fn get_lr_addr(&self) -> u32 {
        ptr::addr_of!(self.lr) as _
    }

    fn emit_restore_sp(&self) -> Vec<u32> {
        let mut opcodes = Vec::new();
        opcodes.extend(AluImm::mov32(Reg::LR, self.get_sp_addr()));
        opcodes.push(LdrStrImm::ldr_al(Reg::SP, Reg::LR));
        opcodes
    }
}

pub struct JitBuf {
    pub instructions: Vec<InstInfo>,
    pub emit_opcodes: Vec<u32>,
    pub block_opcodes: Vec<u32>,
    pub jit_addr_offsets: Vec<u16>,
    pub insts_cycle_counts: Vec<u16>,
}

impl JitBuf {
    fn new() -> Self {
        JitBuf {
            instructions: Vec::new(),
            emit_opcodes: Vec::new(),
            block_opcodes: Vec::new(),
            jit_addr_offsets: Vec::new(),
            insts_cycle_counts: Vec::new(),
        }
    }

    fn clear_all(&mut self) {
        self.instructions.clear();
        self.block_opcodes.clear();
        self.jit_addr_offsets.clear();
        self.insts_cycle_counts.clear();
    }
}

pub struct JitAsm<'a, const CPU: CpuType> {
    pub hle: &'a mut Hle,
    pub jit_buf: JitBuf,
    pub host_regs: Box<HostRegs>,
    pub guest_branch_out_pc: u32,
    pub breakin_addr: u32,
    pub breakout_addr: u32,
    pub breakout_skip_save_regs_addr: u32,
    pub breakin_thumb_addr: u32,
    pub breakout_thumb_addr: u32,
    pub breakout_skip_save_regs_thumb_addr: u32,
    pub restore_host_opcodes: Vec<u32>,
    pub restore_guest_opcodes: Vec<u32>,
    pub restore_host_thumb_opcodes: Vec<u32>,
    pub restore_guest_thumb_opcodes: Vec<u32>,
    pub restore_host_no_save_opcodes: Vec<u32>,
    debug_regs: DebugRegs,
}

impl<'a, const CPU: CpuType> JitAsm<'a, CPU> {
    #[inline(never)]
    pub fn new(hle: &'a mut Hle) -> Self {
        let mut instance = {
            let host_regs = Box::<HostRegs>::default();

            let restore_host_opcodes = {
                let mut opcodes = Vec::new();
                // Save guest
                opcodes.extend(&get_regs!(hle, CPU).save_regs_opcodes);
                // Restore host sp
                opcodes.extend(host_regs.emit_restore_sp());
                opcodes.shrink_to_fit();
                opcodes
            };

            let restore_guest_opcodes = {
                let mut opcodes = Vec::new();
                // Restore guest
                opcodes.push(AluShiftImm::mov_al(Reg::LR, Reg::SP));
                opcodes.extend(&get_regs!(hle, CPU).restore_regs_opcodes);
                opcodes.shrink_to_fit();
                opcodes
            };

            let restore_host_thumb_opcodes = {
                let mut opcodes = Vec::new();
                // Save guest
                opcodes.extend(&get_regs!(hle, CPU).save_regs_thumb_opcodes);

                // Restore host sp
                opcodes.extend(host_regs.emit_restore_sp());
                opcodes.shrink_to_fit();
                opcodes
            };

            let restore_guest_thumb_opcodes = {
                let mut opcodes = Vec::new();
                // Restore guest
                opcodes.push(AluShiftImm::mov_al(Reg::LR, Reg::SP));
                opcodes.extend(&get_regs!(hle, CPU).restore_regs_thumb_opcodes);
                opcodes.shrink_to_fit();
                opcodes
            };

            let restore_host_no_save_opcodes = host_regs.emit_restore_sp();

            JitAsm {
                hle,
                jit_buf: JitBuf::new(),
                host_regs,
                guest_branch_out_pc: 0,
                breakin_addr: 0,
                breakout_addr: 0,
                breakout_skip_save_regs_addr: 0,
                breakin_thumb_addr: 0,
                breakout_thumb_addr: 0,
                breakout_skip_save_regs_thumb_addr: 0,
                restore_host_opcodes,
                restore_guest_opcodes,
                restore_host_thumb_opcodes,
                restore_guest_thumb_opcodes,
                restore_host_no_save_opcodes,
                debug_regs: DebugRegs::default(),
            }
        };

        get_jit_mut!(instance.hle).open();
        {
            let jit_opcodes = &mut instance.jit_buf.emit_opcodes;
            {
                // Common function to enter guest (breakin)
                // Save lr to return to this function
                jit_opcodes.extend(&AluImm::mov32(Reg::R1, instance.host_regs.get_lr_addr()));
                jit_opcodes.extend(&[
                    LdrStrImm::str(4, Reg::R0, Reg::SP, false, false, false, true, Cond::AL), // Save actual function entry
                    LdrStrImm::str_al(Reg::LR, Reg::R1), // Save host lr
                    AluShiftImm::mov_al(Reg::LR, Reg::SP), // Keep host sp in lr
                ]);
                // Restore guest
                let guest_restore_index = jit_opcodes.len();
                jit_opcodes.extend(&get_regs!(instance.hle, CPU).restore_regs_opcodes);

                jit_opcodes.push(LdrStrImm::ldr(
                    4,
                    Reg::PC,
                    Reg::LR,
                    false,
                    false,
                    false,
                    true,
                    Cond::AL,
                ));

                instance.breakin_addr = get_jit_mut!(instance.hle).insert_common(jit_opcodes);

                let restore_regs_thumb_opcodes =
                    &get_regs!(instance.hle, CPU).restore_regs_thumb_opcodes;
                jit_opcodes
                    [guest_restore_index..guest_restore_index + restore_regs_thumb_opcodes.len()]
                    .copy_from_slice(restore_regs_thumb_opcodes);
                instance.breakin_thumb_addr = get_jit_mut!(instance.hle).insert_common(jit_opcodes);

                jit_opcodes.clear();
            }

            {
                // Common function to exit guest (breakout)
                // Save guest
                jit_opcodes.extend(&get_regs!(instance.hle, CPU).save_regs_opcodes);

                // Some emits already save regs themselves, add skip addr
                let jit_skip_save_regs_offset = jit_opcodes.len() as u32;

                let host_sp_addr = instance.host_regs.get_sp_addr();
                jit_opcodes.extend(&AluImm::mov32(Reg::R0, host_sp_addr));
                jit_opcodes.extend(&[
                    LdrStrImm::ldr_al(Reg::SP, Reg::R0),           // Restore host SP
                    LdrStrImm::ldr_offset_al(Reg::PC, Reg::R0, 4), // Restore host LR and write to PC
                ]);

                instance.breakout_addr = get_jit_mut!(instance.hle).insert_common(jit_opcodes);
                instance.breakout_skip_save_regs_addr =
                    instance.breakout_addr + (jit_skip_save_regs_offset << 2);

                jit_opcodes.clear();
            }

            {
                // Common thumb function to exit guest (breakout)
                // Save guest
                jit_opcodes.extend(&get_regs!(instance.hle, CPU).save_regs_thumb_opcodes);

                // Some emits already save regs themselves, add skip addr
                let jit_skip_save_regs_offset = jit_opcodes.len() as u32;

                let host_sp_addr = instance.host_regs.get_sp_addr();
                jit_opcodes.extend(&AluImm::mov32(Reg::R0, host_sp_addr));
                jit_opcodes.extend(&[
                    LdrStrImm::ldr_al(Reg::SP, Reg::R0),           // Restore host SP
                    LdrStrImm::ldr_offset_al(Reg::PC, Reg::R0, 4), // Restore host LR and write to PC
                ]);

                instance.breakout_thumb_addr =
                    get_jit_mut!(instance.hle).insert_common(jit_opcodes);
                instance.breakout_skip_save_regs_thumb_addr =
                    instance.breakout_thumb_addr + (jit_skip_save_regs_offset << 2);

                jit_opcodes.clear();
            }
        }
        get_jit_mut!(instance.hle).close();

        instance
    }

    #[inline(never)]
    fn emit_code_block<const THUMB: bool>(&mut self, guest_pc_block_base: u32) {
        {
            let mut index = 0;
            loop {
                let inst_info = if THUMB {
                    let opcode = self.hle.mem_read::<CPU, u16>(guest_pc_block_base + index);
                    debug_println!("{:?} disassemble thumb {:x} {}", CPU, opcode, opcode);
                    let (op, func) = lookup_thumb_opcode(opcode);
                    InstInfo::from(func(opcode, *op))
                } else {
                    let opcode = self.hle.mem_read::<CPU, u32>(guest_pc_block_base + index);
                    debug_println!("{:?} disassemble {:x} {}", CPU, opcode, opcode);
                    let (op, func) = lookup_opcode(opcode);
                    func(opcode, *op)
                };

                if let Some(last) = self.jit_buf.insts_cycle_counts.last() {
                    assert!(u16::MAX - last >= inst_info.cycle as u16);
                    self.jit_buf
                        .insts_cycle_counts
                        .push(last + inst_info.cycle as u16);
                } else {
                    self.jit_buf.insts_cycle_counts.push(inst_info.cycle as u16);
                }
                self.jit_buf.instructions.push(inst_info);

                index += if THUMB { 2 } else { 4 };
                if index == JIT_BLOCK_SIZE {
                    break;
                }
            }
        }

        for i in 0..self.jit_buf.instructions.len() {
            let pc = ((i as u32) << if THUMB { 1 } else { 2 }) + guest_pc_block_base;
            let opcodes_len = self.jit_buf.block_opcodes.len();
            assert!((opcodes_len << 2) <= u16::MAX as usize);
            self.jit_buf
                .jit_addr_offsets
                .push((opcodes_len << 2) as u16);

            debug_println!("{:?} emitting {:?}", CPU, self.jit_buf.instructions[i]);

            if THUMB {
                self.emit_thumb(i, pc);
            } else {
                self.emit(i, pc);
            };

            if DEBUG_LOG {
                let inst_info = &self.jit_buf.instructions[i];

                self.jit_buf
                    .emit_opcodes
                    .extend(self.debug_regs.emit_save_regs());
                self.jit_buf
                    .emit_opcodes
                    .extend(self.host_regs.emit_restore_sp());

                self.jit_buf
                    .emit_opcodes
                    .extend(AluImm::mov32(Reg::R0, self as *const _ as u32));
                self.jit_buf.emit_opcodes.extend(AluImm::mov32(Reg::R1, pc));
                self.jit_buf
                    .emit_opcodes
                    .extend(AluImm::mov32(Reg::R2, inst_info.opcode));

                Self::emit_host_blx(
                    debug_after_exec_op::<CPU> as *const () as _,
                    &mut self.jit_buf.emit_opcodes,
                );

                self.jit_buf
                    .emit_opcodes
                    .push(AluShiftImm::mov_al(Reg::LR, Reg::SP));
                self.jit_buf
                    .emit_opcodes
                    .extend(self.debug_regs.emit_restore_regs(get_regs!(self.hle, CPU)));
            }

            self.jit_buf
                .block_opcodes
                .extend(&self.jit_buf.emit_opcodes);
            self.jit_buf.emit_opcodes.clear();
        }

        {
            let opcodes = &mut self.jit_buf.block_opcodes;

            let regs = get_regs_mut!(self.hle, CPU);
            if THUMB {
                opcodes.extend(&regs.save_regs_thumb_opcodes);
            } else {
                opcodes.extend(&regs.save_regs_opcodes);
            }

            opcodes.extend(&AluImm::mov32(
                Reg::R0,
                guest_pc_block_base + JIT_BLOCK_SIZE - if THUMB { 1 } else { 4 },
            ));
            opcodes.extend(AluImm::mov32(
                Reg::R1,
                ptr::addr_of_mut!(self.guest_branch_out_pc) as u32,
            ));

            opcodes.push(LdrStrImm::str_al(Reg::R0, Reg::R1));
            opcodes.push(AluImm::add_al(Reg::R2, Reg::R0, if THUMB { 2 } else { 4 }));
            opcodes.extend(get_regs!(self.hle, CPU).emit_set_reg(Reg::PC, Reg::R2, Reg::R3));

            if THUMB {
                Self::emit_host_bx(self.breakout_skip_save_regs_thumb_addr, opcodes);
            } else {
                Self::emit_host_bx(self.breakout_skip_save_regs_addr, opcodes);
            }
        }

        // TODO statically analyze generated insts

        get_jit_mut!(self.hle).insert_block::<CPU, THUMB>(
            &self.jit_buf.block_opcodes,
            JitInsertArgs::new(
                guest_pc_block_base,
                self.jit_buf.jit_addr_offsets.clone(),
                self.jit_buf.insts_cycle_counts.clone(),
            ),
        );

        if DEBUG_LOG {
            for (index, inst_info) in self.jit_buf.instructions.iter().enumerate() {
                let pc = ((index as u32) << if THUMB { 1 } else { 2 }) + guest_pc_block_base;
                let (jit_addr, _) = get_jit!(self.hle)
                    .get_jit_start_addr::<CPU, THUMB>(pc)
                    .unwrap();

                debug_println!(
                    "{:?} Mapping {:#010x} to {:#010x} {:?}",
                    CPU,
                    pc,
                    jit_addr,
                    inst_info
                );
            }
        }
        self.jit_buf.clear_all();
    }

    pub fn execute(&mut self) -> u16 {
        let entry = get_regs!(self.hle, CPU).pc;

        let thumb = (entry & 1) == 1;
        debug_println!("{:?} Execute {:x} thumb {}", CPU, entry, thumb);
        if thumb {
            self.execute_internal::<true>(entry & !1)
        } else {
            self.execute_internal::<false>(entry & !3)
        }
    }

    fn execute_internal<const THUMB: bool>(&mut self, guest_pc: u32) -> u16 {
        get_regs_mut!(self.hle, CPU).set_thumb(THUMB);
        let guest_pc_block_base = guest_pc & !(JIT_BLOCK_SIZE - 1);

        let jit_info = get_jit!(self.hle).get_jit_start_addr::<CPU, THUMB>(guest_pc);
        let (jit_addr, pre_cycle_count_sum) = if likely(jit_info.is_some()) {
            unsafe { jit_info.unwrap_unchecked() }
        } else {
            self.emit_code_block::<THUMB>(guest_pc_block_base);
            get_jit!(self.hle)
                .get_jit_start_addr::<CPU, THUMB>(guest_pc)
                .unwrap()
        };

        debug_println!("{:?} Enter jit addr {:x}", CPU, jit_addr);

        if DEBUG_LOG {
            self.guest_branch_out_pc = 0;
        }

        self.hle.mem.current_jit_block_range = (
            guest_pc,
            guest_pc_block_base + JIT_BLOCK_SIZE - if THUMB { 2 } else { 4 },
        );

        unsafe {
            enter_jit(
                jit_addr,
                self.host_regs.get_sp_addr(),
                if THUMB {
                    self.breakin_thumb_addr
                } else {
                    self.breakin_addr
                },
            )
        };
        debug_assert!(self.guest_branch_out_pc != 0 || (CPU == CpuType::ARM7 && guest_pc == 0));

        let (branch_out_pre_cycle_count_sum, branch_out_inst_cycle_count) =
            get_jit!(self.hle).get_cycle_counts_unchecked::<CPU, THUMB>(self.guest_branch_out_pc);

        let cycle_correction = {
            let regs = get_regs_mut!(self.hle, CPU);
            let correction = regs.cycle_correction;
            regs.cycle_correction = 0;
            correction
        };

        let executed_cycles = (branch_out_pre_cycle_count_sum + branch_out_inst_cycle_count as u16
            - pre_cycle_count_sum
            // + 2 for branching out
            + 2) as i16
            + cycle_correction;

        if DEBUG_LOG {
            println!(
                "{:?} reading opcode of breakout at {:x}",
                CPU, self.guest_branch_out_pc
            );
            let inst_info = if THUMB {
                let opcode = self.hle.mem_read::<CPU, _>(self.guest_branch_out_pc);
                let (op, func) = lookup_thumb_opcode(opcode);
                InstInfo::from(func(opcode, *op))
            } else {
                let opcode = self.hle.mem_read::<CPU, _>(self.guest_branch_out_pc);
                let (op, func) = lookup_opcode(opcode);
                func(opcode, *op)
            };
            debug_inst_info::<CPU>(
                None,
                get_regs!(self.hle, CPU),
                self.guest_branch_out_pc,
                &format!("breakout\n\t{:?} {:?}", CPU, inst_info),
            );
        }

        executed_cycles as u16
    }
}

#[naked]
unsafe extern "C" fn enter_jit(jit_entry: u32, host_sp_addr: u32, breakin_addr: u32) {
    asm!(
        "push {{r4-r11,lr}}",
        "str sp, [r1]",
        "blx r2",
        "pop {{r4-r11,pc}}",
        options(noreturn)
    );
}

fn debug_inst_info<const CPU: CpuType>(
    debug_regs: Option<&DebugRegs>,
    regs: &ThreadRegs,
    pc: u32,
    append: &str,
) {
    let mut output = "Executed ".to_owned();

    match debug_regs {
        Some(debug_regs) => {
            for reg in RegReserve::gp_thumb() {
                output += &format!("{:?}: {:x}, ", reg, debug_regs.gp[reg as usize]);
            }
            for reg in (!RegReserve::gp_thumb()).get_gp_regs() {
                output += &format!(
                    "{:?}: {:x}, ",
                    reg,
                    if regs.is_thumb() {
                        *regs.get_reg(reg)
                    } else {
                        debug_regs.gp[reg as usize]
                    }
                );
            }
            output += &format!("{:?}: {:x}, ", Reg::SP, debug_regs.sp);
            for reg in reg_reserve!(Reg::LR, Reg::PC, Reg::CPSR, Reg::SPSR) {
                let value = if reg != Reg::PC {
                    *regs.get_reg(reg)
                } else {
                    pc
                };
                output += &format!("{:?}: {:x}, ", reg, value);
            }
        }
        None => {
            for reg in
                reg_reserve!(Reg::SP, Reg::LR, Reg::PC, Reg::CPSR, Reg::SPSR) + RegReserve::gp()
            {
                let value = if reg != Reg::PC {
                    *regs.get_reg(reg)
                } else {
                    pc
                };
                output += &format!("{:?}: {:x}, ", reg, value);
            }
        }
    }

    println!("{:?} {}{}", CPU, output, append);
}

unsafe extern "C" fn debug_after_exec_op<const CPU: CpuType>(
    asm: *mut JitAsm<CPU>,
    pc: u32,
    opcode: u32,
) {
    let asm = asm.as_mut().unwrap_unchecked();
    let inst_info = {
        if get_regs!(asm.hle, CPU).is_thumb() {
            let (op, func) = lookup_thumb_opcode(opcode as u16);
            InstInfo::from(func(opcode as u16, *op))
        } else {
            let (op, func) = lookup_opcode(opcode);
            func(opcode, *op)
        }
    };

    debug_inst_info::<CPU>(
        Some(&asm.debug_regs),
        get_regs!(asm.hle, CPU),
        pc,
        &format!("\n\t{:?} {:?}", CPU, inst_info),
    );
}
