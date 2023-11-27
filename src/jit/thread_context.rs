use crate::jit::assembler::arm::alu_assembler::{AluImm, AluReg};
use crate::jit::assembler::arm::transfer_assembler::{LdmStm, LdrStrImm};
use crate::jit::jit::JitAsm;
use crate::jit::reg::{Reg, RegReserve};
use crate::jit::Cond;
use crate::memory::VmManager;
use std::cell::RefCell;
use std::ptr;
use std::rc::Rc;

#[derive(Default)]
pub struct ThreadRegs {
    pub gp_regs: [u32; 13],
    pub sp: u32,
    pub lr: u32,
    pub pc: u32,
}

impl ThreadRegs {
    pub fn emit_restore_regs(&self) -> [u32; 4] {
        let gp_regs_addr = self.gp_regs.as_ptr() as u32;
        let last_regs_addr = ptr::addr_of!(self.gp_regs[self.gp_regs.len() - 1]) as u32;
        let sp_addr = ptr::addr_of!(self.sp) as u32;
        assert_eq!(sp_addr - last_regs_addr, 4);

        let mov = AluImm::mov32(Reg::SP, gp_regs_addr);
        [
            mov[0],
            mov[1],
            LdmStm::pop_al(RegReserve::gp()),
            LdrStrImm::ldr_al(Reg::SP, Reg::SP),
        ]
    }

    pub fn emit_save_regs(&self) -> [u32; 4] {
        let last_regs_addr = ptr::addr_of!(self.gp_regs[self.gp_regs.len() - 1]) as u32;
        let sp_addr = ptr::addr_of!(self.sp) as u32;
        assert_eq!(sp_addr - last_regs_addr, 4);

        let mov = AluImm::mov32(Reg::LR, last_regs_addr);
        [
            mov[0],
            mov[1],
            LdrStrImm::str_offset_al(Reg::SP, Reg::LR, 4),
            LdmStm::push(RegReserve::gp(), Reg::LR, Cond::AL),
        ]
    }

    pub fn emit_get_reg(&self, dest_reg: Reg, src_reg: Reg) -> [u32; 3] {
        let reg_addr = match src_reg {
            Reg::LR => ptr::addr_of!(self.lr),
            _ => todo!(),
        } as u32;

        let mov = AluImm::mov32(dest_reg, reg_addr);
        [mov[0], mov[1], LdrStrImm::ldr_al(dest_reg, dest_reg)]
    }

    pub fn emit_set_reg(&self, dest_reg: Reg, src_reg: Reg, tmp_reg: Reg) -> [u32; 3] {
        let reg_addr = match dest_reg {
            Reg::LR => ptr::addr_of!(self.lr),
            Reg::PC => ptr::addr_of!(self.pc),
            _ => todo!(),
        } as u32;

        let mov = AluImm::mov32(tmp_reg, reg_addr);
        [mov[0], mov[1], LdrStrImm::str_al(src_reg, tmp_reg)]
    }
}

pub struct ThreadCtx {
    jit: JitAsm,
    pub regs: Rc<RefCell<ThreadRegs>>,
    vmm: Rc<RefCell<VmManager>>,
}

impl ThreadCtx {
    pub fn new(vmm: VmManager) -> Self {
        let vmm = Rc::new(RefCell::new(vmm));
        let regs = Rc::new(RefCell::new(ThreadRegs::default()));

        ThreadCtx {
            jit: JitAsm::new(vmm.clone(), regs.clone()),
            regs,
            vmm,
        }
    }

    pub fn run(&mut self) {
        //loop {
        self.jit.execute()
        //}
    }
}
