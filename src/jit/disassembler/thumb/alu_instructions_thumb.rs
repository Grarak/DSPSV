mod alu_thumb_ops {
    use crate::jit::inst_info::{Operand, Operands};
    use crate::jit::inst_info_thumb::InstInfoThumb;
    use crate::jit::reg::{reg_reserve, Reg};
    use crate::jit::Op;
    use crate::utils;

    #[inline]
    pub fn add_reg_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((opcode & 0x7) as u8);
        let op1 = Reg::from(((opcode >> 3) & 0x7) as u8);
        let op2 = Reg::from(((opcode >> 6) & 0x7) as u8);
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_3(Operand::reg(op0), Operand::reg(op1), Operand::reg(op2)),
            reg_reserve!(op1, op2),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn sub_reg_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((opcode & 0x7) as u8);
        let op1 = Reg::from(((opcode >> 3) & 0x7) as u8);
        let op2 = Reg::from(((opcode >> 6) & 0x7) as u8);
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_3(Operand::reg(op0), Operand::reg(op1), Operand::reg(op2)),
            reg_reserve!(op1, op2),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn add_h_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((((opcode >> 4) & 0x8) | (opcode & 0x7)) as u8);
        let op2 = Reg::from(((opcode >> 3) & 0xF) as u8);
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_2(Operand::reg(op0), Operand::reg(op2)),
            reg_reserve!(op0, op2),
            reg_reserve!(op0),
            1,
        )
    }

    #[inline]
    pub fn cmp_h_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op1 = Reg::from((((opcode >> 4) & 0x8) | (opcode & 0x7)) as u8);
        let op2 = Reg::from(((opcode >> 3) & 0xF) as u8);
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_2(Operand::reg(op1), Operand::reg(op2)),
            reg_reserve!(op1, op2),
            reg_reserve!(Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn mov_h_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((((opcode >> 4) & 0x8) | (opcode & 0x7)) as u8);
        let op2 = Reg::from(((opcode >> 3) & 0xF) as u8);
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_2(Operand::reg(op0), Operand::reg(op2)),
            reg_reserve!(op2),
            reg_reserve!(op0),
            1,
        )
    }

    #[inline]
    pub fn add_pc_t(opcode: u16, op: Op) -> InstInfoThumb {
        todo!()
    }

    #[inline]
    pub fn add_sp_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from(((opcode >> 8) & 0x7) as u8);
        let op2 = (opcode & 0xFF) * 4;
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_3(
                Operand::reg(op0),
                Operand::reg(Reg::SP),
                Operand::imm(op2 as u32),
            ),
            reg_reserve!(Reg::SP),
            reg_reserve!(op0),
            1,
        )
    }

    #[inline]
    pub fn add_sp_imm_t(opcode: u16, op: Op) -> InstInfoThumb {
        let mut op2 = (opcode & 0x7F) as u32;
        if opcode & (1 << 7) != 0 {
            op2 = utils::negative(op2);
        }
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_2(Operand::reg(Reg::SP), Operand::imm(op2)),
            reg_reserve!(Reg::SP),
            reg_reserve!(Reg::SP),
            1,
        )
    }

    #[inline]
    pub fn lsl_imm_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((opcode & 0x7) as u8);
        let op1 = Reg::from(((opcode >> 3) & 0x7) as u8);
        let op2 = (opcode >> 6) & 0x1F;
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_3(
                Operand::reg(op0),
                Operand::reg(op1),
                Operand::imm(op2 as u32),
            ),
            reg_reserve!(op1),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn lsr_imm_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((opcode & 0x7) as u8);
        let op1 = Reg::from(((opcode >> 3) & 0x7) as u8);
        let op2 = (opcode >> 6) & 0x1F;
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_3(
                Operand::reg(op0),
                Operand::reg(op1),
                Operand::imm(op2 as u32),
            ),
            reg_reserve!(op1),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn asr_imm_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((opcode & 0x7) as u8);
        let op1 = Reg::from(((opcode >> 3) & 0x7) as u8);
        let op2 = (opcode >> 6) & 0x1F;
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_3(
                Operand::reg(op0),
                Operand::reg(op1),
                Operand::imm(op2 as u32),
            ),
            reg_reserve!(op1),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn add_imm3_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((opcode & 0x7) as u8);
        let op1 = Reg::from(((opcode >> 3) & 0x7) as u8);
        let op2 = (opcode >> 6) & 0x7;
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_3(
                Operand::reg(op0),
                Operand::reg(op1),
                Operand::imm(op2 as u32),
            ),
            reg_reserve!(op1),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn sub_imm3_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((opcode & 0x7) as u8);
        let op1 = Reg::from(((opcode >> 3) & 0x7) as u8);
        let op2 = (opcode >> 6) & 0x7;
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_3(
                Operand::reg(op0),
                Operand::reg(op1),
                Operand::imm(op2 as u32),
            ),
            reg_reserve!(op1),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn add_imm8_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from(((opcode >> 8) & 0x7) as u8);
        let op2 = opcode & 0xFF;
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_3(
                Operand::reg(op0),
                Operand::reg(op0),
                Operand::imm(op2 as u32),
            ),
            reg_reserve!(op0),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn sub_imm8_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from(((opcode >> 8) & 0x7) as u8);
        let op2 = opcode & 0xFF;
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_3(
                Operand::reg(op0),
                Operand::reg(op0),
                Operand::imm(op2 as u32),
            ),
            reg_reserve!(op0),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn cmp_imm8_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from(((opcode >> 8) & 0x7) as u8);
        let op2 = opcode & 0xFF;
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_2(Operand::reg(op0), Operand::imm(op2 as u32)),
            reg_reserve!(op0),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn mov_imm8_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from(((opcode >> 8) & 0x7) as u8);
        let op2 = opcode & 0xFF;
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_2(Operand::reg(op0), Operand::imm(op2 as u32)),
            reg_reserve!(),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn lsl_dp_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((opcode & 0x7) as u8);
        let op2 = Reg::from(((opcode >> 3) & 0x7) as u8);
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_2(Operand::reg(op0), Operand::reg(op2)),
            reg_reserve!(op0, op2),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn lsr_dp_t(opcode: u16, op: Op) -> InstInfoThumb {
        todo!()
    }

    #[inline]
    pub fn asr_dp_t(opcode: u16, op: Op) -> InstInfoThumb {
        todo!()
    }

    #[inline]
    pub fn ror_dp_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((opcode & 0x7) as u8);
        let op2 = Reg::from(((opcode >> 3) & 0x7) as u8);
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_2(Operand::reg(op0), Operand::reg(op2)),
            reg_reserve!(op0, op2),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn and_dp_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((opcode & 0x7) as u8);
        let op2 = Reg::from(((opcode >> 3) & 0x7) as u8);
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_2(Operand::reg(op0), Operand::reg(op2)),
            reg_reserve!(op0, op2),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn eor_dp_t(opcode: u16, op: Op) -> InstInfoThumb {
        todo!()
    }

    #[inline]
    pub fn adc_dp_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((opcode & 0x7) as u8);
        let op2 = Reg::from(((opcode >> 3) & 0x7) as u8);
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_2(Operand::reg(op0), Operand::reg(op2)),
            reg_reserve!(op0, op2),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn sbc_dp_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((opcode & 0x7) as u8);
        let op2 = Reg::from(((opcode >> 3) & 0x7) as u8);
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_2(Operand::reg(op0), Operand::reg(op2)),
            reg_reserve!(op0, op2),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn tst_dp_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op1 = Reg::from((opcode & 0x7) as u8);
        let op2 = Reg::from(((opcode >> 3) & 0x7) as u8);
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_2(Operand::reg(op1), Operand::reg(op2)),
            reg_reserve!(op1, op2),
            reg_reserve!(Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn cmp_dp_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op1 = Reg::from((opcode & 0x7) as u8);
        let op2 = Reg::from(((opcode >> 3) & 0x7) as u8);
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_2(Operand::reg(op1), Operand::reg(op2)),
            reg_reserve!(op1, op2),
            reg_reserve!(Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn cmn_dp_t(opcode: u16, op: Op) -> InstInfoThumb {
        todo!()
    }

    #[inline]
    pub fn orr_dp_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((opcode & 0x7) as u8);
        let op2 = Reg::from(((opcode >> 3) & 0x7) as u8);
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_2(Operand::reg(op0), Operand::reg(op2)),
            reg_reserve!(op0, op2),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn bic_dp_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((opcode & 0x7) as u8);
        let op2 = Reg::from(((opcode >> 3) & 0x7) as u8);
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_2(Operand::reg(op0), Operand::reg(op2)),
            reg_reserve!(op0, op2),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn mvn_dp_t(opcode: u16, op: Op) -> InstInfoThumb {
        todo!()
    }

    #[inline]
    pub fn neg_dp_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((opcode & 0x7) as u8);
        let op2 = Reg::from(((opcode >> 3) & 0x7) as u8);
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_2(Operand::reg(op0), Operand::reg(op2)),
            reg_reserve!(op0, op2),
            reg_reserve!(op0, Reg::CPSR),
            1,
        )
    }

    #[inline]
    pub fn mul_dp_t(opcode: u16, op: Op) -> InstInfoThumb {
        let op0 = Reg::from((opcode & 0x7) as u8);
        let op2 = Reg::from(((opcode >> 3) & 0x7) as u8);
        InstInfoThumb::new(
            opcode,
            op,
            Operands::new_2(Operand::reg(op0), Operand::reg(op2)),
            reg_reserve!(op0, op2),
            reg_reserve!(op0, Reg::CPSR),
            4,
        )
    }
}

pub use alu_thumb_ops::*;
