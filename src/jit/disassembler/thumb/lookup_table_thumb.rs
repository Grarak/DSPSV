use crate::jit::disassembler::thumb::alu_instructions_thumb::*;
use crate::jit::disassembler::thumb::branch_instructions_thumb::*;
use crate::jit::disassembler::thumb::delegations_thumb::*;
use crate::jit::disassembler::thumb::transfer_instructions_thumb::*;
use crate::jit::inst_info_thumb::InstInfoThumb;
use crate::jit::Op;
use crate::jit::Op::*;

pub const fn lookup_thumb_opcode(opcode: u16) -> &'static (Op, fn(u16, Op) -> InstInfoThumb) {
    &LOOKUP_TABLE[((opcode >> 6) & 0x3FF) as usize]
}

const LOOKUP_TABLE: [(Op, fn(u16, Op) -> InstInfoThumb); 1024] = [
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LslImmT, lsl_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (LsrImmT, lsr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AsrImmT, asr_imm_t),
    (AddRegT, add_reg_t),
    (AddRegT, add_reg_t),
    (AddRegT, add_reg_t),
    (AddRegT, add_reg_t),
    (AddRegT, add_reg_t),
    (AddRegT, add_reg_t),
    (AddRegT, add_reg_t),
    (AddRegT, add_reg_t),
    (SubRegT, sub_reg_t),
    (SubRegT, sub_reg_t),
    (SubRegT, sub_reg_t),
    (SubRegT, sub_reg_t),
    (SubRegT, sub_reg_t),
    (SubRegT, sub_reg_t),
    (SubRegT, sub_reg_t),
    (SubRegT, sub_reg_t),
    (AddImm3T, add_imm3_t),
    (AddImm3T, add_imm3_t),
    (AddImm3T, add_imm3_t),
    (AddImm3T, add_imm3_t),
    (AddImm3T, add_imm3_t),
    (AddImm3T, add_imm3_t),
    (AddImm3T, add_imm3_t),
    (AddImm3T, add_imm3_t),
    (SubImm3T, sub_imm3_t),
    (SubImm3T, sub_imm3_t),
    (SubImm3T, sub_imm3_t),
    (SubImm3T, sub_imm3_t),
    (SubImm3T, sub_imm3_t),
    (SubImm3T, sub_imm3_t),
    (SubImm3T, sub_imm3_t),
    (SubImm3T, sub_imm3_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (MovImm8T, mov_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (CmpImm8T, cmp_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (AddImm8T, add_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (SubImm8T, sub_imm8_t),
    (AndDpT, and_dp_t),
    (EorDpT, eor_dp_t),
    (LslDpT, lsl_dp_t),
    (LsrDpT, lsr_dp_t),
    (AsrDpT, asr_dp_t),
    (AdcDpT, adc_dp_t),
    (SbcDpT, sbc_dp_t),
    (RorDpT, ror_dp_t),
    (TstDpT, tst_dp_t),
    (NegDpT, neg_dp_t),
    (CmpDpT, cmp_dp_t),
    (CmnDpT, cmn_dp_t),
    (OrrDpT, orr_dp_t),
    (MulDpT, mul_dp_t),
    (BicDpT, bic_dp_t),
    (MvnDpT, mvn_dp_t),
    (AddHT, add_h_t),
    (AddHT, add_h_t),
    (AddHT, add_h_t),
    (AddHT, add_h_t),
    (CmpHT, cmp_h_t),
    (CmpHT, cmp_h_t),
    (CmpHT, cmp_h_t),
    (CmpHT, cmp_h_t),
    (MovHT, mov_h_t),
    (MovHT, mov_h_t),
    (MovHT, mov_h_t),
    (MovHT, mov_h_t),
    (BxRegT, bx_reg_t),
    (BxRegT, bx_reg_t),
    (BlxRegT, blx_reg_t),
    (BlxRegT, blx_reg_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (LdrPcT, ldr_pc_t),
    (StrRegT, str_reg_t),
    (StrRegT, str_reg_t),
    (StrRegT, str_reg_t),
    (StrRegT, str_reg_t),
    (StrRegT, str_reg_t),
    (StrRegT, str_reg_t),
    (StrRegT, str_reg_t),
    (StrRegT, str_reg_t),
    (StrhRegT, strh_reg_t),
    (StrhRegT, strh_reg_t),
    (StrhRegT, strh_reg_t),
    (StrhRegT, strh_reg_t),
    (StrhRegT, strh_reg_t),
    (StrhRegT, strh_reg_t),
    (StrhRegT, strh_reg_t),
    (StrhRegT, strh_reg_t),
    (StrbRegT, strb_reg_t),
    (StrbRegT, strb_reg_t),
    (StrbRegT, strb_reg_t),
    (StrbRegT, strb_reg_t),
    (StrbRegT, strb_reg_t),
    (StrbRegT, strb_reg_t),
    (StrbRegT, strb_reg_t),
    (StrbRegT, strb_reg_t),
    (LdrsbRegT, ldrsb_reg_t),
    (LdrsbRegT, ldrsb_reg_t),
    (LdrsbRegT, ldrsb_reg_t),
    (LdrsbRegT, ldrsb_reg_t),
    (LdrsbRegT, ldrsb_reg_t),
    (LdrsbRegT, ldrsb_reg_t),
    (LdrsbRegT, ldrsb_reg_t),
    (LdrsbRegT, ldrsb_reg_t),
    (LdrRegT, ldr_reg_t),
    (LdrRegT, ldr_reg_t),
    (LdrRegT, ldr_reg_t),
    (LdrRegT, ldr_reg_t),
    (LdrRegT, ldr_reg_t),
    (LdrRegT, ldr_reg_t),
    (LdrRegT, ldr_reg_t),
    (LdrRegT, ldr_reg_t),
    (LdrhRegT, ldrh_reg_t),
    (LdrhRegT, ldrh_reg_t),
    (LdrhRegT, ldrh_reg_t),
    (LdrhRegT, ldrh_reg_t),
    (LdrhRegT, ldrh_reg_t),
    (LdrhRegT, ldrh_reg_t),
    (LdrhRegT, ldrh_reg_t),
    (LdrhRegT, ldrh_reg_t),
    (LdrbRegT, ldrb_reg_t),
    (LdrbRegT, ldrb_reg_t),
    (LdrbRegT, ldrb_reg_t),
    (LdrbRegT, ldrb_reg_t),
    (LdrbRegT, ldrb_reg_t),
    (LdrbRegT, ldrb_reg_t),
    (LdrbRegT, ldrb_reg_t),
    (LdrbRegT, ldrb_reg_t),
    (LdrshRegT, ldrsh_reg_t),
    (LdrshRegT, ldrsh_reg_t),
    (LdrshRegT, ldrsh_reg_t),
    (LdrshRegT, ldrsh_reg_t),
    (LdrshRegT, ldrsh_reg_t),
    (LdrshRegT, ldrsh_reg_t),
    (LdrshRegT, ldrsh_reg_t),
    (LdrshRegT, ldrsh_reg_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (StrImm5T, str_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (LdrImm5T, ldr_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (StrbImm5T, strb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (LdrbImm5T, ldrb_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (StrhImm5T, strh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (LdrhImm5T, ldrh_imm5_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (StrSpT, str_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (LdrSpT, ldr_sp_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddPcT, add_pc_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpT, add_sp_t),
    (AddSpImmT, add_sp_imm_t),
    (AddSpImmT, add_sp_imm_t),
    (AddSpImmT, add_sp_imm_t),
    (AddSpImmT, add_sp_imm_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (PushT, push_t),
    (PushT, push_t),
    (PushT, push_t),
    (PushT, push_t),
    (PushLrT, push_lr_t),
    (PushLrT, push_lr_t),
    (PushLrT, push_lr_t),
    (PushLrT, push_lr_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (PopT, pop_t),
    (PopT, pop_t),
    (PopT, pop_t),
    (PopT, pop_t),
    (PopPcT, pop_pc_t),
    (PopPcT, pop_pc_t),
    (PopPcT, pop_pc_t),
    (PopPcT, pop_pc_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (StmiaT, stmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (LdmiaT, ldmia_t),
    (BeqT, beq_t),
    (BeqT, beq_t),
    (BeqT, beq_t),
    (BeqT, beq_t),
    (BneT, bne_t),
    (BneT, bne_t),
    (BneT, bne_t),
    (BneT, bne_t),
    (BcsT, bcs_t),
    (BcsT, bcs_t),
    (BcsT, bcs_t),
    (BcsT, bcs_t),
    (BccT, bcc_t),
    (BccT, bcc_t),
    (BccT, bcc_t),
    (BccT, bcc_t),
    (BmiT, bmi_t),
    (BmiT, bmi_t),
    (BmiT, bmi_t),
    (BmiT, bmi_t),
    (BplT, bpl_t),
    (BplT, bpl_t),
    (BplT, bpl_t),
    (BplT, bpl_t),
    (BvsT, bvs_t),
    (BvsT, bvs_t),
    (BvsT, bvs_t),
    (BvsT, bvs_t),
    (BvcT, bvc_t),
    (BvcT, bvc_t),
    (BvcT, bvc_t),
    (BvcT, bvc_t),
    (BhiT, bhi_t),
    (BhiT, bhi_t),
    (BhiT, bhi_t),
    (BhiT, bhi_t),
    (BlsT, bls_t),
    (BlsT, bls_t),
    (BlsT, bls_t),
    (BlsT, bls_t),
    (BgeT, bge_t),
    (BgeT, bge_t),
    (BgeT, bge_t),
    (BgeT, bge_t),
    (BltT, blt_t),
    (BltT, blt_t),
    (BltT, blt_t),
    (BltT, blt_t),
    (BgtT, bgt_t),
    (BgtT, bgt_t),
    (BgtT, bgt_t),
    (BgtT, bgt_t),
    (BleT, ble_t),
    (BleT, ble_t),
    (BleT, ble_t),
    (BleT, ble_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (UnkThumb, unk_t),
    (SwiT, swi_t),
    (SwiT, swi_t),
    (SwiT, swi_t),
    (SwiT, swi_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BT, b_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlxOffT, blx_off_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlSetupT, bl_setup_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
    (BlOffT, bl_off_t),
];
