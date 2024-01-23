use crate::hle::bios_context::BiosContext;
use crate::hle::cp15_context::Cp15Context;
use crate::hle::CpuType;

#[repr(u8)]
pub enum ExceptionVector {
    Reset = 0x0,
    UndefinedInstruction = 0x4,
    SoftwareInterrupt = 0x8,
    PrefetchAbort = 0xC,
    DataAbort = 0x10,
    AddressExceeds26Bit = 0x14,
    NormalInterrupt = 0x18,
    FastInterrupt = 0x1C,
}

mod exception_handler {
    use crate::hle::bios_context::BiosContext;
    use crate::hle::exception_handler::ExceptionVector;
    use crate::hle::CpuType;

    pub fn handle<const CPU: CpuType, const THUMB: bool>(
        exception_addr: Option<u32>,
        dtcm_addr: Option<u32>,
        bios_context: &mut BiosContext<CPU>,
        opcode: u32,
        vector: ExceptionVector,
    ) {
        if CPU == CpuType::ARM7 || exception_addr.unwrap() != 0 {
            match vector {
                ExceptionVector::SoftwareInterrupt => {
                    bios_context.swi(((opcode >> if THUMB { 0 } else { 16 }) & 0xFF) as u8);
                }
                ExceptionVector::NormalInterrupt => {
                    bios_context.interrupt(dtcm_addr);
                }
                _ => todo!(),
            }
        } else {
            todo!()
        }
    }
}

pub(super) use exception_handler::*;

pub unsafe extern "C" fn exception_handler_arm9<const THUMB: bool>(
    cp15_context: *const Cp15Context,
    bios_context: *mut BiosContext<{ CpuType::ARM9 }>,
    opcode: u32,
    vector: ExceptionVector,
) {
    let cp15_context = cp15_context.as_ref().unwrap();
    handle::<{ CpuType::ARM9 }, THUMB>(
        Some(cp15_context.exception_addr),
        Some(cp15_context.dtcm_addr),
        bios_context.as_mut().unwrap(),
        opcode,
        vector,
    )
}

pub unsafe extern "C" fn exception_handler_arm7<const THUMB: bool>(
    bios_context: *mut BiosContext<{ CpuType::ARM7 }>,
    opcode: u32,
    vector: ExceptionVector,
) {
    handle::<{ CpuType::ARM7 }, THUMB>(None, None, bios_context.as_mut().unwrap(), opcode, vector)
}
