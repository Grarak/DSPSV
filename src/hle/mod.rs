use std::ops;

mod bios;
mod bios_lookup_table;
pub mod cp15_context;
mod cpu_regs;
pub mod exception_handler;
mod gpu;
pub mod ipc_handler;
pub mod memory;
mod spu_context;
pub mod thread_context;
pub mod thread_regs;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum CpuType {
    ARM9 = 0,
    ARM7 = 1,
}

impl ops::Not for CpuType {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            CpuType::ARM9 => CpuType::ARM7,
            CpuType::ARM7 => CpuType::ARM9,
        }
    }
}

impl Default for CpuType {
    fn default() -> Self {
        CpuType::ARM9
    }
}
