use crate::hle::cp15::TcmState;
use crate::hle::hle::{get_cp15, Hle};
use crate::hle::memory::bios::{BiosArm7, BiosArm9};
use crate::hle::memory::io_arm7::IoArm7;
use crate::hle::memory::io_arm9::IoArm9;
use crate::hle::memory::main::Main;
use crate::hle::memory::mmu::{MmuArm7, MmuArm9, MMU_BLOCK_SIZE};
use crate::hle::memory::oam::Oam;
use crate::hle::memory::palettes::Palettes;
use crate::hle::memory::regions;
use crate::hle::memory::tcm::Tcm;
use crate::hle::memory::vram::Vram;
use crate::hle::memory::wram::Wram;
use crate::hle::CpuType;
use crate::hle::CpuType::ARM9;
use crate::jit::jit_memory::JitMemory;
use crate::logging::debug_println;
use crate::utils::Convert;
use std::intrinsics::{likely, unlikely};
use std::mem;
use CpuType::ARM7;

pub struct Memory {
    pub tcm: Tcm,
    pub main: Main,
    pub wram: Wram,
    pub io_arm7: IoArm7,
    pub io_arm9: IoArm9,
    pub palettes: Palettes,
    pub vram: Vram,
    pub oam: Oam,
    pub jit: JitMemory,
    pub bios_arm9: BiosArm9,
    pub bios_arm7: BiosArm7,
    pub current_jit_block_range: (u32, u32), // Check if the jit block we are currently executing gets written to
    pub breakout_imm: bool,
    pub mmu_arm9: MmuArm9,
    pub mmu_arm7: MmuArm7,
}

macro_rules! get_mem_mmu {
    ($mem:expr, $cpu:expr) => {{
        match $cpu {
            crate::hle::CpuType::ARM9 => &$mem.mmu_arm9 as &dyn crate::hle::memory::mmu::Mmu,
            crate::hle::CpuType::ARM7 => &$mem.mmu_arm7 as &dyn crate::hle::memory::mmu::Mmu,
        }
    }};
}

impl Memory {
    pub fn new() -> Self {
        Memory {
            tcm: Tcm::new(),
            main: Main::new(),
            wram: Wram::new(),
            io_arm7: IoArm7::new(),
            io_arm9: IoArm9::new(),
            palettes: Palettes::new(),
            vram: Vram::new(),
            oam: Oam::new(),
            jit: JitMemory::new(),
            bios_arm9: BiosArm9::new(),
            bios_arm7: BiosArm7::new(),
            current_jit_block_range: (0, 0),
            breakout_imm: false,
            mmu_arm9: MmuArm9::new(),
            mmu_arm7: MmuArm7::new(),
        }
    }

    pub fn read<const CPU: CpuType, T: Convert>(&mut self, addr: u32, hle: &mut Hle) -> T {
        self.read_internal::<CPU, true, T>(addr, hle)
    }

    pub fn read_no_tcm<const CPU: CpuType, T: Convert>(&mut self, addr: u32, hle: &mut Hle) -> T {
        self.read_internal::<CPU, false, T>(addr, hle)
    }

    fn read_internal<const CPU: CpuType, const TCM: bool, T: Convert>(
        &mut self,
        addr: u32,
        hle: &mut Hle,
    ) -> T {
        debug_println!("{:?} memory read at {:x}", CPU, addr);
        let aligned_addr = addr & !(mem::size_of::<T>() as u32 - 1);

        let addr_base = aligned_addr & 0xFF000000;
        let addr_offset = aligned_addr - addr_base;

        if CPU == ARM7 || TCM {
            let base_ptr = get_mem_mmu!(self, CPU).get_base_ptr(aligned_addr);
            if likely(!base_ptr.is_null()) {
                return unsafe {
                    (base_ptr.add((aligned_addr & (MMU_BLOCK_SIZE - 1)) as usize) as *const T)
                        .read()
                };
            }
        }

        if CPU == ARM9 && TCM {
            let cp15 = get_cp15!(hle, ARM9);
            if aligned_addr >= cp15.dtcm_addr
                && aligned_addr < cp15.dtcm_addr + cp15.dtcm_size
                && cp15.dtcm_state == TcmState::RW
            {
                return self.tcm.read_dtcm(aligned_addr - cp15.dtcm_addr);
            }
        }

        let ret = match addr_base {
            regions::INSTRUCTION_TCM_OFFSET | regions::INSTRUCTION_TCM_MIRROR_OFFSET => match CPU {
                ARM9 => {
                    let mut ret = T::from(0);
                    if TCM {
                        let cp15 = get_cp15!(hle, ARM9);
                        if aligned_addr < cp15.itcm_size && cp15.itcm_state == TcmState::RW {
                            ret = self.tcm.read_itcm(addr_offset);
                        }
                    }
                    ret
                }
                // Bios of arm7 has same offset as itcm on arm9
                ARM7 => {
                    if aligned_addr < regions::ARM7_BIOS_SIZE {
                        self.bios_arm7.read(aligned_addr)
                    } else {
                        todo!("{:x} {:x}", aligned_addr, addr_base)
                    }
                }
            },
            regions::MAIN_MEMORY_OFFSET => self.main.read(addr_offset),
            regions::SHARED_WRAM_OFFSET => self.wram.read::<CPU, _>(addr_offset),
            regions::IO_PORTS_OFFSET => match CPU {
                ARM9 => self.io_arm9.read(addr_offset, hle),
                ARM7 => self.io_arm7.read(addr_offset, hle),
            },
            regions::STANDARD_PALETTES_OFFSET => self.palettes.read(addr_offset),
            regions::VRAM_OFFSET => self.vram.read::<CPU, _>(addr_offset),
            regions::OAM_OFFSET => self.oam.read(addr_offset),
            regions::GBA_ROM_OFFSET | regions::GBA_ROM_OFFSET2 | regions::GBA_RAM_OFFSET => {
                T::from(0xFFFFFFFF)
            }
            0xFF000000 => match CPU {
                ARM9 => {
                    if (aligned_addr & 0xFFFF8000) == regions::ARM9_BIOS_OFFSET {
                        self.bios_arm9.read(aligned_addr)
                    } else {
                        todo!("{:x} {:x}", aligned_addr, addr_base)
                    }
                }
                ARM7 => {
                    todo!("{:x} {:x}", aligned_addr, addr_base)
                }
            },
            _ => {
                todo!("{:?} {:x} {:x}", CPU, aligned_addr, addr_base)
            }
        };

        debug_println!(
            "{:?} memory read at {:x} with value {:x}",
            CPU,
            addr,
            ret.into()
        );

        ret
    }

    pub fn write<const CPU: CpuType, T: Convert>(&mut self, addr: u32, value: T, hle: &mut Hle) {
        self.write_internal::<CPU, true, T>(addr, value, hle)
    }

    pub fn write_no_tcm<const CPU: CpuType, T: Convert>(
        &mut self,
        addr: u32,
        value: T,
        hle: &mut Hle,
    ) {
        self.write_internal::<CPU, false, T>(addr, value, hle)
    }

    fn write_internal<const CPU: CpuType, const TCM: bool, T: Convert>(
        &mut self,
        addr: u32,
        value: T,
        hle: &mut Hle,
    ) {
        debug_println!(
            "{:?} memory write at {:x} with value {:x}",
            CPU,
            addr,
            value.into(),
        );
        let aligned_addr = addr & !(mem::size_of::<T>() as u32 - 1);

        let addr_base = aligned_addr & 0xFF000000;
        let addr_offset = aligned_addr - addr_base;

        if CPU == ARM9 && TCM {
            let cp15 = get_cp15!(hle, ARM9);
            if aligned_addr >= cp15.dtcm_addr
                && aligned_addr < cp15.dtcm_addr + cp15.dtcm_size
                && cp15.dtcm_state != TcmState::Disabled
            {
                self.tcm.write_dtcm(aligned_addr - cp15.dtcm_addr, value);
                return;
            }
        }

        let mut invalidate_jit = |size: u32, offset: u32| {
            let jit_block_region = self.current_jit_block_range.0 & 0xFF000000;
            if addr_base == jit_block_region {
                let addr_base = aligned_addr & !(offset - 1);
                let current_jit_block_base = self.current_jit_block_range.0 & !(offset - 1);
                if unlikely(
                    addr_base == current_jit_block_base
                        && addr_offset & (size - 1) >= self.current_jit_block_range.0 & (size - 1)
                        && addr_offset & (size - 1) <= self.current_jit_block_range.1 & (size - 1),
                ) {
                    self.breakout_imm = true;
                }
            }

            self.jit
                .invalidate_block::<CPU>(aligned_addr, mem::size_of::<T>() as u32);
        };

        match addr_base {
            regions::INSTRUCTION_TCM_OFFSET | regions::INSTRUCTION_TCM_MIRROR_OFFSET => match CPU {
                ARM9 => {
                    if TCM {
                        let cp15 = get_cp15!(hle, ARM9);
                        if aligned_addr < cp15.itcm_size && cp15.itcm_state != TcmState::Disabled {
                            self.tcm.write_itcm(addr_offset, value);
                            invalidate_jit(
                                regions::INSTRUCTION_TCM_SIZE,
                                regions::INSTRUCTION_TCM_OFFSET,
                            );
                        }
                    }
                }
                // Bios of arm7 has same offset as itcm on arm9
                ARM7 => {
                    todo!("{:x} {:x}", aligned_addr, addr_base)
                }
            },
            regions::MAIN_MEMORY_OFFSET => {
                self.main.write(addr_offset, value);
                invalidate_jit(regions::MAIN_MEMORY_SIZE, regions::MAIN_MEMORY_OFFSET);
            }
            regions::SHARED_WRAM_OFFSET => {
                let (size, offset) = self.wram.write::<CPU, _>(addr_offset, value);
                invalidate_jit(size, offset);
            }
            regions::IO_PORTS_OFFSET => match CPU {
                ARM9 => self.io_arm9.write(addr_offset, value, hle),
                ARM7 => self.io_arm7.write(addr_offset, value, hle),
            },
            regions::STANDARD_PALETTES_OFFSET => self.palettes.write(addr_offset, value),
            regions::VRAM_OFFSET => self.vram.write::<CPU, _>(addr_offset, value),
            regions::OAM_OFFSET => self.oam.write(addr_offset, value),
            regions::GBA_ROM_OFFSET => {}
            _ => {
                todo!("{:x} {:x}", aligned_addr, addr_base)
            }
        };
    }
}
