use crate::hle::memory::io_ports::IoPorts;
use crate::hle::CpuType;
use crate::logging::debug_println;
use crate::utils::Convert;
use crate::DEBUG;
use dspsv_macros::{io_ports_read, io_ports_write};
use std::mem;

impl<const CPU: CpuType> IoPorts<CPU> {
    pub fn read_arm7<T: Convert>(&self, addr_offset: u32) -> T {
        /*
         * Use moving windows to handle reads and writes
         * |0|0|0|  x  |   x   |   x   |   x   |0|0|0|
         *         addr   + 1     + 2     + 3
         */
        let mut bytes_window = [0u8; 10];

        let mut addr_offset_tmp = addr_offset;
        let mut index = 3usize;
        while (index - 3) < mem::size_of::<T>() {
            #[rustfmt::skip]
            io_ports_read!(match addr_offset + (index - 3) as u32 {
                io16(0x4) => self.gpu_context.read().unwrap().get_disp_stat(CpuType::ARM7),
                io16(0x6) => todo!(),
                io32(0xB0) => todo!(),
                io32(0xB4) => todo!(),
                io32(0xB8) => todo!(),
                io32(0xBC) => todo!(),
                io32(0xC0) => todo!(),
                io32(0xC4) => todo!(),
                io32(0xC8) => todo!(),
                io32(0xCC) => todo!(),
                io32(0xD0) => todo!(),
                io32(0xD4) => todo!(),
                io32(0xD8) => todo!(),
                io32(0xDC) => self.dma.read().unwrap().get_cnt(3),
                io16(0x100) => todo!(),
                io16(0x102) => todo!(),
                io16(0x104) => todo!(),
                io16(0x106) => todo!(),
                io16(0x108) => todo!(),
                io16(0x10A) => todo!(),
                io16(0x10C) => todo!(),
                io16(0x10E) => todo!(),
                io16(0x130) => self.input_context.read().unwrap().key_input,
                io16(0x136) => self.input_context.read().unwrap().ext_key_in,
                io8(0x138) => self.rtc_context.borrow().get_rtc(),
                io16(0x180) => self.ipc_handler.read().unwrap().get_sync_reg::<{ CpuType::ARM7 }>(),
                io16(0x184) => self.ipc_handler.read().unwrap().get_fifo_cnt::<{ CpuType::ARM7 }>(),
                io16(0x1A0) => todo!(),
                io8(0x1A2) => todo!(),
                io32(0x1A4) => todo!(),
                io16(0x1C0) => self.spi_context.read().unwrap().cnt,
                io8(0x1C2) => self.spi_context.read().unwrap().data,
                io8(0x208) => self.cpu_regs.get_ime(),
                io32(0x210) => self.cpu_regs.get_ie(),
                io32(0x214) => self.cpu_regs.get_irf(),
                io8(0x240) => self.vram_context.get_stat(),
                io8(0x241) => self.wram_context.get_cnt(),
                io8(0x300) => todo!(),
                io8(0x301) => todo!(),
                io32(0x400) => todo!(),
                io32(0x410) => todo!(),
                io32(0x420) => todo!(),
                io32(0x430) => todo!(),
                io32(0x440) => todo!(),
                io32(0x450) => todo!(),
                io32(0x460) => todo!(),
                io32(0x470) => todo!(),
                io32(0x480) => todo!(),
                io32(0x490) => todo!(),
                io32(0x4A0) => todo!(),
                io32(0x4B0) => todo!(),
                io32(0x4C0) => todo!(),
                io32(0x4D0) => todo!(),
                io32(0x4E0) => todo!(),
                io32(0x4F0) => todo!(),
                io16(0x500) => self.spu_context.borrow().main_sound_cnt,
                io16(0x504) => todo!(),
                io8(0x508) => todo!(),
                io8(0x509) => todo!(),
                io32(0x510) => todo!(),
                io32(0x518) => todo!(),
                io32(0x100000) => todo!(),
                io32(0x100010) => todo!(),
                io16(0x800006) => todo!(),
                io16(0x800010) => todo!(),
                io16(0x800012) => todo!(),
                io16(0x800018) => todo!(),
                io16(0x80001A) => todo!(),
                io16(0x80001C) => todo!(),
                io16(0x800020) => todo!(),
                io16(0x800022) => todo!(),
                io16(0x800024) => todo!(),
                io16(0x80002A) => todo!(),
                io16(0x800030) => todo!(),
                io16(0x80003C) => todo!(),
                io16(0x800040) => todo!(),
                io16(0x800050) => todo!(),
                io16(0x800052) => todo!(),
                io16(0x800054) => todo!(),
                io16(0x800056) => todo!(),
                io16(0x800058) => todo!(),
                io16(0x80005A) => todo!(),
                io16(0x80005C) => todo!(),
                io16(0x800060) => todo!(),
                io16(0x800062) => todo!(),
                io16(0x800064) => todo!(),
                io16(0x800068) => todo!(),
                io16(0x80006C) => todo!(),
                io16(0x800074) => todo!(),
                io16(0x800076) => todo!(),
                io16(0x800080) => todo!(),
                io16(0x80008C) => todo!(),
                io16(0x800090) => todo!(),
                io16(0x8000A0) => todo!(),
                io16(0x8000A4) => todo!(),
                io16(0x8000A8) => todo!(),
                io16(0x8000B0) => todo!(),
                io16(0x8000E8) => todo!(),
                io16(0x8000EA) => todo!(),
                io16(0x800110) => todo!(),
                io16(0x80011C) => todo!(),
                io16(0x800120) => todo!(),
                io16(0x800122) => todo!(),
                io16(0x800124) => todo!(),
                io16(0x800128) => todo!(),
                io16(0x800130) => todo!(),
                io16(0x800132) => todo!(),
                io16(0x800134) => todo!(),
                io16(0x800140) => todo!(),
                io16(0x800142) => todo!(),
                io16(0x800144) => todo!(),
                io16(0x800146) => todo!(),
                io16(0x800148) => todo!(),
                io16(0x80014A) => todo!(),
                io16(0x80014C) => todo!(),
                io16(0x800150) => todo!(),
                io16(0x800154) => todo!(),
                io16(0x80015C) => todo!(),
                _ => {
                    if DEBUG && index == 3 {
                        debug_println!(
                            "{:?} unknown io port read at {:x}",
                            CpuType::ARM7,
                            addr_offset
                        );
                    }

                    bytes_window[index] = 0;
                }
            });
            index += 1;
        }
        T::from(u32::from_le_bytes([
            bytes_window[3],
            bytes_window[4],
            bytes_window[5],
            bytes_window[6],
        ]))
    }

    pub fn write_arm7<T: Convert>(&self, addr_offset: u32, value: T) {
        let value_array = [value];
        let (_, bytes, _) = unsafe { value_array.align_to::<u8>() };
        /*
         * Use moving windows to handle reads and writes
         * |0|0|0|  x  |   x   |   x   |   x   |0|0|0|
         *         addr   + 1     + 2     + 3
         */
        let mut bytes_window = [0u8; 10];
        let mut mask_window = [0u8; 10];
        bytes_window[3..3 + mem::size_of::<T>()].copy_from_slice(bytes);
        mask_window[3..3 + mem::size_of::<T>()].fill(0xFF);

        let mut addr_offset_tmp = addr_offset;
        let mut index = 3usize;
        while (index - 3) < bytes.len() {
            #[rustfmt::skip]
            io_ports_write!(match addr_offset + (index - 3) as u32 {
                io16(0x4) => self.gpu_context.write().unwrap().set_disp_stat::<{ CpuType::ARM7 }>(mask, value),
                io32(0xB0) => todo!(),
                io32(0xB4) => todo!(),
                io32(0xB8) => todo!(),
                io32(0xBC) => todo!(),
                io32(0xC0) => todo!(),
                io32(0xC4) => todo!(),
                io32(0xC8) => todo!(),
                io32(0xCC) => todo!(),
                io32(0xD0) => todo!(),
                io32(0xD4) => self.dma.write().unwrap().set_sad(3, mask, value),
                io32(0xD8) => self.dma.write().unwrap().set_dad(3, mask, value),
                io32(0xDC) => self.dma.write().unwrap().set_cnt(3, mask, value),
                io16(0x100) => todo!(),
                io16(0x102) => todo!(),
                io16(0x104) => todo!(),
                io16(0x106) => todo!(),
                io16(0x108) => todo!(),
                io16(0x10A) => todo!(),
                io16(0x10C) => todo!(),
                io16(0x10E) => todo!(),
                io8(0x138) => self.rtc_context.borrow_mut().set_rtc(value),
                io16(0x180) => self.ipc_handler.write().unwrap().set_sync_reg::<{ CpuType::ARM7 }>(mask, value),
                io16(0x184) => self.ipc_handler.write().unwrap().set_fifo_cnt::<{ CpuType::ARM7 }>(mask, value),
                io32(0x188) => self.ipc_handler.write().unwrap().fifo_send::<{ CpuType::ARM7 }>(mask, value),
                io16(0x1A0) => todo!(),
                io8(0x1A2) => todo!(),
                io32(0x1A4) => todo!(),
                io32(0x1A8) => todo!(),
                io32(0x1AC) => todo!(),
                io16(0x1C0) => self.spi_context.write().unwrap().set_cnt(mask, value),
                io8(0x1C2) => self.spi_context.write().unwrap().set_data(value),
                io8(0x208) => self.cpu_regs.set_ime(value),
                io32(0x210) => self.cpu_regs.set_ie(mask, value),
                io32(0x214) => self.cpu_regs.set_irf(mask, value),
                io8(0x300) => self.cpu_regs.set_post_flg(value),
                io8(0x301) => todo!(),
                io32(0x400) => self.spu_context.borrow_mut().set_cnt(0, mask, value),
                io32(0x404) => self.spu_context.borrow_mut().set_sad(0, mask, value),
                io16(0x408) => self.spu_context.borrow_mut().set_tmr(0, mask, value),
                io16(0x40A) => self.spu_context.borrow_mut().set_pnt(0, mask, value),
                io32(0x40C) => self.spu_context.borrow_mut().set_len(0, mask, value),
                io32(0x410) => self.spu_context.borrow_mut().set_cnt(1, mask, value),
                io32(0x414) => self.spu_context.borrow_mut().set_sad(1, mask, value),
                io16(0x418) => self.spu_context.borrow_mut().set_tmr(1, mask, value),
                io16(0x41A) => self.spu_context.borrow_mut().set_pnt(1, mask, value),
                io32(0x41C) => self.spu_context.borrow_mut().set_len(1, mask, value),
                io32(0x420) => self.spu_context.borrow_mut().set_cnt(2, mask, value),
                io32(0x424) => self.spu_context.borrow_mut().set_sad(2, mask, value),
                io16(0x428) => self.spu_context.borrow_mut().set_tmr(2, mask, value),
                io16(0x42A) => self.spu_context.borrow_mut().set_pnt(2, mask, value),
                io32(0x42C) => self.spu_context.borrow_mut().set_len(2, mask, value),
                io32(0x430) => self.spu_context.borrow_mut().set_cnt(3, mask, value),
                io32(0x434) => self.spu_context.borrow_mut().set_sad(3, mask, value),
                io16(0x438) => self.spu_context.borrow_mut().set_tmr(3, mask, value),
                io16(0x43A) => self.spu_context.borrow_mut().set_pnt(3, mask, value),
                io32(0x43C) => self.spu_context.borrow_mut().set_len(3, mask, value),
                io32(0x440) => self.spu_context.borrow_mut().set_cnt(4, mask, value),
                io32(0x444) => self.spu_context.borrow_mut().set_sad(4, mask, value),
                io16(0x448) => self.spu_context.borrow_mut().set_tmr(4, mask, value),
                io16(0x44A) => self.spu_context.borrow_mut().set_pnt(4, mask, value),
                io32(0x44C) => self.spu_context.borrow_mut().set_len(4, mask, value),
                io32(0x450) => self.spu_context.borrow_mut().set_cnt(5, mask, value),
                io32(0x454) => self.spu_context.borrow_mut().set_sad(5, mask, value),
                io16(0x458) => self.spu_context.borrow_mut().set_tmr(5, mask, value),
                io16(0x45A) => self.spu_context.borrow_mut().set_pnt(5, mask, value),
                io32(0x45C) => self.spu_context.borrow_mut().set_len(5, mask, value),
                io32(0x460) => self.spu_context.borrow_mut().set_cnt(6, mask, value),
                io32(0x464) => self.spu_context.borrow_mut().set_sad(6, mask, value),
                io16(0x468) => self.spu_context.borrow_mut().set_tmr(6, mask, value),
                io16(0x46A) => self.spu_context.borrow_mut().set_pnt(6, mask, value),
                io32(0x46C) => self.spu_context.borrow_mut().set_len(6, mask, value),
                io32(0x470) => self.spu_context.borrow_mut().set_cnt(7, mask, value),
                io32(0x474) => self.spu_context.borrow_mut().set_sad(7, mask, value),
                io16(0x478) => self.spu_context.borrow_mut().set_tmr(7, mask, value),
                io16(0x47A) => self.spu_context.borrow_mut().set_pnt(7, mask, value),
                io32(0x47C) => self.spu_context.borrow_mut().set_len(7, mask, value),
                io32(0x480) => self.spu_context.borrow_mut().set_cnt(8, mask, value),
                io32(0x484) => self.spu_context.borrow_mut().set_sad(8, mask, value),
                io16(0x488) => self.spu_context.borrow_mut().set_tmr(8, mask, value),
                io16(0x48A) => self.spu_context.borrow_mut().set_pnt(8, mask, value),
                io32(0x48C) => self.spu_context.borrow_mut().set_len(8, mask, value),
                io32(0x490) => self.spu_context.borrow_mut().set_cnt(9, mask, value),
                io32(0x494) => self.spu_context.borrow_mut().set_sad(9, mask, value),
                io16(0x498) => self.spu_context.borrow_mut().set_tmr(9, mask, value),
                io16(0x49A) => self.spu_context.borrow_mut().set_pnt(9, mask, value),
                io32(0x49C) => self.spu_context.borrow_mut().set_len(9, mask, value),
                io32(0x4A0) => self.spu_context.borrow_mut().set_cnt(10, mask, value),
                io32(0x4A4) => self.spu_context.borrow_mut().set_sad(10, mask, value),
                io16(0x4A8) => self.spu_context.borrow_mut().set_tmr(10, mask, value),
                io16(0x4AA) => self.spu_context.borrow_mut().set_pnt(10, mask, value),
                io32(0x4AC) => self.spu_context.borrow_mut().set_len(10, mask, value),
                io32(0x4B0) => self.spu_context.borrow_mut().set_cnt(11, mask, value),
                io32(0x4B4) => self.spu_context.borrow_mut().set_sad(11, mask, value),
                io16(0x4B8) => self.spu_context.borrow_mut().set_tmr(11, mask, value),
                io16(0x4BA) => self.spu_context.borrow_mut().set_pnt(11, mask, value),
                io32(0x4BC) => self.spu_context.borrow_mut().set_len(11, mask, value),
                io32(0x4C0) => self.spu_context.borrow_mut().set_cnt(12, mask, value),
                io32(0x4C4) => self.spu_context.borrow_mut().set_sad(12, mask, value),
                io16(0x4C8) => self.spu_context.borrow_mut().set_tmr(12, mask, value),
                io16(0x4CA) => self.spu_context.borrow_mut().set_pnt(12, mask, value),
                io32(0x4CC) => self.spu_context.borrow_mut().set_len(12, mask, value),
                io32(0x4D0) => self.spu_context.borrow_mut().set_cnt(13, mask, value),
                io32(0x4D4) => self.spu_context.borrow_mut().set_sad(13, mask, value),
                io16(0x4D8) => self.spu_context.borrow_mut().set_tmr(13, mask, value),
                io16(0x4DA) => self.spu_context.borrow_mut().set_pnt(13, mask, value),
                io32(0x4DC) => self.spu_context.borrow_mut().set_len(13, mask, value),
                io32(0x4E0) => self.spu_context.borrow_mut().set_cnt(14, mask, value),
                io32(0x4E4) => self.spu_context.borrow_mut().set_sad(14, mask, value),
                io16(0x4E8) => self.spu_context.borrow_mut().set_tmr(14, mask, value),
                io16(0x4EA) => self.spu_context.borrow_mut().set_pnt(14, mask, value),
                io32(0x4EC) => self.spu_context.borrow_mut().set_len(14, mask, value),
                io32(0x4F0) => self.spu_context.borrow_mut().set_cnt(15, mask, value),
                io32(0x4F4) => self.spu_context.borrow_mut().set_sad(15, mask, value),
                io16(0x4F8) => self.spu_context.borrow_mut().set_tmr(15, mask, value),
                io16(0x4FA) => self.spu_context.borrow_mut().set_pnt(15, mask, value),
                io32(0x4FC) => self.spu_context.borrow_mut().set_len(15, mask, value),
                io16(0x500) => self.spu_context.borrow_mut().set_main_sound_cnt(mask, value),
                io16(0x504) => self.spu_context.borrow_mut().set_sound_bias(mask, value),
                io8(0x508) => todo!(),
                io8(0x509) => todo!(),
                io32(0x510) => todo!(),
                io16(0x514) => todo!(),
                io32(0x518) => todo!(),
                io16(0x51C) => todo!(),
                io16(0x800006) => todo!(),
                io16(0x800010) => todo!(),
                io16(0x800012) => todo!(),
                io16(0x800018) => todo!(),
                io16(0x80001A) => todo!(),
                io16(0x80001C) => todo!(),
                io16(0x800020) => todo!(),
                io16(0x800022) => todo!(),
                io16(0x800024) => todo!(),
                io16(0x80002A) => todo!(),
                io16(0x800030) => todo!(),
                io16(0x80003C) => todo!(),
                io16(0x800040) => todo!(),
                io16(0x800050) => todo!(),
                io16(0x800052) => todo!(),
                io16(0x800056) => todo!(),
                io16(0x800058) => todo!(),
                io16(0x80005A) => todo!(),
                io16(0x80005C) => todo!(),
                io16(0x800062) => todo!(),
                io16(0x800064) => todo!(),
                io16(0x800068) => todo!(),
                io16(0x80006C) => todo!(),
                io16(0x800070) => todo!(),
                io16(0x800074) => todo!(),
                io16(0x800076) => todo!(),
                io16(0x800080) => todo!(),
                io16(0x80008C) => todo!(),
                io16(0x800090) => todo!(),
                io16(0x8000A0) => todo!(),
                io16(0x8000A4) => todo!(),
                io16(0x8000A8) => todo!(),
                io16(0x8000AC) => todo!(),
                io16(0x8000AE) => todo!(),
                io16(0x8000E8) => todo!(),
                io16(0x8000EA) => todo!(),
                io16(0x800110) => todo!(),
                io16(0x80011C) => todo!(),
                io16(0x800120) => todo!(),
                io16(0x800122) => todo!(),
                io16(0x800124) => todo!(),
                io16(0x800128) => todo!(),
                io16(0x800130) => todo!(),
                io16(0x800132) => todo!(),
                io16(0x800134) => todo!(),
                io16(0x800140) => todo!(),
                io16(0x800142) => todo!(),
                io16(0x800144) => todo!(),
                io16(0x800146) => todo!(),
                io16(0x800148) => todo!(),
                io16(0x80014A) => todo!(),
                io16(0x80014C) => todo!(),
                io16(0x800150) => todo!(),
                io16(0x800154) => todo!(),
                io16(0x800158) => todo!(),
                io16(0x80015A) => todo!(),
                io16(0x80021C) => todo!(),
                _ => {
                    if DEBUG && index == 3 {
                        debug_println!(
                            "{:?} unknown io port write at {:x} with value {:x}",
                            CpuType::ARM7,
                            addr_offset,
                            value.into()
                        );
                    }
                }
            });
            index += 1;
        }
    }
}
