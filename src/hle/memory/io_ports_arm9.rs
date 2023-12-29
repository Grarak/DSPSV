use crate::hle::memory::io_ports::IoPorts;
use crate::hle::CpuType;
use crate::logging::debug_println;
use crate::utils::Convert;
use crate::DEBUG;
use dspsv_macros::{io_ports_read, io_ports_write};
use std::mem;

impl IoPorts {
    pub fn read_arm9<T: Convert>(&self, addr_offset: u32) -> T {
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
                io32(0x0) => todo!(),
                io16(0x4) => todo!(),
                io16(0x6) => todo!(),
                io16(0x8) => todo!(),
                io16(0xA) => todo!(),
                io16(0xC) => todo!(),
                io16(0xE) => todo!(),
                io16(0x48) => todo!(),
                io16(0x4A) => todo!(),
                io16(0x50) => todo!(),
                io16(0x52) => todo!(),
                io16(0x60) => todo!(),
                io32(0x64) => todo!(),
                io16(0x6C) => todo!(),
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
                io32(0xDC) => self.dma.borrow().get_cnt(3),
                io32(0xE0) => self.dma.borrow().get_fill(0),
                io32(0xE4) => self.dma.borrow().get_fill(1),
                io32(0xE8) => self.dma.borrow().get_fill(2),
                io32(0xEC) => self.dma.borrow().get_fill(3),
                io16(0x100) => todo!(),
                io16(0x102) => todo!(),
                io16(0x104) => todo!(),
                io16(0x106) => todo!(),
                io16(0x108) => todo!(),
                io16(0x10A) => todo!(),
                io16(0x10C) => todo!(),
                io16(0x10E) => todo!(),
                io16(0x130) => todo!(),
                io16(0x180) => self.ipc_handler.read().unwrap().get_sync_reg(CpuType::ARM9),
                io16(0x184) => todo!(),
                io16(0x1A0) => todo!(),
                io8(0x1A2) => todo!(),
                io32(0x1A4) => todo!(),
                io8(0x208) => self.cpu_regs.borrow().ime,
                io32(0x210) => self.cpu_regs.borrow().ie,
                io32(0x214) => self.cpu_regs.borrow().irf,
                io8(0x240) => self.vram_context.get_cnt(0),
                io8(0x242) => self.vram_context.get_cnt(1),
                io8(0x243) => self.vram_context.get_cnt(2),
                io8(0x241) => self.vram_context.get_cnt(3),
                io8(0x244) => self.vram_context.get_cnt(4),
                io8(0x245) => self.vram_context.get_cnt(5),
                io8(0x246) => self.vram_context.get_cnt(6),
                io8(0x247) => self.wram_context.get_cnt(),
                io8(0x248) => self.vram_context.get_cnt(7),
                io8(0x249) => self.vram_context.get_cnt(8),
                io16(0x280) => todo!(),
                io32(0x290) => todo!(),
                io32(0x294) => todo!(),
                io32(0x298) => todo!(),
                io32(0x29C) => todo!(),
                io32(0x2A0) => todo!(),
                io32(0x2A4) => todo!(),
                io32(0x2A8) => todo!(),
                io32(0x2AC) => todo!(),
                io16(0x2B0) => todo!(),
                io32(0x2B4) => todo!(),
                io32(0x2B8) => todo!(),
                io32(0x2BC) => todo!(),
                io8(0x300) => todo!(),
                io16(0x304) => todo!(),
                io32(0x600) => todo!(),
                io32(0x604) => todo!(),
                io32(0x620) => todo!(),
                io32(0x624) => todo!(),
                io32(0x628) => todo!(),
                io32(0x62C) => todo!(),
                io16(0x630) => todo!(),
                io16(0x632) => todo!(),
                io16(0x634) => todo!(),
                io32(0x640) => todo!(),
                io32(0x644) => todo!(),
                io32(0x648) => todo!(),
                io32(0x64C) => todo!(),
                io32(0x650) => todo!(),
                io32(0x654) => todo!(),
                io32(0x658) => todo!(),
                io32(0x65C) => todo!(),
                io32(0x660) => todo!(),
                io32(0x664) => todo!(),
                io32(0x668) => todo!(),
                io32(0x66C) => todo!(),
                io32(0x670) => todo!(),
                io32(0x674) => todo!(),
                io32(0x678) => todo!(),
                io32(0x67C) => todo!(),
                io32(0x680) => todo!(),
                io32(0x684) => todo!(),
                io32(0x688) => todo!(),
                io32(0x68C) => todo!(),
                io32(0x690) => todo!(),
                io32(0x694) => todo!(),
                io32(0x698) => todo!(),
                io32(0x69C) => todo!(),
                io32(0x6A0) => todo!(),
                io32(0x1000) => todo!(),
                io16(0x1008) => todo!(),
                io16(0x100A) => todo!(),
                io16(0x100C) => todo!(),
                io16(0x100E) => todo!(),
                io16(0x1048) => todo!(),
                io16(0x104A) => todo!(),
                io16(0x1050) => todo!(),
                io16(0x1052) => todo!(),
                io16(0x106C) => todo!(),
                io32(0x100000) => todo!(),
                io32(0x100010) => todo!(),
                _ => {
                    if DEBUG && index == 3 {
                        debug_println!(
                            "{:?} unknown io port read at {:x}",
                            CpuType::ARM9,
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

    pub fn write_arm9<T: Convert>(&self, addr_offset: u32, value: T) {
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
                io32(0x0) => self.gpu_2d_context_0.borrow_mut().set_disp_cnt(mask, value),
                io16(0x4) => self.gpu_context.borrow_mut().set_disp_stat(mask, value),
                io16(0x08) => self.gpu_2d_context_0.borrow_mut().set_bg_cnt(0, mask, value),
                io16(0xA) => self.gpu_2d_context_0.borrow_mut().set_bg_cnt(1, mask, value),
                io16(0xC) => self.gpu_2d_context_0.borrow_mut().set_bg_cnt(2, mask, value),
                io16(0xE) => self.gpu_2d_context_0.borrow_mut().set_bg_cnt(3, mask, value),
                io16(0x10) => self.gpu_2d_context_0.borrow_mut().set_bg_h_ofs(0, mask, value),
                io16(0x12) => self.gpu_2d_context_0.borrow_mut().set_bg_v_ofs(0, mask, value),
                io16(0x14) => self.gpu_2d_context_0.borrow_mut().set_bg_h_ofs(1, mask, value),
                io16(0x16) => self.gpu_2d_context_0.borrow_mut().set_bg_v_ofs(1, mask, value),
                io16(0x18) => self.gpu_2d_context_0.borrow_mut().set_bg_h_ofs(2, mask, value),
                io16(0x1A) => self.gpu_2d_context_0.borrow_mut().set_bg_v_ofs(2, mask, value),
                io16(0x1C) => self.gpu_2d_context_0.borrow_mut().set_bg_h_ofs(3, mask, value),
                io16(0x1E) => self.gpu_2d_context_0.borrow_mut().set_bg_v_ofs(3, mask, value),
                io16(0x20) => self.gpu_2d_context_0.borrow_mut().set_bg_p_a(2, mask, value),
                io16(0x22) => self.gpu_2d_context_0.borrow_mut().set_bg_p_b(2, mask, value),
                io16(0x24) => self.gpu_2d_context_0.borrow_mut().set_bg_p_c(2, mask, value),
                io16(0x26) => self.gpu_2d_context_0.borrow_mut().set_bg_p_d(2, mask, value),
                io32(0x28) => self.gpu_2d_context_0.borrow_mut().set_bg_x(2, mask, value),
                io32(0x2C) => self.gpu_2d_context_0.borrow_mut().set_bg_y(2, mask, value),
                io16(0x30) => self.gpu_2d_context_0.borrow_mut().set_bg_p_a(3, mask, value),
                io16(0x32) => self.gpu_2d_context_0.borrow_mut().set_bg_p_b(3, mask, value),
                io16(0x34) => self.gpu_2d_context_0.borrow_mut().set_bg_p_c(3, mask, value),
                io16(0x36) => self.gpu_2d_context_0.borrow_mut().set_bg_p_d(3, mask, value),
                io32(0x38) => self.gpu_2d_context_0.borrow_mut().set_bg_x(3, mask, value),
                io32(0x3C) => self.gpu_2d_context_0.borrow_mut().set_bg_y(3, mask, value),
                io16(0x40) => self.gpu_2d_context_0.borrow_mut().set_win_h(0, mask, value),
                io16(0x42) => self.gpu_2d_context_0.borrow_mut().set_win_h(1, mask, value),
                io16(0x44) => self.gpu_2d_context_0.borrow_mut().set_win_v(0, mask, value),
                io16(0x46) => self.gpu_2d_context_0.borrow_mut().set_win_v(1, mask, value),
                io16(0x48) => self.gpu_2d_context_0.borrow_mut().set_win_in(mask, value),
                io16(0x4A) => self.gpu_2d_context_0.borrow_mut().set_win_out(mask, value),
                io16(0x4C) => self.gpu_2d_context_0.borrow_mut().set_mosaic(mask, value),
                io16(0x50) => self.gpu_2d_context_0.borrow_mut().set_bld_cnt(mask, value),
                io16(0x52) => self.gpu_2d_context_0.borrow_mut().set_bld_alpha(mask, value),
                io8(0x54) => self.gpu_2d_context_0.borrow_mut().set_bld_y(value),
                io16(0x60) => todo!(),
                io32(0x64) => todo!(),
                io16(0x6C) => todo!(),
                io32(0xB0) => self.dma.borrow_mut().set_sad(0, mask, value),
                io32(0xB4) => self.dma.borrow_mut().set_dad(0, mask, value),
                io32(0xB8) => self.dma.borrow_mut().set_cnt(0, mask, value),
                io32(0xBC) => self.dma.borrow_mut().set_sad(1, mask, value),
                io32(0xC0) => self.dma.borrow_mut().set_dad(1, mask, value),
                io32(0xC4) => self.dma.borrow_mut().set_cnt(1, mask, value),
                io32(0xC8) => self.dma.borrow_mut().set_sad(2, mask, value),
                io32(0xCC) => self.dma.borrow_mut().set_dad(2, mask, value),
                io32(0xD0) => self.dma.borrow_mut().set_cnt(2, mask, value),
                io32(0xD4) => self.dma.borrow_mut().set_sad(3, mask, value),
                io32(0xD8) => self.dma.borrow_mut().set_dad(3, mask, value),
                io32(0xDC) => self.dma.borrow_mut().set_cnt(3, mask, value),
                io32(0xE0) => self.dma.borrow_mut().set_fill(0, mask, value),
                io32(0xE4) => self.dma.borrow_mut().set_fill(1, mask, value),
                io32(0xE8) => self.dma.borrow_mut().set_fill(2, mask, value),
                io32(0xEC) => self.dma.borrow_mut().set_fill(3, mask, value),
                io16(0x100) => self.timers_context.borrow_mut().set_cnt_l(0, mask, value),
                io16(0x102) => self.timers_context.borrow_mut().set_cnt_h(0, mask, value),
                io16(0x104) => self.timers_context.borrow_mut().set_cnt_l(1, mask, value),
                io16(0x106) => self.timers_context.borrow_mut().set_cnt_h(1, mask, value),
                io16(0x108) => self.timers_context.borrow_mut().set_cnt_l(2, mask, value),
                io16(0x10A) => self.timers_context.borrow_mut().set_cnt_h(2, mask, value),
                io16(0x10C) => self.timers_context.borrow_mut().set_cnt_l(3, mask, value),
                io16(0x10E) => self.timers_context.borrow_mut().set_cnt_h(3, mask, value),
                io16(0x180) => self.ipc_handler.write().unwrap().set_sync_reg(CpuType::ARM9, mask, value),
                io16(0x184) => self.ipc_handler.write().unwrap().set_fifo_cnt(CpuType::ARM9, mask, value),
                io32(0x188) => todo!(),
                io16(0x1A0) => todo!(),
                io8(0x1A2) => todo!(),
                io32(0x1A4) => todo!(),
                io32(0x1A8) => todo!(),
                io32(0x1AC) => todo!(),
                io8(0x208) => self.cpu_regs.borrow_mut().set_ime(value),
                io32(0x210) => self.cpu_regs.borrow_mut().set_ie(mask, value),
                io32(0x214) => self.cpu_regs.borrow_mut().set_irf(mask, value),
                io8(0x240) => self.vram_context.set_cnt(0, value),
                io8(0x241) => self.vram_context.set_cnt(1, value),
                io8(0x242) => self.vram_context.set_cnt(2, value),
                io8(0x243) => self.vram_context.set_cnt(3, value),
                io8(0x244) => self.vram_context.set_cnt(4, value),
                io8(0x245) => self.vram_context.set_cnt(5, value),
                io8(0x246) => self.vram_context.set_cnt(6, value),
                io8(0x247) => self.wram_context.set_cnt(value),
                io8(0x248) => self.vram_context.set_cnt(7, value),
                io8(0x249) => self.vram_context.set_cnt(8, value),
                io16(0x280) => todo!(),
                io32(0x290) => todo!(),
                io32(0x294) => todo!(),
                io32(0x298) => todo!(),
                io32(0x29C) => todo!(),
                io16(0x2B0) => todo!(),
                io32(0x2B8) => todo!(),
                io32(0x2BC) => todo!(),
                io8(0x300) => self.cpu_regs.borrow_mut().set_post_flg(value),
                io16(0x304) => self.gpu_context.borrow_mut().set_pow_cnt1(mask, value),
                io16(0x330) => todo!(),
                io16(0x332) => todo!(),
                io16(0x334) => todo!(),
                io16(0x336) => todo!(),
                io16(0x338) => todo!(),
                io16(0x33A) => todo!(),
                io16(0x33C) => todo!(),
                io16(0x33E) => todo!(),
                io32(0x350) => todo!(),
                io16(0x354) => todo!(),
                io32(0x358) => todo!(),
                io16(0x35C) => todo!(),
                io8(0x360) => todo!(),
                io8(0x361) => todo!(),
                io8(0x362) => todo!(),
                io8(0x363) => todo!(),
                io8(0x364) => todo!(),
                io8(0x365) => todo!(),
                io8(0x366) => todo!(),
                io8(0x367) => todo!(),
                io8(0x368) => todo!(),
                io8(0x369) => todo!(),
                io8(0x36A) => todo!(),
                io8(0x36B) => todo!(),
                io8(0x36C) => todo!(),
                io8(0x36D) => todo!(),
                io8(0x36E) => todo!(),
                io8(0x36F) => todo!(),
                io8(0x370) => todo!(),
                io8(0x371) => todo!(),
                io8(0x372) => todo!(),
                io8(0x373) => todo!(),
                io8(0x374) => todo!(),
                io8(0x375) => todo!(),
                io8(0x376) => todo!(),
                io8(0x377) => todo!(),
                io8(0x378) => todo!(),
                io8(0x379) => todo!(),
                io8(0x37A) => todo!(),
                io8(0x37B) => todo!(),
                io8(0x37C) => todo!(),
                io8(0x37D) => todo!(),
                io8(0x37E) => todo!(),
                io8(0x37F) => todo!(),
                io16(0x380) => todo!(),
                io16(0x382) => todo!(),
                io16(0x384) => todo!(),
                io16(0x386) => todo!(),
                io16(0x388) => todo!(),
                io16(0x38A) => todo!(),
                io16(0x38C) => todo!(),
                io16(0x38E) => todo!(),
                io16(0x390) => todo!(),
                io16(0x392) => todo!(),
                io16(0x394) => todo!(),
                io16(0x396) => todo!(),
                io16(0x398) => todo!(),
                io16(0x39A) => todo!(),
                io16(0x39C) => todo!(),
                io16(0x39E) => todo!(),
                io16(0x3A0) => todo!(),
                io16(0x3A2) => todo!(),
                io16(0x3A4) => todo!(),
                io16(0x3A6) => todo!(),
                io16(0x3A8) => todo!(),
                io16(0x3AA) => todo!(),
                io16(0x3AC) => todo!(),
                io16(0x3AE) => todo!(),
                io16(0x3B0) => todo!(),
                io16(0x3B2) => todo!(),
                io16(0x3B4) => todo!(),
                io16(0x3B6) => todo!(),
                io16(0x3B8) => todo!(),
                io16(0x3BA) => todo!(),
                io16(0x3BC) => todo!(),
                io16(0x3BE) => todo!(),
                io32(0x400) => todo!(),
                io32(0x404) => todo!(),
                io32(0x408) => todo!(),
                io32(0x40C) => todo!(),
                io32(0x410) => todo!(),
                io32(0x414) => todo!(),
                io32(0x418) => todo!(),
                io32(0x41C) => todo!(),
                io32(0x420) => todo!(),
                io32(0x424) => todo!(),
                io32(0x428) => todo!(),
                io32(0x42C) => todo!(),
                io32(0x430) => todo!(),
                io32(0x434) => todo!(),
                io32(0x438) => todo!(),
                io32(0x43C) => todo!(),
                io32(0x440) => todo!(),
                io32(0x444) => todo!(),
                io32(0x448) => todo!(),
                io32(0x44C) => todo!(),
                io32(0x450) => todo!(),
                io32(0x454) => todo!(),
                io32(0x458) => todo!(),
                io32(0x45C) => todo!(),
                io32(0x460) => todo!(),
                io32(0x464) => todo!(),
                io32(0x468) => todo!(),
                io32(0x46C) => todo!(),
                io32(0x470) => todo!(),
                io32(0x480) => todo!(),
                io32(0x484) => todo!(),
                io32(0x488) => todo!(),
                io32(0x48C) => todo!(),
                io32(0x490) => todo!(),
                io32(0x494) => todo!(),
                io32(0x498) => todo!(),
                io32(0x49C) => todo!(),
                io32(0x4A0) => todo!(),
                io32(0x4A4) => todo!(),
                io32(0x4A8) => todo!(),
                io32(0x4AC) => todo!(),
                io32(0x4C0) => todo!(),
                io32(0x4C4) => todo!(),
                io32(0x4C8) => todo!(),
                io32(0x4CC) => todo!(),
                io32(0x4D0) => todo!(),
                io32(0x500) => todo!(),
                io32(0x504) => todo!(),
                io32(0x540) => todo!(),
                io32(0x580) => todo!(),
                io32(0x5C0) => todo!(),
                io32(0x5C4) => todo!(),
                io32(0x5C8) => todo!(),
                io32(0x600) => todo!(),
                io32(0x1000) => self.gpu_2d_context_1.borrow_mut().set_disp_cnt(mask, value),
                io16(0x1008) => self.gpu_2d_context_1.borrow_mut().set_bg_cnt(0, mask, value),
                io16(0x100A) => self.gpu_2d_context_1.borrow_mut().set_bg_cnt(1, mask, value),
                io16(0x100C) => self.gpu_2d_context_1.borrow_mut().set_bg_cnt(2, mask, value),
                io16(0x100E) => self.gpu_2d_context_1.borrow_mut().set_bg_cnt(3, mask, value),
                io16(0x1010) => self.gpu_2d_context_1.borrow_mut().set_bg_h_ofs(0, mask, value),
                io16(0x1012) => self.gpu_2d_context_1.borrow_mut().set_bg_v_ofs(0, mask, value),
                io16(0x1014) => self.gpu_2d_context_1.borrow_mut().set_bg_h_ofs(1, mask, value),
                io16(0x1016) => self.gpu_2d_context_1.borrow_mut().set_bg_v_ofs(1, mask, value),
                io16(0x1018) => self.gpu_2d_context_1.borrow_mut().set_bg_h_ofs(2, mask, value),
                io16(0x101A) => self.gpu_2d_context_1.borrow_mut().set_bg_v_ofs(2, mask, value),
                io16(0x101C) => self.gpu_2d_context_1.borrow_mut().set_bg_h_ofs(3, mask, value),
                io16(0x101E) => self.gpu_2d_context_1.borrow_mut().set_bg_v_ofs(3, mask, value),
                io16(0x1020) => self.gpu_2d_context_1.borrow_mut().set_bg_p_a(2, mask, value),
                io16(0x1022) => self.gpu_2d_context_1.borrow_mut().set_bg_p_b(2, mask, value),
                io16(0x1024) => self.gpu_2d_context_1.borrow_mut().set_bg_p_c(2, mask, value),
                io16(0x1026) => self.gpu_2d_context_1.borrow_mut().set_bg_p_d(2, mask, value),
                io32(0x1028) => self.gpu_2d_context_1.borrow_mut().set_bg_x(2, mask, value),
                io32(0x102C) => self.gpu_2d_context_1.borrow_mut().set_bg_y(2, mask, value),
                io16(0x1030) => self.gpu_2d_context_1.borrow_mut().set_bg_p_a(3, mask, value),
                io16(0x1032) => self.gpu_2d_context_1.borrow_mut().set_bg_p_b(3, mask, value),
                io16(0x1034) => self.gpu_2d_context_1.borrow_mut().set_bg_p_c(3, mask, value),
                io16(0x1036) => self.gpu_2d_context_1.borrow_mut().set_bg_p_d(3, mask, value),
                io32(0x1038) => self.gpu_2d_context_1.borrow_mut().set_bg_x(3, mask, value),
                io32(0x103C) => self.gpu_2d_context_1.borrow_mut().set_bg_y(3, mask, value),
                io16(0x1040) => self.gpu_2d_context_1.borrow_mut().set_win_h(0, mask, value),
                io16(0x1042) => self.gpu_2d_context_1.borrow_mut().set_win_h(1, mask, value),
                io16(0x1044) => self.gpu_2d_context_1.borrow_mut().set_win_v(0, mask, value),
                io16(0x1046) => self.gpu_2d_context_1.borrow_mut().set_win_v(1, mask, value),
                io16(0x1048) => self.gpu_2d_context_1.borrow_mut().set_win_in(mask, value),
                io16(0x104A) => self.gpu_2d_context_1.borrow_mut().set_win_out(mask, value),
                io16(0x104C) => self.gpu_2d_context_1.borrow_mut().set_mosaic(mask, value),
                io16(0x1050) => self.gpu_2d_context_1.borrow_mut().set_bld_cnt(mask, value),
                io16(0x1052) => self.gpu_2d_context_1.borrow_mut().set_bld_alpha(mask, value),
                io8(0x1054) => self.gpu_2d_context_1.borrow_mut().set_bld_y(value),
                io16(0x106C) => self.gpu_2d_context_1.borrow_mut().set_master_bright(mask, value),
                _ => {
                    if DEBUG && index == 3 {
                        debug_println!(
                            "{:?} unknown io port write at {:x} with value {:x}",
                            CpuType::ARM9,
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
