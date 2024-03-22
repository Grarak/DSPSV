use crate::hle::memory::regions;
use crate::hle::CpuType;
use crate::utils;
use crate::utils::{Convert, HeapMemU8};
use std::hint::unreachable_unchecked;
use std::ops::{Deref, DerefMut};
use std::{ptr, slice};

struct SharedWramMap {
    shared_ptr: *const u8,
    size: usize,
}

impl SharedWramMap {
    fn new(shared_ptr: *const u8, size: usize) -> Self {
        SharedWramMap { shared_ptr, size }
    }

    fn as_ptr(&self) -> *const u8 {
        self.shared_ptr
    }

    fn len(&self) -> usize {
        self.size
    }

    fn as_mut(&mut self) -> SharedWramMapMut {
        SharedWramMapMut::new(self.shared_ptr as *mut _, self.size)
    }
}

impl Default for SharedWramMap {
    fn default() -> Self {
        SharedWramMap::new(ptr::null(), 0)
    }
}

impl Deref for SharedWramMap {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}

impl AsRef<[u8]> for SharedWramMap {
    fn as_ref(&self) -> &[u8] {
        self.deref()
    }
}

struct SharedWramMapMut {
    shared_ptr: *mut u8,
    size: usize,
}

impl SharedWramMapMut {
    fn new(shared_ptr: *mut u8, size: usize) -> Self {
        SharedWramMapMut { shared_ptr, size }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.shared_ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.shared_ptr
    }

    pub fn len(&self) -> usize {
        self.size
    }
}

impl Deref for SharedWramMapMut {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}

impl AsRef<[u8]> for SharedWramMapMut {
    fn as_ref(&self) -> &[u8] {
        self.deref()
    }
}

impl DerefMut for SharedWramMapMut {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }
}

impl AsMut<[u8]> for SharedWramMapMut {
    fn as_mut(&mut self) -> &mut [u8] {
        self.deref_mut()
    }
}

pub struct Wram {
    wram_arm7: HeapMemU8<{ regions::ARM7_WRAM_SIZE as usize }>,
    pub cnt: u8,
    shared_mem: HeapMemU8<{ regions::SHARED_WRAM_SIZE as usize }>,
    arm9_map: SharedWramMap,
    arm7_map: SharedWramMap,
}

impl Wram {
    pub fn new() -> Self {
        let mut instance = Wram {
            wram_arm7: HeapMemU8::new(),
            cnt: 0,
            shared_mem: HeapMemU8::new(),
            arm9_map: SharedWramMap::default(),
            arm7_map: SharedWramMap::default(),
        };
        instance.init_maps();
        instance
    }

    fn init_maps(&mut self) {
        const SHARED_LEN: usize = regions::SHARED_WRAM_SIZE as usize;

        match self.cnt {
            0 => {
                self.arm9_map = SharedWramMap::new(self.shared_mem.as_ptr(), SHARED_LEN);
                self.arm7_map = SharedWramMap::new(self.wram_arm7.as_ptr(), self.wram_arm7.len());
            }
            1 => {
                self.arm9_map = SharedWramMap::new(
                    self.shared_mem[SHARED_LEN / 2..].as_mut_ptr(),
                    SHARED_LEN / 2,
                );
                self.arm7_map = SharedWramMap::new(self.shared_mem.as_mut_ptr(), SHARED_LEN / 2);
            }
            2 => {
                self.arm9_map = SharedWramMap::new(self.shared_mem.as_mut_ptr(), SHARED_LEN / 2);
                self.arm7_map = SharedWramMap::new(
                    self.shared_mem[SHARED_LEN / 2..].as_mut_ptr(),
                    SHARED_LEN / 2,
                );
            }
            3 => {
                self.arm9_map = SharedWramMap::default();
                self.arm7_map = SharedWramMap::new(self.shared_mem.as_ptr(), SHARED_LEN);
            }
            _ => {
                unsafe { unreachable_unchecked() };
            }
        }
    }

    pub fn set_cnt(&mut self, value: u8) {
        self.cnt = value & 0x3;
        self.init_maps();
    }

    fn read_arm9<T: Convert>(&self, addr_offset: u32) -> T {
        let mem = &self.arm9_map;
        utils::read_from_mem(mem, addr_offset & (mem.len() - 1) as u32)
    }

    fn read_arm7<T: Convert>(&self, addr_offset: u32) -> T {
        let mem = &self.arm7_map;
        utils::read_from_mem(mem, addr_offset & (mem.len() - 1) as u32)
    }

    fn write_arm9<T: Convert>(&mut self, addr_offset: u32, value: T) {
        let mut mem = self.arm9_map.as_mut();
        let mem_len = mem.len();
        utils::write_to_mem(&mut mem, addr_offset & (mem_len - 1) as u32, value);
    }

    fn write_arm7<T: Convert>(&mut self, addr_offset: u32, value: T) {
        let mut mem = self.arm7_map.as_mut();
        let mem_len = mem.len();
        utils::write_to_mem(&mut mem, addr_offset & (mem_len - 1) as u32, value);
    }

    pub fn read<const CPU: CpuType, T: Convert>(&self, addr_offset: u32) -> T {
        match CPU {
            CpuType::ARM9 => self.read_arm9(addr_offset),
            CpuType::ARM7 => {
                if addr_offset & regions::ARM7_WRAM_OFFSET != 0 {
                    utils::read_from_mem(
                        self.wram_arm7.as_slice(),
                        addr_offset & (regions::ARM7_WRAM_SIZE - 1),
                    )
                } else {
                    self.read_arm7(addr_offset)
                }
            }
        }
    }

    pub fn write<const CPU: CpuType, T: Convert>(
        &mut self,
        addr_offset: u32,
        value: T,
    ) -> (u32, u32) {
        match CPU {
            CpuType::ARM9 => {
                self.write_arm9(addr_offset, value);
                (self.arm9_map.size as u32, regions::SHARED_WRAM_OFFSET)
            }
            CpuType::ARM7 => {
                if self.cnt == 0 || addr_offset & regions::ARM7_WRAM_OFFSET != 0 {
                    utils::write_to_mem(
                        self.wram_arm7.as_mut_slice(),
                        addr_offset & (regions::ARM7_WRAM_SIZE - 1),
                        value,
                    );
                    (regions::ARM7_WRAM_SIZE, regions::ARM7_WRAM_OFFSET)
                } else {
                    self.write_arm7(addr_offset, value);
                    (self.arm7_map.size as u32, regions::SHARED_WRAM_OFFSET)
                }
            }
        }
    }
}
