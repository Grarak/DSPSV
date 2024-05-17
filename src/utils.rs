use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{BuildHasher, Hasher};
use std::ops::{Deref, DerefMut};
use std::{cmp, mem};

pub const fn align_up(n: u32, align: u32) -> u32 {
    (n + align - 1) & !(align - 1)
}

pub trait Convert: Copy + Into<u32> {
    fn from(value: u32) -> Self;
}

impl Convert for u8 {
    fn from(value: u32) -> Self {
        value as u8
    }
}

impl Convert for u16 {
    fn from(value: u32) -> Self {
        value as u16
    }
}

impl Convert for u32 {
    fn from(value: u32) -> Self {
        value
    }
}

pub fn read_from_mem<T: Clone>(mem: &[u8], addr: u32) -> T {
    let aligned: &[T] = unsafe { mem::transmute(&mem[addr as usize..]) };
    aligned[0].clone()
}

pub fn read_from_mem_slice<T: Copy>(mem: &[u8], addr: u32, slice: &mut [T]) -> usize {
    let aligned: &[T] = unsafe { mem::transmute(&mem[addr as usize..]) };
    let read_amount = cmp::min(aligned.len(), slice.len());
    slice[..read_amount].copy_from_slice(&aligned[..read_amount]);
    read_amount
}

pub fn write_to_mem<T>(mem: &mut [u8], addr: u32, value: T) {
    let aligned: &mut [T] = unsafe { mem::transmute(&mut mem[addr as usize..]) };
    aligned[0] = value;
}

pub fn write_to_mem_slice<T: Copy>(mem: &mut [u8], addr: u32, slice: &[T]) -> usize {
    let aligned: &mut [T] = unsafe { mem::transmute(&mut mem[addr as usize..]) };
    let write_amount = cmp::min(aligned.len(), slice.len());
    aligned[..write_amount].copy_from_slice(&slice[..write_amount]);
    write_amount
}

pub struct StrErr {
    str: String,
}

impl StrErr {
    pub fn new(str: impl Into<String>) -> Self {
        StrErr { str: str.into() }
    }
}

impl Debug for StrErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.str, f)
    }
}

impl Display for StrErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.str, f)
    }
}

impl Error for StrErr {}

pub type HeapMemU8<const SIZE: usize> = HeapMem<u8, SIZE>;
pub type HeapMemI8<const SIZE: usize> = HeapMem<i8, SIZE>;
pub type HeapMemU32<const SIZE: usize> = HeapMem<u32, SIZE>;

pub struct HeapMem<T: Sized, const SIZE: usize>(Box<[T; SIZE]>);

impl<T: Sized + Default, const SIZE: usize> HeapMem<T, SIZE> {
    pub fn new() -> Self {
        HeapMem::default()
    }
}

impl<T: Sized, const SIZE: usize> Deref for HeapMem<T, SIZE> {
    type Target = [T; SIZE];

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<T: Sized, const SIZE: usize> DerefMut for HeapMem<T, SIZE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.deref_mut()
    }
}

impl<T: Sized + Default, const SIZE: usize> Default for HeapMem<T, SIZE> {
    fn default() -> Self {
        let mut mem: Box<[T; SIZE]> = unsafe { Box::new_zeroed().assume_init() };
        mem.fill_with(|| T::default());
        HeapMem(mem)
    }
}

impl<T: Sized + Copy, const SIZE: usize> Clone for HeapMem<T, SIZE> {
    fn clone(&self) -> Self {
        let mut mem: Box<[T; SIZE]> = unsafe { Box::new_zeroed().assume_init() };
        mem.copy_from_slice(self.deref());
        HeapMem(mem)
    }
}

pub const fn crc16(mut crc: u32, buf: &[u8], start: usize, size: usize) -> u16 {
    const TABLE: [u16; 8] = [0xC0C1, 0xC181, 0xC301, 0xC601, 0xCC01, 0xD801, 0xF001, 0xA001];

    let mut i = start;
    while i < start + size {
        crc ^= buf[i] as u32;
        let mut j = 0;
        while j < TABLE.len() {
            crc = (crc >> 1) ^ if crc & 1 != 0 { (TABLE[j] as u32) << (7 - j) } else { 0 };
            j += 1;
        }
        i += 1;
    }
    crc as u16
}

pub struct NoHasher {
    state: u32,
}

impl Hasher for NoHasher {
    fn finish(&self) -> u64 {
        self.state as u64
    }

    fn write(&mut self, _: &[u8]) {
        unreachable!()
    }

    fn write_u32(&mut self, i: u32) {
        self.state = i;
    }

    fn write_i32(&mut self, i: i32) {
        self.state = i as u32;
    }
}

#[derive(Clone, Default)]
pub struct BuildNoHasher;

impl BuildHasher for BuildNoHasher {
    type Hasher = NoHasher;
    fn build_hasher(&self) -> NoHasher {
        NoHasher { state: 0 }
    }
}

pub type NoHashMap<V> = HashMap<u32, V, BuildNoHasher>;

pub enum ThreadPriority {
    Low,
    Default,
    High,
}

pub enum ThreadAffinity {
    Core0,
    Core1,
    Core2,
}

#[cfg(target_os = "linux")]
pub fn set_thread_prio_affinity(_: ThreadPriority, _: ThreadAffinity) {}

#[cfg(target_os = "vita")]
pub fn set_thread_prio_affinity(thread_priority: ThreadPriority, thread_affinity: ThreadAffinity) {
    unsafe {
        let id = vitasdk_sys::sceKernelGetThreadId();
        vitasdk_sys::sceKernelChangeThreadPriority(
            id,
            match thread_priority {
                ThreadPriority::Low => vitasdk_sys::SCE_KERNEL_PROCESS_PRIORITY_USER_LOW,
                ThreadPriority::Default => vitasdk_sys::SCE_KERNEL_PROCESS_PRIORITY_USER_DEFAULT,
                ThreadPriority::High => vitasdk_sys::SCE_KERNEL_PROCESS_PRIORITY_USER_HIGH,
            } as _,
        );
        vitasdk_sys::sceKernelChangeThreadCpuAffinityMask(
            id,
            match thread_affinity {
                ThreadAffinity::Core0 => vitasdk_sys::SCE_KERNEL_CPU_MASK_USER_0,
                ThreadAffinity::Core1 => vitasdk_sys::SCE_KERNEL_CPU_MASK_USER_1,
                ThreadAffinity::Core2 => vitasdk_sys::SCE_KERNEL_CPU_MASK_USER_2,
            } as _,
        );
    }
}
