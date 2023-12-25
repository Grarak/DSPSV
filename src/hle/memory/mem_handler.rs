use crate::hle::cp15_context::{Cp15Context, TcmState};
use crate::hle::memory::io_ports::IoPorts;
use crate::hle::memory::main_memory::MainMemory;
use crate::hle::memory::regions;
use crate::hle::memory::tcm::Tcm;
use crate::hle::memory::wram_context::WramContext;
use crate::hle::CpuType;
use crate::jit::jit_asm::JitState;
use crate::logging::debug_println;
use crate::utils::{Convert, FastCell};
use std::mem;
use std::rc::Rc;
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};

pub struct MemHandler {
    pub cpu_type: CpuType,
    pub memory: Arc<RwLock<MainMemory>>,
    pub wram_context: Arc<WramContext>,
    cp15_context: Rc<FastCell<Cp15Context>>,
    tcm: Rc<FastCell<Tcm>>,
    pub io_ports: IoPorts,
    pub jit_state: Arc<Mutex<JitState>>,
    pub dma_transfer_lock: Arc<RwLock<()>>,
}

unsafe impl Send for MemHandler {}
unsafe impl Sync for MemHandler {}

impl MemHandler {
    pub fn new(
        cpu_type: CpuType,
        memory: Arc<RwLock<MainMemory>>,
        wram_context: Arc<WramContext>,
        cp15_context: Rc<FastCell<Cp15Context>>,
        io_ports: IoPorts,
    ) -> Self {
        MemHandler {
            cpu_type,
            memory,
            wram_context,
            cp15_context,
            tcm: Rc::new(FastCell::new(Tcm::new())),
            io_ports,
            jit_state: Arc::new(Mutex::new(JitState::new())),
            dma_transfer_lock: Arc::new(RwLock::new(())),
        }
    }

    fn lock_dma<const LOCK: bool>(&self) -> Option<RwLockReadGuard<()>> {
        if LOCK {
            Some(self.dma_transfer_lock.read().unwrap())
        } else {
            None
        }
    }

    pub fn read<T: Convert>(&self, addr: u32) -> T {
        self.read_lock::<true, T>(addr)
    }

    pub fn read_lock<const LOCK_DMA: bool, T: Convert>(&self, addr: u32) -> T {
        let mut buf = [T::from(0)];
        self.read_slice_lock::<LOCK_DMA, T>(addr, &mut buf);

        debug_println!(
            "{:?} memory read at {:x} with value {:x}",
            self.cpu_type,
            addr,
            buf[0].into()
        );

        buf[0]
    }

    pub fn read_slice<T: Convert>(&self, addr: u32, slice: &mut [T]) {
        self.read_slice_lock::<true, T>(addr, slice);
    }

    pub fn read_slice_lock<const LOCK_DMA: bool, T: Convert>(&self, addr: u32, slice: &mut [T]) {
        let addr_end = addr + (slice.len() * mem::size_of::<T>()) as u32;

        let addr_base = addr & 0xFF000000;
        let addr_end_base = addr_end & 0xFF000000;
        debug_assert_eq!(addr_base, addr_end_base);

        let addr_offset = addr - addr_base;

        match addr_base {
            regions::MAIN_MEMORY_OFFSET => {
                let _lock = self.lock_dma::<LOCK_DMA>();
                self.memory
                    .read()
                    .unwrap()
                    .read_main_slice(addr_offset, slice)
            }
            regions::SHARED_WRAM_OFFSET => {
                let _lock = self.lock_dma::<LOCK_DMA>();
                self.wram_context
                    .read_slice(self.cpu_type, addr_offset, slice)
            }
            regions::IO_PORTS_OFFSET => {
                let _lock = self.lock_dma::<LOCK_DMA>();
                for (i, value) in slice.iter_mut().enumerate() {
                    *value = self
                        .io_ports
                        .read(addr_offset + (i * mem::size_of::<T>()) as u32);
                }
            }
            _ => todo!(),
        };
    }

    pub fn write<T: Convert>(&self, addr: u32, value: T) {
        self.write_lock::<true, T>(addr, value);
    }

    pub fn write_lock<const LOCK_DMA: bool, T: Convert>(&self, addr: u32, value: T) {
        debug_println!(
            "{:?} memory write at {:x} with value {:x}",
            self.cpu_type,
            addr,
            value.into(),
        );

        self.write_slice_lock::<LOCK_DMA, T>(addr, &[value]);
    }

    pub fn write_slice<T: Convert>(&self, addr: u32, slice: &[T]) {
        self.write_slice_lock::<true, T>(addr, slice);
    }

    pub fn write_slice_lock<const LOCK_DMA: bool, T: Convert>(&self, addr: u32, slice: &[T]) {
        let addr_end = addr + (slice.len() * mem::size_of::<T>()) as u32;

        let addr_base = addr & 0xFF000000;
        let addr_end_base = addr_end & 0xFF000000;
        debug_assert_eq!(addr_base, addr_end_base);

        let addr_offset = addr - addr_base;
        let mut invalidate_jit = false;
        match addr_base {
            regions::MAIN_MEMORY_OFFSET => {
                let _lock = self.lock_dma::<LOCK_DMA>();
                self.memory
                    .write()
                    .unwrap()
                    .write_main_slice(addr_offset, slice);
            }
            regions::SHARED_WRAM_OFFSET => {
                let _lock = self.lock_dma::<LOCK_DMA>();
                self.wram_context
                    .write_slice(self.cpu_type, addr_offset, slice);
                invalidate_jit = true;
            }
            regions::IO_PORTS_OFFSET => {
                let _lock = self.lock_dma::<LOCK_DMA>();
                for (i, value) in slice.iter().enumerate() {
                    self.io_ports
                        .write(addr_offset + (i * mem::size_of::<T>()) as u32, *value);
                }
            }
            _ => {
                let mut handled = false;

                if self.cpu_type == CpuType::ARM9 {
                    let cp15_context = self.cp15_context.borrow();
                    if addr < cp15_context.itcm_size {
                        if cp15_context.itcm_state != TcmState::Disabled {
                            self.tcm.borrow_mut().write_itcm_slice(addr, slice);
                            invalidate_jit = true;
                        }
                        handled = true;
                    } else if addr >= cp15_context.dtcm_addr
                        && addr < cp15_context.dtcm_addr + cp15_context.dtcm_size
                    {
                        if cp15_context.dtcm_state != TcmState::Disabled {
                            self.tcm
                                .borrow_mut()
                                .write_dtcm_slice(addr - cp15_context.dtcm_addr, slice);
                        }
                        handled = true;
                    }
                }

                if !handled {
                    todo!("{:x}", addr)
                }
            }
        }

        if invalidate_jit {
            let mut jit_state = self.jit_state.lock().unwrap();
            for addr in (addr..addr + (slice.len() * mem::size_of::<T>()) as u32)
                .step_by(mem::size_of::<T>())
            {
                jit_state.invalidated_addrs.insert(addr);

                let (current_jit_block_start, current_jit_block_end) =
                    jit_state.current_block_range;
                if addr >= current_jit_block_start && addr <= current_jit_block_end {
                    todo!()
                }
            }
        }
    }
}