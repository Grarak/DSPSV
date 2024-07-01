use crate::emu::cpu_regs::InterruptFlag;
use crate::emu::emu::{get_cm_mut, get_cpu_regs_mut, get_mem_mut, io_dma, Emu};
use crate::emu::gpu::gpu::{DISPLAY_HEIGHT, DISPLAY_WIDTH};
use crate::emu::memory::dma::DmaTransferMode;
use crate::emu::CpuType::ARM9;
use bilge::prelude::*;
use std::collections::VecDeque;
use std::hint::unreachable_unchecked;
use std::{mem, ops};

#[bitsize(32)]
#[derive(Copy, Clone, FromBits)]
struct GxStat {
    box_pos_vec_test_busy: bool,
    box_test_result: u1,
    not_used: u6,
    pos_vec_mtx_stack_lvl: u5,
    proj_mtx_stack_lvl: u1,
    mtx_stack_busy: bool,
    mtx_stack_overflow_underflow_err: bool,
    num_entries_cmd_fifo: u9,
    cmd_fifo_less_half_full: bool,
    cmd_fifo_empty: bool,
    geometry_busy: bool,
    not_used2: u2,
    cmd_fifo_irq: u2,
}

#[bitsize(32)]
#[derive(Copy, Clone, FromBits)]
struct Viewport {
    x1: u8,
    y1: u8,
    x2: u8,
    y2: u8,
}

impl Default for Viewport {
    fn default() -> Self {
        let mut viewport = Viewport::from(0);
        viewport.set_x2(DISPLAY_WIDTH as u8);
        viewport.set_y2(DISPLAY_HEIGHT as u8);
        viewport
    }
}

const FIFO_PARAM_COUNTS: [u8; 128] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0x00-0x0F
    1, 0, 1, 1, 1, 0, 16, 12, 16, 12, 9, 3, 3, 0, 0, 0, // 0x10-0x1F
    1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, // 0x20-0x2F
    1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0x30-0x3F
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0x40-0x4F
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0x50-0x5F
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0x60-0x6F
    3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0x70-0x7F
];

#[derive(Copy, Clone)]
struct Entry {
    cmd: u8,
    param: u32,
}

impl Entry {
    fn new(cmd: u8, param: u32) -> Self {
        Entry { cmd, param }
    }
}

#[derive(Copy, Clone)]
struct Matrix([i32; 16]);

impl Matrix {
    #[inline(never)]
    fn some_mul_test(&self, rhs: Self) -> Self {
        let mut ret = Matrix::default();

        for y in 0..4 {
            for x in 0..4 {
                ret.0[y * 4 + x] = ((self.0[y * 4] as i64 * rhs.0[x] as i64
                    + self.0[y * 4 + 1] as i64 * rhs.0[4 + x] as i64
                    + self.0[y * 4 + 2] as i64 * rhs.0[8 + x] as i64
                    + self.0[y * 4 + 3] as i64 * rhs.0[12 + x] as i64)
                    >> 12) as i32;
            }
        }
        ret
    }
}

impl ops::Mul for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Self::Output {
        self.some_mul_test(rhs)
    }
}

impl Default for Matrix {
    fn default() -> Self {
        #[rustfmt::skip]
        Matrix([
            1 << 12, 0 << 12, 0 << 12, 0 << 12,
            0 << 12, 1 << 12, 0 << 12, 0 << 12,
            0 << 12, 0 << 12, 1 << 12, 0 << 12,
            0 << 12, 0 << 12, 0 << 12, 1 << 12,
        ])
    }
}

struct Vector<const SIZE: usize>([i32; SIZE]);

struct Vertex {
    coords: Vector<4>,
    tex_coords: Vector<2>,
    color: u32,
}

#[derive(Default)]
struct Polygon {
    w_buffer: bool,
    pal_addr: u32,
}

#[repr(u8)]
enum MtxMode {
    Projection = 0,
    ModelView = 1,
    ModelViewVec = 2,
    Texture,
}

impl From<u8> for MtxMode {
    fn from(value: u8) -> Self {
        debug_assert!(value <= MtxMode::Texture as u8);
        unsafe { mem::transmute(value) }
    }
}

#[derive(Default)]
struct Matrices {
    proj: Matrix,
    proj_stack: Matrix,
    model: Matrix,
    model_stack: [Matrix; 32],
    vec: Matrix,
    vec_stack: [Matrix; 32],
    tex: Matrix,
    tex_stack: Matrix,
    clip: Matrix,
}

#[derive(Default)]
struct Polygons {
    saved: Polygon,
}

pub struct Gpu3D {
    gx_stat: GxStat,
    gx_fifo: u32,
    gx_fifo_count: u32,
    cmd_fifo: VecDeque<Entry>,
    cmd_pipe_size: u8,
    last_total_cycles: u64,
    mtx_queue: u32,
    mtx_mode: MtxMode,
    matrices: Matrices,
    clip_dirty: bool,
    pub flushed: bool,
    polygons: Polygons,
    polygon_attr: u32,
    viewport: [Viewport; 2],
    test_queue: u32,
}

impl Gpu3D {
    pub fn new() -> Self {
        Gpu3D {
            gx_stat: GxStat::from(0x04000000),
            gx_fifo: 0,
            gx_fifo_count: 0,
            cmd_fifo: VecDeque::new(),
            cmd_pipe_size: 0,
            last_total_cycles: 0,
            mtx_queue: 0,
            mtx_mode: MtxMode::Projection,
            matrices: Matrices::default(),
            clip_dirty: false,
            flushed: false,
            polygons: Polygons::default(),
            polygon_attr: 0,
            viewport: [Viewport::default(); 2],
            test_queue: 0,
        }
    }

    fn is_cmd_fifo_full(&self) -> bool {
        self.cmd_fifo.len() - self.cmd_pipe_size as usize >= 256
    }

    fn is_cmd_fifo_half_full(&self) -> bool {
        self.cmd_fifo.len() - self.cmd_pipe_size as usize >= 128
    }

    fn is_cmd_fifo_empty(&self) -> bool {
        self.cmd_fifo.len() <= 4
    }

    fn is_cmd_pipe_full(&self) -> bool {
        self.cmd_pipe_size == 4
    }

    pub fn run_cmds(&mut self, total_cycles: u64, emu: &mut Emu) {
        if self.cmd_fifo.is_empty() || !self.gx_stat.geometry_busy() || self.flushed {
            self.last_total_cycles = total_cycles;
            return;
        }

        let cycle_diff = (total_cycles - self.last_total_cycles) as u32;
        self.last_total_cycles = total_cycles;
        let mut executed_cycles = 0;

        let refresh_state = |gpu_3d: &mut Self| {
            gpu_3d.gx_stat.set_num_entries_cmd_fifo(u9::new(gpu_3d.cmd_fifo.len() as u16 - gpu_3d.cmd_pipe_size as u16));
            gpu_3d.gx_stat.set_cmd_fifo_empty(gpu_3d.is_cmd_fifo_empty());
            gpu_3d.gx_stat.set_geometry_busy(!gpu_3d.cmd_fifo.is_empty());

            if !gpu_3d.gx_stat.cmd_fifo_less_half_full() && !gpu_3d.is_cmd_fifo_half_full() {
                gpu_3d.gx_stat.set_cmd_fifo_less_half_full(true);
                io_dma!(emu, ARM9).trigger_all(DmaTransferMode::GeometryCmdFifo, get_cm_mut!(emu));
            }

            match u8::from(gpu_3d.gx_stat.cmd_fifo_irq()) {
                0 | 3 => {}
                1 => {
                    if gpu_3d.gx_stat.cmd_fifo_less_half_full() {
                        get_cpu_regs_mut!(emu, ARM9).send_interrupt(InterruptFlag::GeometryCmdFifo, get_cm_mut!(emu));
                    }
                }
                2 => {
                    if gpu_3d.gx_stat.cmd_fifo_empty() {
                        get_cpu_regs_mut!(emu, ARM9).send_interrupt(InterruptFlag::GeometryCmdFifo, get_cm_mut!(emu));
                    }
                }
                _ => unsafe { unreachable_unchecked() },
            }
        };

        while !self.cmd_fifo.is_empty() && executed_cycles < cycle_diff && !self.flushed {
            let mut params = Vec::new();
            let entry = unsafe { *self.cmd_fifo.front().unwrap_unchecked() };
            let mut param_count = FIFO_PARAM_COUNTS[entry.cmd as usize];
            if param_count > 1 {
                if param_count as usize > self.cmd_fifo.len() {
                    refresh_state(self);
                    break;
                }

                params.reserve(param_count as usize);
                for _ in 0..param_count {
                    params.push(unsafe { self.cmd_fifo.pop_front().unwrap_unchecked().param });
                }
            } else {
                param_count = 1;
                self.cmd_fifo.pop_front();
            }

            match entry.cmd {
                0x10 => self.exe_mtx_mode(entry.param),
                0x11 => self.exe_mtx_push(),
                0x12 => self.exe_mtx_pop(entry.param),
                0x13 => self.exe_mtx_store(entry.param),
                0x14 => self.exe_mtx_restore(entry.param),
                0x15 => self.exe_mtx_identity(),
                0x16 => self.exe_mtx_load44(params.try_into().unwrap()),
                0x17 => self.exe_mtx_load43(params.try_into().unwrap()),
                0x18 => self.exe_mtx_mult44(params.try_into().unwrap()),
                0x19 => self.exe_mtx_mult43(params.try_into().unwrap()),
                0x1A => self.exe_mtx_mult33(params.try_into().unwrap()),
                0x1B => self.exe_mtx_scale(params.try_into().unwrap()),
                0x1C => self.exe_mtx_trans(params.try_into().unwrap()),
                0x21 => {}
                0x22 => {}
                0x23 => {}
                0x24 => {}
                0x25 => {}
                0x26 => {}
                0x27 => {}
                0x28 => {}
                0x29 => self.exe_polygon_attr(entry.param),
                0x2A => {}
                0x2B => self.exe_pltt_base(entry.param),
                0x30 => {}
                0x31 => {}
                0x32 => {}
                0x33 => {}
                0x34 => {}
                0x40 => {}
                0x41 => {}
                0x50 => self.exe_swap_buffers(entry.param),
                0x60 => self.exe_viewport(entry.param),
                0x70 => {}
                0x71 => {}
                0x72 => {}
                _ => {
                    todo!("{:x}", entry.cmd);
                }
            }
            executed_cycles += 2;

            self.cmd_pipe_size = 4 - ((self.cmd_pipe_size + param_count) & 1);
            if self.cmd_pipe_size as usize > self.cmd_fifo.len() {
                self.cmd_pipe_size = self.cmd_fifo.len() as u8;
            }

            refresh_state(self);
        }

        if !self.is_cmd_fifo_full() {
            get_cpu_regs_mut!(emu, ARM9).unhalt(1);
        }
    }

    fn exe_mtx_mode(&mut self, param: u32) {
        self.mtx_mode = MtxMode::from((param & 0x3) as u8);
    }

    fn decrease_mtx_queue(&mut self) {
        self.mtx_queue -= 1;
        if self.mtx_queue == 0 {
            self.gx_stat.set_mtx_stack_busy(false);
        }
    }

    fn exe_mtx_push(&mut self) {
        match self.mtx_mode {
            MtxMode::Projection => {
                if u8::from(self.gx_stat.proj_mtx_stack_lvl()) == 0 {
                    self.matrices.proj_stack = self.matrices.proj;
                    self.gx_stat.set_proj_mtx_stack_lvl(u1::new(1));
                } else {
                    self.gx_stat.set_mtx_stack_overflow_underflow_err(true);
                }
            }
            MtxMode::ModelView | MtxMode::ModelViewVec => {
                let ptr = u8::from(self.gx_stat.pos_vec_mtx_stack_lvl());

                if ptr >= 30 {
                    self.gx_stat.set_mtx_stack_overflow_underflow_err(true);
                }

                if ptr < 31 {
                    self.matrices.model_stack[ptr as usize] = self.matrices.model;
                    self.matrices.vec_stack[ptr as usize] = self.matrices.vec;
                    self.gx_stat.set_pos_vec_mtx_stack_lvl(u5::new(ptr + 1));
                }
            }
            MtxMode::Texture => self.matrices.tex_stack = self.matrices.tex,
        }

        self.decrease_mtx_queue();
    }

    fn exe_mtx_pop(&mut self, param: u32) {
        match self.mtx_mode {
            MtxMode::Projection => {
                if u8::from(self.gx_stat.proj_mtx_stack_lvl()) == 1 {
                    self.matrices.proj = self.matrices.proj_stack;
                    self.gx_stat.set_proj_mtx_stack_lvl(u1::new(0));
                    self.clip_dirty = true;
                } else {
                    self.gx_stat.set_mtx_stack_overflow_underflow_err(true);
                }
            }
            MtxMode::ModelView | MtxMode::ModelViewVec => {
                let ptr = u8::from(self.gx_stat.pos_vec_mtx_stack_lvl()) as i8 - (((param << 2) as i8) >> 2);
                if ptr >= 30 {
                    self.gx_stat.set_mtx_stack_overflow_underflow_err(true);
                }

                if ptr < 31 {
                    self.gx_stat.set_pos_vec_mtx_stack_lvl(u5::new(ptr as u8));
                    self.matrices.model = self.matrices.model_stack[ptr as usize];
                    self.matrices.vec = self.matrices.vec_stack[ptr as usize];
                    self.clip_dirty = true;
                }
            }
            MtxMode::Texture => self.matrices.tex = self.matrices.tex_stack,
        }

        self.decrease_mtx_queue();
    }

    fn exe_mtx_store(&mut self, param: u32) {
        match self.mtx_mode {
            MtxMode::Projection => self.matrices.proj_stack = self.matrices.proj,
            MtxMode::ModelView | MtxMode::ModelViewVec => {
                let addr = param & 0x1F;

                if addr == 31 {
                    self.gx_stat.set_mtx_stack_overflow_underflow_err(true);
                }

                self.matrices.model_stack[addr as usize] = self.matrices.model;
                self.matrices.vec_stack[addr as usize] = self.matrices.vec;
            }
            MtxMode::Texture => self.matrices.tex_stack = self.matrices.tex,
        }
    }

    fn exe_mtx_restore(&mut self, param: u32) {
        match self.mtx_mode {
            MtxMode::Projection => {
                self.matrices.proj = self.matrices.proj_stack;
                self.clip_dirty = true;
            }
            MtxMode::ModelView | MtxMode::ModelViewVec => {
                let addr = param & 0x1F;

                if addr == 31 {
                    self.gx_stat.set_mtx_stack_overflow_underflow_err(true);
                }

                self.matrices.model = self.matrices.model_stack[addr as usize];
                self.matrices.vec = self.matrices.vec_stack[addr as usize];
                self.clip_dirty = true;
            }
            MtxMode::Texture => self.matrices.tex = self.matrices.tex_stack,
        }
    }

    fn exe_mtx_identity(&mut self) {
        match self.mtx_mode {
            MtxMode::Projection => self.matrices.proj = Matrix::default(),
            MtxMode::ModelView => self.matrices.model = Matrix::default(),
            MtxMode::ModelViewVec => {
                self.matrices.model = Matrix::default();
                self.matrices.vec = Matrix::default();
            }
            MtxMode::Texture => self.matrices.tex = Matrix::default(),
        }
    }

    fn mtx_load(&mut self, mtx: Matrix) {
        match self.mtx_mode {
            MtxMode::Projection => {
                self.matrices.proj = mtx;
                self.clip_dirty = true;
            }
            MtxMode::ModelView => {
                self.matrices.model = mtx;
                self.clip_dirty = true;
            }
            MtxMode::ModelViewVec => {
                self.matrices.model = mtx;
                self.matrices.vec = mtx;
                self.clip_dirty = true;
            }
            MtxMode::Texture => self.matrices.tex = mtx,
        }
    }

    fn exe_mtx_load44(&mut self, param: [u32; 16]) {
        self.mtx_load(unsafe { mem::transmute(param) });
    }

    fn exe_mtx_load43(&mut self, param: [u32; 12]) {
        let mut mtx = Matrix::default();
        for i in 0..4 {
            mtx.0[i * 4..i * 4 + 3].copy_from_slice(unsafe { mem::transmute(&param[i * 3..i * 3 + 3]) });
        }
        self.mtx_load(mtx);
    }

    fn mtx_mult(&mut self, mtx: Matrix) {
        match self.mtx_mode {
            MtxMode::Projection => {
                self.matrices.proj = mtx * self.matrices.proj;
                self.clip_dirty = true;
            }
            MtxMode::ModelView => {
                self.matrices.model = mtx * self.matrices.model;
                self.clip_dirty = true;
            }
            MtxMode::ModelViewVec => {
                self.matrices.model = mtx * self.matrices.model;
                self.matrices.vec = mtx * self.matrices.vec;
                self.clip_dirty = true;
            }
            MtxMode::Texture => {
                self.matrices.tex = mtx * self.matrices.tex;
            }
        }
    }

    fn exe_mtx_mult44(&mut self, param: [u32; 16]) {
        self.mtx_mult(unsafe { mem::transmute(param) });
    }

    fn exe_mtx_mult43(&mut self, param: [u32; 12]) {
        let mut mtx = Matrix::default();
        for i in 0..4 {
            mtx.0[i * 4..i * 4 + 3].copy_from_slice(unsafe { mem::transmute(&param[i * 3..i * 3 + 3]) });
        }
        self.mtx_mult(mtx);
    }

    fn exe_mtx_mult33(&mut self, param: [u32; 9]) {
        let mut mtx = Matrix::default();
        for i in 0..3 {
            mtx.0[i * 4..i * 4 + 3].copy_from_slice(unsafe { mem::transmute(&param[i * 3..i * 3 + 3]) });
        }
        self.mtx_mult(mtx);
    }

    fn exe_mtx_scale(&mut self, param: [u32; 3]) {
        let mut mtx = Matrix::default();
        for i in 0..3 {
            mtx.0[i * 5] = param[i] as i32;
        }
        self.mtx_mult(mtx);
    }

    fn exe_mtx_trans(&mut self, param: [u32; 3]) {
        let mut mtx = Matrix::default();
        mtx.0[12..15].copy_from_slice(unsafe { mem::transmute(param.as_slice()) });
        self.mtx_mult(mtx);
    }

    fn exe_polygon_attr(&mut self, param: u32) {
        self.polygon_attr = param;
    }

    fn exe_pltt_base(&mut self, param: u32) {
        self.polygons.saved.pal_addr = param & 0x1FFF;
    }

    fn exe_swap_buffers(&mut self, param: u32) {
        self.polygons.saved.w_buffer = (param & 0x2) != 0;
        self.flushed = true;
    }

    fn exe_viewport(&mut self, param: u32) {
        self.viewport[0] = Viewport::from(param);
    }

    pub fn swap_buffers(&mut self) {
        self.flushed = false;
    }

    fn queue_entry(&mut self, entry: Entry, emu: &mut Emu) {
        if self.cmd_fifo.is_empty() && !self.is_cmd_pipe_full() {
            self.cmd_fifo.push_back(entry);
            self.cmd_pipe_size += 1;
            self.gx_stat.set_geometry_busy(true);
        } else {
            if self.is_cmd_fifo_full() {
                get_mem_mut!(emu).breakout_imm = true;
                get_cpu_regs_mut!(emu, ARM9).halt(1);
            }

            self.cmd_fifo.push_back(entry);
            self.gx_stat.set_num_entries_cmd_fifo(u9::new(self.cmd_fifo.len() as u16 - self.cmd_pipe_size as u16));
            self.gx_stat.set_cmd_fifo_empty(false);

            self.gx_stat.set_cmd_fifo_less_half_full(!self.is_cmd_fifo_half_full());
        }

        match entry.cmd {
            0x11 | 0x12 => {
                self.mtx_queue += 1;
                self.gx_stat.set_mtx_stack_busy(true);
            }
            0x70 | 0x71 | 0x72 => {
                self.test_queue += 1;
            }
            _ => {}
        }
    }

    pub fn get_clip_mtx_result(&self, index: usize) -> u32 {
        0
    }

    pub fn get_vec_mtx_result(&self, index: usize) -> u32 {
        0
    }

    pub fn get_gx_stat(&self) -> u32 {
        u32::from(self.gx_stat)
    }

    pub fn set_gx_fifo(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        if self.gx_fifo == 0 {
            self.gx_fifo = value & mask;
        } else {
            self.queue_entry(Entry::new(self.gx_fifo as u8, value & mask), emu);
            self.gx_fifo_count += 1;

            if self.gx_fifo_count == FIFO_PARAM_COUNTS[(self.gx_fifo & 0xFF) as usize] as u32 {
                self.gx_fifo >>= 8;
                self.gx_fifo_count = 0;
            }
        }

        while self.gx_fifo != 0 && FIFO_PARAM_COUNTS[(self.gx_fifo & 0xFF) as usize] == 0 {
            self.queue_entry(Entry::new(self.gx_fifo as u8, value & mask), emu);
            self.gx_fifo >>= 8;
        }
    }

    pub fn set_mtx_mode(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x10, value & mask), emu);
    }

    pub fn set_mtx_push(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x11, value & mask), emu);
    }

    pub fn set_mtx_pop(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x12, value & mask), emu);
    }

    pub fn set_mtx_store(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x13, value & mask), emu);
    }

    pub fn set_mtx_restore(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x14, value & mask), emu);
    }

    pub fn set_mtx_identity(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x15, value & mask), emu);
    }

    pub fn set_mtx_load44(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x16, value & mask), emu);
    }

    pub fn set_mtx_load43(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x17, value & mask), emu);
    }

    pub fn set_mtx_mult44(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x18, value & mask), emu);
    }

    pub fn set_mtx_mult43(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x19, value & mask), emu);
    }

    pub fn set_mtx_mult33(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x1A, value & mask), emu);
    }

    pub fn set_mtx_scale(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x1B, value & mask), emu);
    }

    pub fn set_mtx_trans(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x1C, value & mask), emu);
    }

    pub fn set_color(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x20, value & mask), emu);
    }

    pub fn set_normal(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x21, value & mask), emu);
    }

    pub fn set_tex_coord(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x22, value & mask), emu);
    }

    pub fn set_vtx16(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x23, value & mask), emu);
    }

    pub fn set_vtx10(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x24, value & mask), emu);
    }

    pub fn set_vtx_x_y(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x25, value & mask), emu);
    }

    pub fn set_vtx_x_z(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x26, value & mask), emu);
    }

    pub fn set_vtx_y_z(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x27, value & mask), emu);
    }

    pub fn set_vtx_diff(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x28, value & mask), emu);
    }

    pub fn set_polygon_attr(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x29, value & mask), emu);
    }

    pub fn set_tex_image_param(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x2A, value & mask), emu);
    }

    pub fn set_pltt_base(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x2B, value & mask), emu);
    }

    pub fn set_dif_amb(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x30, value & mask), emu);
    }

    pub fn set_spe_emi(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x31, value & mask), emu);
    }

    pub fn set_light_vector(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x32, value & mask), emu);
    }

    pub fn set_light_color(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x33, value & mask), emu);
    }

    pub fn set_shininess(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x34, value & mask), emu);
    }

    pub fn set_begin_vtxs(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x40, value & mask), emu);
    }

    pub fn set_end_vtxs(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x41, value & mask), emu);
    }

    pub fn set_swap_buffers(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x50, value & mask), emu);
    }

    pub fn set_viewport(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x60, value & mask), emu);
    }

    pub fn set_box_test(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x70, value & mask), emu);
    }

    pub fn set_pos_test(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x71, value & mask), emu);
    }

    pub fn set_vec_test(&mut self, mask: u32, value: u32, emu: &mut Emu) {
        self.queue_entry(Entry::new(0x72, value & mask), emu);
    }

    pub fn set_gx_stat(&mut self, mut mask: u32, value: u32) {
        if value & (1 << 15) != 0 {
            self.gx_stat = (u32::from(self.gx_stat) & !0xA000).into();
        }

        mask &= 0xC0000000;
        self.gx_stat = ((u32::from(self.gx_stat) & !mask) | (value & mask)).into();
    }
}
