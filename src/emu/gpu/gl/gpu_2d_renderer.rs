use crate::emu::gpu::gl::gl_utils::{
    create_fb_color, create_fb_depth, create_mem_texture1d, create_mem_texture2d, create_pal_texture, create_program, create_shader, sub_mem_texture1d, sub_mem_texture2d, sub_pal_texture,
};
use crate::emu::gpu::gpu::{DISPLAY_HEIGHT, DISPLAY_WIDTH};
use crate::emu::gpu::gpu_2d::Gpu2DEngine::{A, B};
use crate::emu::gpu::gpu_2d::{DispCnt, Gpu2DInner};
use crate::emu::memory::mem::Memory;
use crate::emu::memory::oam::{OamAttrib0, OamAttrib1, OamAttribs, OamGfxMode, OamObjMode};
use crate::emu::memory::{regions, vram};
use crate::presenter::{Presenter, PRESENTER_SCREEN_HEIGHT, PRESENTER_SCREEN_WIDTH, PRESENTER_SUB_TOP_SCREEN};
use crate::utils;
use crate::utils::{HeapMemU32, HeapMemU8, StrErr};
use gl::types::{GLint, GLuint};
use gl::{TEXTURE0, TEXTURE1, TEXTURE2};
use std::ops::Deref;
use std::sync::{Condvar, Mutex};
use std::time::Duration;
use std::{mem, ptr, thread};

#[derive(Default)]
pub struct GpuMemBuf {
    pub bg_a: HeapMemU8<{ vram::BG_A_SIZE as usize }>,
    pub obj_a: HeapMemU8<{ vram::OBJ_A_SIZE as usize }>,
    pub bg_a_ext_palette: HeapMemU8<{ 32 * 1024 }>,
    pub obj_a_ext_palette: HeapMemU8<{ 8 * 1024 }>,
    pub bg_a_ext_palette_mapped: [bool; 4],
    pub obj_a_ext_palette_mapped: bool,

    pub bg_b: HeapMemU8<{ vram::BG_B_SIZE as usize }>,
    pub obj_b: HeapMemU8<{ vram::OBJ_B_SIZE as usize }>,
    pub bg_b_ext_palette: HeapMemU8<{ 32 * 1024 }>,
    pub obj_b_ext_palette: HeapMemU8<{ 8 * 1024 }>,
    pub bg_b_ext_palette_mapped: [bool; 4],
    pub obj_b_ext_palette_mapped: bool,

    pub pal_a: HeapMemU8<{ regions::STANDARD_PALETTES_SIZE as usize / 2 }>,
    pub pal_b: HeapMemU8<{ regions::STANDARD_PALETTES_SIZE as usize / 2 }>,
    pub oam_a: HeapMemU8<{ regions::OAM_SIZE as usize / 2 }>,
    pub oam_b: HeapMemU8<{ regions::OAM_SIZE as usize / 2 }>,
}

impl GpuMemBuf {
    fn read(&mut self, mem: &mut Memory) {
        mem.vram.read_all_bg_a(&mut self.bg_a);
        mem.vram.read_all_obj_a(&mut self.obj_a);
        mem.vram.read_all_bg_a_ext_palette(&mut self.bg_a_ext_palette);
        mem.vram.read_all_obj_a_ext_palette(&mut self.obj_a_ext_palette);
        for slot in 0..4 {
            self.bg_a_ext_palette_mapped[slot] = mem.vram.is_bg_ext_palette_mapped::<{ A }>(slot);
        }
        self.obj_a_ext_palette_mapped = mem.vram.is_obj_ext_palette_mapped::<{ A }>();

        mem.vram.read_bg_b(&mut self.bg_b);
        mem.vram.read_all_obj_b(&mut self.obj_b);
        mem.vram.read_all_bg_b_ext_palette(&mut self.bg_b_ext_palette);
        mem.vram.read_all_obj_b_ext_palette(&mut self.obj_b_ext_palette);
        for slot in 0..4 {
            self.bg_b_ext_palette_mapped[slot] = mem.vram.is_bg_ext_palette_mapped::<{ B }>(slot);
        }
        self.obj_b_ext_palette_mapped = mem.vram.is_obj_ext_palette_mapped::<{ B }>();

        if mem.palettes.dirty {
            mem.palettes.dirty = false;
            self.pal_a.copy_from_slice(&mem.palettes.mem[..mem.palettes.mem.len() / 2]);
            self.pal_b.copy_from_slice(&mem.palettes.mem[mem.palettes.mem.len() / 2..]);
        }

        if mem.oam.dirty {
            mem.oam.dirty = false;
            self.oam_a.copy_from_slice(&mem.oam.mem[..mem.oam.mem.len() / 2]);
            self.oam_b.copy_from_slice(&mem.oam.mem[mem.oam.mem.len() / 2..]);
        }
    }
}

#[derive(Clone)]
#[repr(C)]
struct WinBgUbo {
    win_h: [u32; DISPLAY_HEIGHT * 2],
    win_v: [u32; DISPLAY_HEIGHT * 2],
    win_in: [u32; DISPLAY_HEIGHT],
    win_out: [u32; DISPLAY_HEIGHT],
}

#[derive(Clone)]
#[repr(C)]
struct BgUbo {
    h_ofs: [u32; DISPLAY_HEIGHT * 4],
    v_ofs: [u32; DISPLAY_HEIGHT * 4],
}

#[derive(Clone)]
struct GpuRegs {
    disp_cnts: [u32; DISPLAY_HEIGHT],
    bg_cnts: [u32; DISPLAY_HEIGHT * 4],
    win_bg_ubo: WinBgUbo,
    bg_ubo: BgUbo,
    batch_counts: [u8; DISPLAY_HEIGHT],
    current_batch_count_index: usize,
}

impl Default for GpuRegs {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

impl GpuRegs {
    fn on_scanline(&mut self, inner: &Gpu2DInner, line: u8) {
        let updated = self.disp_cnts[self.current_batch_count_index] != u32::from(inner.disp_cnt);
        let updated = updated || {
            let mut updated = false;
            for i in 0..4 {
                if self.bg_cnts[self.current_batch_count_index * 4 + i] != u16::from(inner.bg_cnt[i]) as u32 {
                    updated = true;
                    break;
                }
            }
            updated
        };

        if updated {
            // println!(
            //     "{line} {:x} {:x} {:x} {:x} {:x} {:x} {:x} {:x} {:x} {:x}",
            //     self.disp_cnts[self.current_batch_count_index],
            //     u32::from(inner.disp_cnt),
            //     self.bg_cnts[self.current_batch_count_index],
            //     u16::from(inner.bg_cnt[0]),
            //     self.bg_cnts[DISPLAY_HEIGHT + self.current_batch_count_index],
            //     u16::from(inner.bg_cnt[1]),
            //     self.bg_cnts[2 * DISPLAY_HEIGHT + self.current_batch_count_index],
            //     u16::from(inner.bg_cnt[2]),
            //     self.bg_cnts[3 * DISPLAY_HEIGHT + self.current_batch_count_index],
            //     u16::from(inner.bg_cnt[3]),
            // );
            self.disp_cnts[line as usize] = u32::from(inner.disp_cnt);
            for i in 0..4 {
                self.bg_cnts[line as usize * 4 + i] = u16::from(inner.bg_cnt[i]) as u32;
            }
            self.current_batch_count_index = line as usize;
        } else {
            self.batch_counts[self.current_batch_count_index] += 1;
        }

        for i in 0..2 {
            self.win_bg_ubo.win_h[i * DISPLAY_HEIGHT + line as usize] = inner.win_h[i] as u32;
            self.win_bg_ubo.win_v[i * DISPLAY_HEIGHT + line as usize] = inner.win_v[i] as u32;
        }
        self.win_bg_ubo.win_in[line as usize] = inner.win_in as u32;
        self.win_bg_ubo.win_out[line as usize] = inner.win_out as u32;

        for i in 0..4 {
            self.bg_ubo.h_ofs[i * DISPLAY_HEIGHT + line as usize] = inner.bg_h_ofs[i] as u32;
            self.bg_ubo.v_ofs[i * DISPLAY_HEIGHT + line as usize] = inner.bg_v_ofs[i] as u32;
        }
    }
}

macro_rules! shader_source {
    ($name:expr) => {{
        #[cfg(target_os = "vita")]
        {
            include_bytes!(concat!("shaders/cg/", $name, ".cg"))
        }
        #[cfg(target_os = "linux")]
        {
            include_bytes!(concat!("shaders/glsl/", $name, ".glsl"))
        }
    }};
}

const WIN_BG_VERT_SHADER_SRC: &[u8] = shader_source!("win_bg_vert");
const WIN_BG_FRAG_SHADER_SRC: &[u8] = shader_source!("win_bg_frag");

const OBJ_VERT_SHADER_SRC: &[u8] = shader_source!("obj_vert");
const OBJ_FRAG_SHADER_SRC: &[u8] = shader_source!("obj_frag");

const BG_VERT_SHADER_SRC: &[u8] = shader_source!("bg_vert");
const BG_FRAG_SHADER_SRC: &[u8] = shader_source!("bg_frag");

const fn generate_obj_vertices() -> [f32; 128 * 4 * 2] {
    let mut vertices: [f32; 128 * 4 * 2] = unsafe { mem::zeroed() };
    let mut i = 0;
    while i < vertices.len() {
        // top left
        vertices[i] = 0f32;
        vertices[i + 1] = 0f32;
        // top right
        vertices[i + 2] = 1f32;
        vertices[i + 3] = 0f32;
        // bottom right
        vertices[i + 4] = 1f32;
        vertices[i + 5] = 1f32;
        // bottom left
        vertices[i + 6] = 0f32;
        vertices[i + 7] = 1f32;
        i += 4 * 2;
    }
    vertices
}

const OBJ_VERTICES: [f32; 128 * 4 * 2] = generate_obj_vertices();

const fn generate_oam_indices() -> [u8; 128 * 4] {
    let mut indices: [u8; 128 * 4] = unsafe { mem::zeroed() };
    let mut i = 0;
    while i < indices.len() {
        indices[i] = (i / 4) as u8;
        indices[i + 1] = (i / 4) as u8;
        indices[i + 2] = (i / 4) as u8;
        indices[i + 3] = (i / 4) as u8;
        i += 4;
    }
    indices
}

const OAM_INDICES: [u8; 128 * 4] = generate_oam_indices();

struct Gpu2dTextures {
    oam: GLuint,
    obj: GLuint,
    obj_width: u32,
    obj_height: u32,
    bg: GLuint,
    bg_width: u32,
    bg_height: u32,
    pal: GLuint,
}

impl Gpu2dTextures {
    fn new(oam: GLuint, obj: GLuint, obj_width: u32, obj_height: u32, bg: GLuint, bg_width: u32, bg_height: u32, pal: GLuint) -> Self {
        Gpu2dTextures {
            oam,
            obj,
            obj_width,
            obj_height,
            bg,
            bg_width,
            bg_height,
            pal,
        }
    }
}

struct Gpu2dShared {
    win_bg_program: GLuint,
    win_bg_disp_cnt_loc: GLint,
    win_bg_ubo: GLuint,
    obj_program: GLuint,
    obj_vao: GLuint,
    oam_indices: Vec<[u16; 6]>,
    obj_disp_cnt_loc: GLint,
    bg_program: GLuint,
    bg_disp_cnt_loc: GLint,
    bg_cnts_loc: GLint,
    bg_ubo: GLuint,
    bg_vertices_buf: Vec<[f32; 6 * 3]>,
    read_buf: HeapMemU32<{ DISPLAY_WIDTH * DISPLAY_HEIGHT }>,
}

impl Gpu2dShared {
    fn new() -> Self {
        unsafe {
            let (win_bg_program, win_bg_disp_cnt_loc, win_bg_ubo) = {
                let vert_shader = create_shader(WIN_BG_VERT_SHADER_SRC, gl::VERTEX_SHADER).unwrap();
                let frag_shader = create_shader(WIN_BG_FRAG_SHADER_SRC, gl::FRAGMENT_SHADER).unwrap();
                let program = create_program(&[vert_shader, frag_shader]).unwrap();
                gl::DeleteShader(vert_shader);
                gl::DeleteShader(frag_shader);

                gl::UseProgram(program);

                gl::BindAttribLocation(program, 0, "position\0".as_ptr() as _);

                let disp_cnt_loc = gl::GetUniformLocation(program, "dispCnt\0".as_ptr() as _);

                let mut ubo = 0;
                gl::GenBuffers(1, ptr::addr_of_mut!(ubo));
                gl::BindBuffer(gl::UNIFORM_BUFFER, ubo);

                gl::UniformBlockBinding(program, gl::GetUniformBlockIndex(program, "WinBgUbo\0".as_ptr() as _), 0);

                gl::BindBuffer(gl::UNIFORM_BUFFER, 0);
                gl::UseProgram(0);

                (program, disp_cnt_loc, ubo)
            };

            let (obj_program, obj_vao, obj_disp_cnt_loc) = {
                let vert_shader = create_shader(OBJ_VERT_SHADER_SRC, gl::VERTEX_SHADER).unwrap();
                let frag_shader = create_shader(OBJ_FRAG_SHADER_SRC, gl::FRAGMENT_SHADER).unwrap();
                let program = create_program(&[vert_shader, frag_shader]).unwrap();
                gl::DeleteShader(vert_shader);
                gl::DeleteShader(frag_shader);

                gl::UseProgram(program);

                let mut vertices_buf = 0;
                gl::GenBuffers(1, ptr::addr_of_mut!(vertices_buf));
                gl::BindBuffer(gl::ARRAY_BUFFER, vertices_buf);
                gl::BufferData(gl::ARRAY_BUFFER, (mem::size_of::<f32>() * OBJ_VERTICES.len()) as _, OBJ_VERTICES.as_ptr() as _, gl::STATIC_DRAW);

                let mut indices_buf = 0;
                gl::GenBuffers(1, ptr::addr_of_mut!(indices_buf));
                gl::BindBuffer(gl::ARRAY_BUFFER, indices_buf);
                gl::BufferData(gl::ARRAY_BUFFER, OAM_INDICES.len() as _, OAM_INDICES.as_ptr() as _, gl::STATIC_DRAW);

                let mut vao = 0;
                gl::GenVertexArrays(1, ptr::addr_of_mut!(vao));
                gl::BindVertexArray(vao);

                gl::BindBuffer(gl::ARRAY_BUFFER, vertices_buf);
                gl::EnableVertexAttribArray(0);
                gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, 0, 0 as _);

                gl::BindBuffer(gl::ARRAY_BUFFER, indices_buf);
                gl::EnableVertexAttribArray(1);
                gl::VertexAttribPointer(1, 1, gl::UNSIGNED_BYTE, gl::FALSE, 0, 0 as _);

                gl::BindVertexArray(0);
                gl::BindBuffer(gl::ARRAY_BUFFER, 0);

                gl::Uniform1i(gl::GetUniformLocation(program, "oamTex\0".as_ptr() as _), 0);
                gl::BindAttribLocation(program, 0, "position\0".as_ptr() as _);
                gl::BindAttribLocation(program, 1, "oamIndex\0".as_ptr() as _);

                gl::Uniform1i(gl::GetUniformLocation(program, "objTex\0".as_ptr() as _), 1);
                gl::Uniform1i(gl::GetUniformLocation(program, "palTex\0".as_ptr() as _), 2);

                let disp_cnt_loc = gl::GetUniformLocation(program, "dispCnt\0".as_ptr() as _);

                gl::UseProgram(0);

                (program, vao, disp_cnt_loc)
            };

            let (bg_program, bg_disp_cnt_loc, bg_cnts_loc, bg_ubo) = {
                let vert_shader = create_shader(BG_VERT_SHADER_SRC, gl::VERTEX_SHADER).unwrap();
                let frag_shader = create_shader(BG_FRAG_SHADER_SRC, gl::FRAGMENT_SHADER).unwrap();
                let program = create_program(&[vert_shader, frag_shader]).unwrap();
                gl::DeleteShader(vert_shader);
                gl::DeleteShader(frag_shader);

                gl::UseProgram(program);

                gl::BindAttribLocation(program, 0, "position\0".as_ptr() as _);

                let disp_cnt_loc = gl::GetUniformLocation(program, "dispCnt\0".as_ptr() as _);
                let bg_cnts_loc = gl::GetUniformLocation(program, "bgCnts\0".as_ptr() as _);

                gl::Uniform1i(gl::GetUniformLocation(program, "bgTex\0".as_ptr() as _), 0);
                gl::Uniform1i(gl::GetUniformLocation(program, "palTex\0".as_ptr() as _), 1);
                gl::Uniform1i(gl::GetUniformLocation(program, "winTex\0".as_ptr() as _), 2);

                let mut ubo = 0;
                gl::GenBuffers(1, ptr::addr_of_mut!(ubo));
                gl::BindBuffer(gl::UNIFORM_BUFFER, ubo);

                gl::UniformBlockBinding(program, gl::GetUniformBlockIndex(program, "BgUbo\0".as_ptr() as _), 0);

                gl::BindBuffer(gl::UNIFORM_BUFFER, 0);
                gl::UseProgram(0);

                (program, disp_cnt_loc, bg_cnts_loc, ubo)
            };

            Gpu2dShared {
                win_bg_program,
                win_bg_disp_cnt_loc,
                win_bg_ubo,
                obj_program,
                obj_vao,
                oam_indices: Vec::new(),
                obj_disp_cnt_loc,
                bg_program,
                bg_disp_cnt_loc,
                bg_cnts_loc,
                bg_ubo,
                bg_vertices_buf: Vec::new(),
                read_buf: HeapMemU32::new(),
            }
        }
    }

    unsafe fn draw_windows(&mut self, regs: &GpuRegs, fbs: &Gpu2dFbs, mem_buf: &GpuMemBuf, from_line: u8, to_line: u8) {
        let disp_cnt = DispCnt::from(regs.disp_cnts[from_line as usize]);
        if disp_cnt.obj_window_display_flag() {
            todo!()
        }
        if !disp_cnt.is_any_window_enabled() {
            return;
        }

        #[rustfmt::skip]
        let vertices = [
            -1f32, from_line as f32,
            1f32, from_line as f32,
            1f32, to_line as f32,
            -1f32, to_line as f32,
        ];

        gl::Uniform1i(self.win_bg_disp_cnt_loc, u32::from(disp_cnt) as _);

        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, 0, vertices.as_ptr() as _);
        gl::DrawArrays(gl::TRIANGLE_FAN, 0, 4);
    }

    unsafe fn draw_objects(&mut self, regs: &GpuRegs, fbs: &Gpu2dFbs, mem_buf: &GpuMemBuf, from_line: u8, to_line: u8) {
        let disp_cnt = DispCnt::from(regs.disp_cnts[from_line as usize]);
        if !disp_cnt.screen_display_obj() {
            return;
        }

        if disp_cnt.obj_window_display_flag() {
            self.assemble_oam::<true>(mem_buf, from_line, to_line);
        } else {
            self.assemble_oam::<false>(mem_buf, from_line, to_line);
        }

        if self.oam_indices.is_empty() {
            return;
        }

        gl::Uniform1i(self.obj_disp_cnt_loc, u32::from(disp_cnt) as _);

        gl::DrawElements(gl::TRIANGLES, (6 * self.oam_indices.len()) as _, gl::UNSIGNED_SHORT, self.oam_indices.as_ptr() as _);
    }

    unsafe fn draw_bg(&mut self, regs: &GpuRegs, fbs: &Gpu2dFbs, mem_buf: &GpuMemBuf, from_line: u8, to_line: u8) {
        let disp_cnt = regs.disp_cnts[from_line as usize];
        self.bg_vertices_buf.clear();

        let disp_cnt = DispCnt::from(disp_cnt);
        if disp_cnt.screen_display_bg3() {
            #[rustfmt::skip]
            self.bg_vertices_buf.push([
                -1f32, from_line as f32, 3f32,
                1f32, from_line as f32, 3f32,
                1f32, to_line as f32, 3f32,
                -1f32, from_line as f32, 3f32,
                1f32, to_line as f32, 3f32,
                -1f32, to_line as f32, 3f32,
            ])
        }
        if disp_cnt.screen_display_bg2() {
            #[rustfmt::skip]
            self.bg_vertices_buf.push([
                -1f32, from_line as f32, 2f32,
                1f32, from_line as f32, 2f32,
                1f32, to_line as f32, 2f32,
                -1f32, from_line as f32, 2f32,
                1f32, to_line as f32, 2f32,
                -1f32, to_line as f32, 2f32,
            ])
        }
        if disp_cnt.screen_display_bg1() {
            #[rustfmt::skip]
            self.bg_vertices_buf.push([
                -1f32, from_line as f32, 1f32,
                1f32, from_line as f32, 1f32,
                1f32, to_line as f32, 1f32,
                -1f32, from_line as f32, 1f32,
                1f32, to_line as f32, 1f32,
                -1f32, to_line as f32, 1f32,
            ])
        }
        if disp_cnt.screen_display_bg0() {
            #[rustfmt::skip]
            self.bg_vertices_buf.push([
                -1f32, from_line as f32, 0f32,
                1f32, from_line as f32, 0f32,
                1f32, to_line as f32, 0f32,
                -1f32, from_line as f32, 0f32,
                1f32, to_line as f32, 0f32,
                -1f32, to_line as f32, 0f32,
            ])
        }

        if self.bg_vertices_buf.is_empty() {
            return;
        }

        gl::Uniform1i(self.bg_disp_cnt_loc, u32::from(disp_cnt) as _);
        gl::Uniform1iv(self.bg_cnts_loc, 4, regs.bg_cnts[from_line as usize * 4..].as_ptr() as _);

        gl::BindBuffer(gl::UNIFORM_BUFFER, self.bg_ubo);
        gl::BufferData(gl::UNIFORM_BUFFER, mem::size_of::<BgUbo>() as _, ptr::addr_of!(regs.bg_ubo) as _, gl::DYNAMIC_DRAW);
        gl::BindBufferBase(gl::UNIFORM_BUFFER, 0, self.bg_ubo);

        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 0, self.bg_vertices_buf.as_ptr() as _);
        gl::DrawArrays(gl::TRIANGLES, 0, (self.bg_vertices_buf.len() * 6) as _);

        gl::BindBuffer(gl::UNIFORM_BUFFER, 0);
    }

    unsafe fn draw(&mut self, regs: &GpuRegs, fbs: &Gpu2dFbs, texs: &Gpu2dTextures, mem_buf: &GpuMemBuf) {
        macro_rules! draw_scanlines {
            ($draw_fn:expr) => {{
                let mut line = 0;
                while line < DISPLAY_HEIGHT {
                    let batch_count = regs.batch_counts[line];
                    let from_line = line as u8;
                    let to_line = line as u8 + batch_count as u8 + 1;
                    $draw_fn(self, regs, fbs, mem_buf, from_line, to_line);
                    line = to_line as usize;
                }
            }};
        }

        {
            gl::BindFramebuffer(gl::FRAMEBUFFER, fbs.win.fbo);
            gl::Viewport(0, 0, DISPLAY_WIDTH as _, DISPLAY_HEIGHT as _);
            gl::ClearColor(1f32, 0f32, 0f32, 0f32);
            gl::Clear(gl::COLOR_BUFFER_BIT);

            gl::UseProgram(self.win_bg_program);

            gl::BindBuffer(gl::UNIFORM_BUFFER, self.win_bg_ubo);
            gl::BufferData(gl::UNIFORM_BUFFER, mem::size_of::<WinBgUbo>() as _, ptr::addr_of!(regs.win_bg_ubo) as _, gl::DYNAMIC_DRAW);
            gl::BindBufferBase(gl::UNIFORM_BUFFER, 0, self.win_bg_ubo);

            draw_scanlines!(Self::draw_windows);

            // gl::ReadPixels(0, 0, DISPLAY_WIDTH as _, DISPLAY_HEIGHT as _, gl::RGBA, gl::UNSIGNED_BYTE, self.read_buf.as_mut_ptr() as _);

            gl::BindBuffer(gl::UNIFORM_BUFFER, 0);
            gl::UseProgram(0);
            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
        }

        gl::BindFramebuffer(gl::FRAMEBUFFER, fbs.composed.fbo);
        gl::Viewport(0, 0, DISPLAY_WIDTH as _, DISPLAY_HEIGHT as _);

        let backdrop = utils::read_from_mem::<u16>(mem_buf.pal_a.deref(), 0);
        let (r, g, b, _) = Self::rgb5_to_float8(backdrop);
        gl::ClearColor(r, g, b, 1f32);
        gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

        gl::BindTexture(gl::TEXTURE_2D, texs.oam);
        sub_mem_texture1d(regions::OAM_SIZE / 2, mem_buf.oam_a.deref());

        gl::BindTexture(gl::TEXTURE_2D, texs.obj);
        sub_mem_texture2d(texs.obj_width, texs.obj_height, mem_buf.obj_a.deref());

        gl::BindTexture(gl::TEXTURE_2D, texs.bg);
        sub_mem_texture2d(texs.bg_width, texs.bg_height, mem_buf.bg_a.deref());

        gl::BindTexture(gl::TEXTURE_2D, texs.pal);
        sub_pal_texture(regions::STANDARD_PALETTES_SIZE / 2, mem_buf.pal_a.deref());

        gl::BindTexture(gl::TEXTURE_2D, 0);

        {
            gl::UseProgram(self.obj_program);

            gl::BindVertexArray(self.obj_vao);

            gl::ActiveTexture(TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, texs.oam);

            gl::ActiveTexture(TEXTURE1);
            gl::BindTexture(gl::TEXTURE_2D, texs.obj);

            gl::ActiveTexture(TEXTURE2);
            gl::BindTexture(gl::TEXTURE_2D, texs.pal);

            draw_scanlines!(Self::draw_objects);

            gl::BindTexture(gl::TEXTURE_2D, 0);
            gl::BindVertexArray(0);
        }

        {
            gl::UseProgram(self.bg_program);

            gl::ActiveTexture(TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, texs.bg);

            gl::ActiveTexture(TEXTURE1);
            gl::BindTexture(gl::TEXTURE_2D, texs.pal);

            gl::ActiveTexture(TEXTURE2);
            gl::BindTexture(gl::TEXTURE_2D, fbs.win.color);

            draw_scanlines!(Self::draw_bg);

            gl::BindTexture(gl::TEXTURE_2D, 0);
        }

        gl::UseProgram(0);
    }

    fn assemble_oam<const OBJ_WINDOW: bool>(&mut self, mem_buf: &GpuMemBuf, from_line: u8, to_line: u8) {
        const OAM_COUNT: usize = regions::OAM_SIZE as usize / 2 / mem::size_of::<OamAttribs>();
        let oams = unsafe { (mem_buf.oam_a.as_ptr() as *const [OamAttribs; OAM_COUNT]).as_ref().unwrap_unchecked() };

        self.oam_indices.clear();
        for (i, oam) in oams.iter().enumerate() {
            let attrib0 = OamAttrib0::from(oam.attr0);
            let obj_mode = attrib0.get_obj_mode();
            if obj_mode == OamObjMode::Disabled {
                continue;
            }
            let gfx_mode = attrib0.get_gfx_mode();
            if OBJ_WINDOW && gfx_mode == OamGfxMode::Window {
                todo!()
            }

            let attrib1 = OamAttrib1::from(oam.attr1);
            let mut x = u16::from(attrib1.x()) as i32;
            if x >= DISPLAY_WIDTH as i32 {
                x -= 512;
            }
            let mut y = attrib0.y() as i32;
            if y >= DISPLAY_HEIGHT as i32 {
                y -= 256;
            }

            let (width, height) = match (u8::from(attrib0.shape()) << 2) | u8::from(attrib1.size()) {
                0x0 => (8, 8),
                0x1 => (16, 16),
                0x2 => (32, 32),
                0x3 => (64, 64),
                0x4 => (16, 8),
                0x5 => (32, 8),
                0x6 => (32, 16),
                0x7 => (64, 32),
                0x8 => (8, 16),
                0x9 => (8, 32),
                0xA => (16, 32),
                0xB => (32, 64),
                _ => continue,
            };

            if x + width < 0 || y + height < from_line as i32 || x >= DISPLAY_WIDTH as i32 || y >= to_line as i32 {
                continue;
            }

            if gfx_mode == OamGfxMode::Bitmap {
                todo!()
            }

            if obj_mode != OamObjMode::Normal {
                todo!()
            }

            if attrib0.is_8bit() {
                todo!()
            }

            let index_base = (i * 4) as u16;
            self.oam_indices.push([index_base, index_base + 1, index_base + 2, index_base, index_base + 2, index_base + 3]);
        }
    }

    fn rgb5_to_float8(color: u16) -> (f32, f32, f32, f32) {
        let r = (color & 0x1F) as f32;
        let g = ((color >> 5) & 0x1F) as f32;
        let b = ((color >> 10) & 0x1F) as f32;
        let a = ((color >> 15) & 1) as f32;
        (r / 31f32, g / 31f32, b / 31f32, a)
    }
}

struct Fbo {
    color: GLuint,
    depth: Option<GLuint>,
    fbo: GLuint,
}

impl Fbo {
    fn new(width: u32, height: u32, depth: bool) -> Result<Self, StrErr> {
        unsafe {
            let color = create_fb_color(width, height);

            let mut fbo = 0;
            gl::GenFramebuffers(1, ptr::addr_of_mut!(fbo));
            gl::BindFramebuffer(gl::FRAMEBUFFER, fbo);
            gl::FramebufferTexture2D(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_2D, color, 0);

            let depth = if depth {
                let depth = create_fb_depth(width, height);
                gl::FramebufferRenderbuffer(gl::FRAMEBUFFER, gl::DEPTH_ATTACHMENT, gl::RENDERBUFFER, depth);
                Some(depth)
            } else {
                None
            };

            let status = gl::CheckFramebufferStatus(gl::FRAMEBUFFER);
            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
            if status != gl::FRAMEBUFFER_COMPLETE {
                Err(StrErr::new(format!("Failed to create fbo: {status}")))
            } else {
                Ok(Fbo { color, depth, fbo })
            }
        }
    }
}

struct Gpu2dFbs {
    win: Fbo,
    composed: Fbo,
}

impl Gpu2dFbs {
    fn new() -> Self {
        Gpu2dFbs {
            win: Fbo::new(DISPLAY_WIDTH as u32, DISPLAY_HEIGHT as u32, false).unwrap(),
            composed: Fbo::new(DISPLAY_WIDTH as u32, DISPLAY_HEIGHT as u32, true).unwrap(),
        }
    }
}

pub struct Gpu2dRenderer {
    regs_a: [GpuRegs; 2],
    regs_b: [GpuRegs; 2],
    mem_buf: GpuMemBuf,
    mem_buf_swap: GpuMemBuf,
    drawing: Mutex<bool>,
    drawing_condvar: Condvar,
    tex_a: Gpu2dTextures,
    tex_b: Gpu2dTextures,
    shared: Gpu2dShared,
    fbs_a: Gpu2dFbs,
    fbs_b: Gpu2dFbs,
}

impl Gpu2dRenderer {
    pub fn new() -> Self {
        unsafe {
            gl::Enable(gl::DEPTH_TEST);
            gl::DepthFunc(gl::LESS);

            let oam_tex_a = create_mem_texture1d(regions::OAM_SIZE / 2);
            let obj_tex_a = create_mem_texture2d(1024, 256);
            let bg_tex_a = create_mem_texture2d(1024, 512);
            let pal_tex_a = create_pal_texture(regions::STANDARD_PALETTES_SIZE / 2);

            let oam_tex_b = create_mem_texture1d(regions::OAM_SIZE / 2);
            let obj_tex_b = create_mem_texture2d(1024, 128);
            let bg_tex_b = create_mem_texture2d(1024, 128);
            let pal_tex_b = create_pal_texture(regions::STANDARD_PALETTES_SIZE / 2);

            Gpu2dRenderer {
                regs_a: [GpuRegs::default(), GpuRegs::default()],
                regs_b: [GpuRegs::default(), GpuRegs::default()],
                mem_buf: GpuMemBuf::default(),
                mem_buf_swap: GpuMemBuf::default(),
                drawing: Mutex::new(false),
                drawing_condvar: Condvar::new(),
                tex_a: Gpu2dTextures::new(oam_tex_a, obj_tex_a, 1024, 256, bg_tex_a, 1024, 512, pal_tex_a),
                tex_b: Gpu2dTextures::new(oam_tex_b, obj_tex_b, 1024, 128, bg_tex_b, 1024, 128, pal_tex_b),
                shared: Gpu2dShared::new(),
                fbs_a: Gpu2dFbs::new(),
                fbs_b: Gpu2dFbs::new(),
            }
        }
    }

    pub fn on_scanline(&mut self, inner_a: &Gpu2DInner, inner_b: &Gpu2DInner, line: u8) {
        self.regs_a[1].on_scanline(inner_a, line);
        self.regs_b[1].on_scanline(inner_b, line);
    }

    pub fn on_frame(&mut self, mem: &mut Memory) {
        self.mem_buf_swap.read(mem);
    }

    pub fn reload_registers(&mut self) {
        self.regs_a[1] = GpuRegs::default();
        self.regs_b[1] = GpuRegs::default();
    }

    pub fn start_drawing(&mut self, mem: &mut Memory) {
        let mut drawing = self.drawing.lock().unwrap();
        // let mut drawing = self.drawing_condvar.wait_while(drawing, |drawing| *drawing).unwrap();
        // thread::sleep(Duration::from_millis(20));

        if !*drawing {
            self.mem_buf_swap.read(mem);
            // self.mem_buf.bg_a.copy_from_slice(self.mem_buf_swap.bg_a.deref());
            // self.mem_buf.obj_a.copy_from_slice(self.mem_buf_swap.obj_a.deref());
            // self.mem_buf.bg_a_ext_palette.copy_from_slice(self.mem_buf_swap.bg_a_ext_palette.deref());
            // self.mem_buf.obj_a_ext_palette.copy_from_slice(self.mem_buf_swap.obj_a_ext_palette.deref());
            // self.mem_buf.bg_a_ext_palette_mapped = self.mem_buf_swap.bg_a_ext_palette_mapped;
            // self.mem_buf.obj_a_ext_palette_mapped = self.mem_buf_swap.obj_a_ext_palette_mapped;
            // 
            // self.mem_buf.bg_b.copy_from_slice(self.mem_buf_swap.bg_b.deref());
            // self.mem_buf.obj_b.copy_from_slice(self.mem_buf_swap.obj_b.deref());
            // self.mem_buf.bg_b_ext_palette.copy_from_slice(self.mem_buf_swap.bg_b_ext_palette.deref());
            // self.mem_buf.obj_b_ext_palette.copy_from_slice(self.mem_buf_swap.obj_b_ext_palette.deref());
            // self.mem_buf.bg_b_ext_palette_mapped = self.mem_buf_swap.bg_b_ext_palette_mapped;
            // self.mem_buf.obj_b_ext_palette_mapped = self.mem_buf_swap.obj_b_ext_palette_mapped;
            // 
            // self.mem_buf.pal_a.copy_from_slice(self.mem_buf_swap.pal_a.deref());
            // self.mem_buf.pal_b.copy_from_slice(self.mem_buf_swap.pal_b.deref());
            // 
            // self.mem_buf.oam_a.copy_from_slice(self.mem_buf_swap.oam_a.deref());
            // self.mem_buf.oam_b.copy_from_slice(self.mem_buf_swap.oam_b.deref());

            self.regs_a[0] = self.regs_a[1].clone();
            self.regs_b[0] = self.regs_b[1].clone();

            *drawing = true;
            self.drawing_condvar.notify_one();
        }
    }

    pub fn wait_for_drawing(&self) {
        let drawing = self.drawing.lock().unwrap();
        let _drawing = self.drawing_condvar.wait_while(drawing, |drawing| *drawing).unwrap();
    }

    pub unsafe fn draw(&mut self, presenter: &mut Presenter) {
        {
            let drawing = self.drawing.lock().unwrap();
            let _drawing = self.drawing_condvar.wait_while(drawing, |drawing| !*drawing).unwrap();
        }

        self.shared.draw(&self.regs_a[0], &self.fbs_a, &self.tex_a, &self.mem_buf_swap);

        // gl::ReadPixels(0, 0, DISPLAY_WIDTH as _, DISPLAY_HEIGHT as _, gl::RGBA, gl::UNSIGNED_BYTE, self.read_buf.as_mut_ptr() as _);

        gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
        gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, 0);
        gl::BindFramebuffer(gl::READ_FRAMEBUFFER, self.fbs_a.composed.fbo);
        gl::Viewport(0, 0, PRESENTER_SCREEN_WIDTH as _, PRESENTER_SCREEN_HEIGHT as _);

        gl::ClearColor(0f32, 0f32, 0f32, 1f32);
        gl::Clear(gl::COLOR_BUFFER_BIT);

        gl::BlitFramebuffer(
            0,
            0,
            DISPLAY_WIDTH as _,
            DISPLAY_HEIGHT as _,
            PRESENTER_SUB_TOP_SCREEN.x as _,
            PRESENTER_SUB_TOP_SCREEN.y as _,
            PRESENTER_SUB_TOP_SCREEN.width as _,
            (PRESENTER_SUB_TOP_SCREEN.y + PRESENTER_SUB_TOP_SCREEN.height) as _,
            gl::COLOR_BUFFER_BIT,
            gl::NEAREST,
        );

        presenter.gl_swap_window();

        // let mut s = "coords:".to_string();
        // for i in 0..DISPLAY_HEIGHT {
        //     s += &format!(" {:x}", self.shared.read_buf[i * DISPLAY_WIDTH]);
        // }
        // println!("{s}");

        {
            let mut drawing = self.drawing.lock().unwrap();
            *drawing = false;
            // self.drawing_condvar.notify_one();
        }
    }
}