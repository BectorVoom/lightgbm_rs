use cubecl_common::backtrace::BackTrace;
use cubecl_cpp::formatter::format_cpp;
use cubecl_cpp::{cuda::arch::CudaArchitecture, shared::CompilationOptions};
use cubecl_runtime::{
    compiler::CompilationError,
    validation::{validate_cube_dim, validate_units},
};

use super::storage::gpu::GpuResource;
use crate::{CudaCompiler, compute::stream::Stream};
use crate::{
    CudaComputeKernel,
    install::{cccl_include_path, include_path},
};
use cubecl_core::{
    compilation_cache::CompilationCache,
    hash::StableHash,
    server::ResourceLimitError,
    {ir::DeviceProperties, prelude::*},
};
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use cubecl_runtime::{compiler::CubeTask, logging::ServerLogger};
use cudarc::driver::DriverError;
use cudarc::driver::sys::CUfunc_st;
use cudarc::driver::sys::{CUctx_st, CUfunction_attribute, CUtensorMap};
use std::collections::HashMap;
use std::ffi::CString;
use std::ffi::c_char;
use std::str::FromStr;
use std::sync::Arc;
use std::{ffi::CStr, os::raw::c_void};

use cubecl_common::cache::CacheOption;

/// LIGHTGBM_RS FORK: cheap two-level module-cache key (§12 round 3). The full
/// `KernelId` hash costs ~86µs/launch on the bench profile (the comptime `Info`
/// payload), so the hot resolve buckets by (kernel type-name ptr+len, execution
/// mode, cube dim) — all `Copy`, hashed in nanoseconds — and disambiguates inside
/// the bucket with the full `KernelId` EQUALITY (field compares, no hashing). A
/// name-pointer collision across types is impossible (`&'static str` per
/// monomorphization); a same-type pointer SPLIT (theoretical, multi-codegen-unit)
/// only causes duplicate buckets — correctness is carried by the full-id equality.
pub(crate) type FastKernelKey = (usize, usize, ExecutionMode, u32, u32, u32);

#[derive(Debug)]
pub(crate) struct CudaContext {
    pub context: *mut CUctx_st,
    pub module_names: HashMap<KernelId, CompiledKernel>,
    /// LIGHTGBM_RS FORK: two-level fast resolve (see [`FastKernelKey`]).
    pub fast_modules: HashMap<FastKernelKey, Vec<(KernelId, ResolvedKernel)>>,
    ptx_cache: Option<CompilationCache<StableHash, PtxCacheEntry>>,
    pub timestamps: TimestampProfiler,
    pub arch: CudaArchitecture,
    pub compilation_options: CompilationOptions,
    pub properties: DeviceProperties,
}

#[derive(Debug)]
pub struct CompiledKernel {
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
    func: *mut CUfunc_st,
}

/// LIGHTGBM_RS FORK: the `Copy` launch data of a compiled kernel, resolved with a
/// single `module_names` lookup (see [`CudaContext::resolve_kernel`]).
#[derive(Clone, Copy, Debug)]
pub struct ResolvedKernel {
    pub(crate) func: *mut CUfunc_st,
    pub(crate) cube_dim: CubeDim,
    pub(crate) shared_mem_bytes: usize,
}

/// LIGHTGBM_RS FORK: restore upstream's per-launch `cuFuncSetAttribute` call
/// (`CUBECL_CUDA_FUNCATTR_EVERY=1`); default OFF — the attribute is set once at
/// module load.
fn funcattr_every() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CUBECL_CUDA_FUNCATTR_EVERY").is_ok_and(|v| v == "1")
    })
}

#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub struct PtxCacheEntry {
    entrypoint_name: String,
    shared_mem_bytes: usize,
    ptx: Vec<std::ffi::c_char>,
}

impl CudaContext {
    pub fn new(
        compilation_options: CompilationOptions,
        properties: DeviceProperties,
        context: *mut CUctx_st,
        arch: CudaArchitecture,
    ) -> Self {
        Self {
            context,
            module_names: HashMap::new(),
            fast_modules: HashMap::new(),
            ptx_cache: {
                use cubecl_runtime::config::RuntimeConfig;
                let config = cubecl_runtime::config::CubeClRuntimeConfig::get();
                if let Some(cache) = &config.compilation.cache {
                    let root = cache.root();
                    Some(CompilationCache::new(
                        "ptx",
                        CacheOption::default().name("cuda").root(root),
                    ))
                } else if std::env::var("CUBECL_CUDA_PTX_CACHE").map(|v| v != "0").unwrap_or(true) {
                    // LIGHTGBM_RS FORK (§12 round 3d): upstream's default is NO ptx cache
                    // (`CompilationConfig::default().cache == None`), so every fresh
                    // process NVRTC-compiles every kernel INSIDE the first train (~1.7s
                    // per process on the P100 bench — the cost round 3c unmasked).
                    // Default the cache ON at the global root (persists across
                    // processes); a user cubecl.toml still takes precedence above, and
                    // `CUBECL_CUDA_PTX_CACHE=0` restores upstream behavior.
                    let root = cubecl_runtime::config::cache::CacheConfig::Global.root();
                    Some(CompilationCache::new(
                        "ptx",
                        CacheOption::default().name("cuda").root(root),
                    ))
                } else {
                    None
                }
            },
            arch,
            timestamps: TimestampProfiler::default(),
            compilation_options,
            properties,
        }
    }

    /// Switches the current CUDA context to this context.
    pub fn unsafe_set_current(&self) -> Result<(), DriverError> {
        // SAFETY: `self.context` is a valid CUDA context obtained from `primary_ctx::retain`
        // during server initialization and remains valid for the server's lifetime.
        unsafe { cudarc::driver::result::ctx::set_current(self.context) }
    }

    pub fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        kernel: Box<dyn CubeTask<CudaCompiler>>,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        if crate::compute::arena::prof_enabled() {
            crate::compute::arena::COMPILE_COUNT
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        }
        let hash = if let Some(cache) = &self.ptx_cache {
            let hash = kernel_id.stable_hash();

            if let Some(entry) = cache.get(&hash) {
                log::trace!("Using PTX cache");
                if crate::compute::arena::prof_enabled() {
                    crate::compute::arena::PTXCACHE_HIT_COUNT
                        .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                }

                self.load_ptx(
                    entry.ptx.clone(),
                    kernel_id.clone(),
                    entry.entrypoint_name.clone(),
                    kernel_id.cube_dim,
                    entry.shared_mem_bytes,
                )?;
                return Ok(());
            }
            Some(hash)
        } else {
            None
        };

        log::trace!("Compiling kernel");
        let prof_nvrtc = crate::compute::arena::prof_enabled().then(std::time::Instant::now);

        validate_cube_dim(&self.properties, kernel_id)?;
        validate_units(&self.properties, kernel_id)?;

        let mut kernel_compiled = kernel.compile(
            &mut Default::default(),
            &self.compilation_options,
            mode,
            kernel.address_type(),
        )?;

        self.validate_shared(&kernel_compiled.repr)?;

        if logger.compilation_activated() {
            kernel_compiled.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&kernel_compiled.source) {
                kernel_compiled.source = formatted;
            }
        }

        let cube_dim = kernel_compiled.cube_dim;
        let arch = if self.arch.version >= 90 {
            format!("--gpu-architecture=sm_{}a", self.arch)
        } else {
            format!("--gpu-architecture=sm_{}", self.arch)
        };

        let include_path = include_path();
        let include_option = format!("--include-path={}", include_path.to_str().unwrap());
        let cccl_include_path = cccl_include_path();
        let cccl_include_option = format!("--include-path={}", cccl_include_path.to_str().unwrap());
        let mut options = vec![arch.as_str(), include_option.as_str(), "-lineinfo"];
        if cccl_include_path.exists() {
            options.push(&cccl_include_option);
        }

        logger.log_compilation(&kernel_compiled);

        // SAFETY: Calling NVRTC FFI to create, compile, and extract PTX from a program.
        // The `CString` source is null-terminated and outlives the program. On compilation
        // failure, the error log is retrieved and reported before returning.
        let ptx = unsafe {
            // I'd like to set the name to the kernel name, but keep getting UTF-8 errors so let's
            // leave it `None` for now
            let source = CString::from_str(&kernel_compiled.source).unwrap();
            let program =
                cudarc::nvrtc::result::create_program(source.as_c_str(), None).map_err(|err| {
                    CompilationError::Generic {
                        reason: format!("{err:?}"),
                        backtrace: BackTrace::capture(),
                    }
                })?;
            if cudarc::nvrtc::result::compile_program(program, &options).is_err() {
                let log_raw = cudarc::nvrtc::result::get_program_log(program).map_err(|err| {
                    CompilationError::Generic {
                        reason: format!("{err:?}"),
                        backtrace: BackTrace::capture(),
                    }
                })?;

                let log_ptr = log_raw.as_ptr();
                let log = CStr::from_ptr(log_ptr).to_str().unwrap();
                let mut message = "[Compilation Error] ".to_string();
                for line in log.split('\n') {
                    if !line.is_empty() {
                        message += format!("\n    {line}").as_str();
                    }
                }
                let source = kernel
                    .compile(
                        &mut Default::default(),
                        &self.compilation_options,
                        mode,
                        kernel.address_type(),
                    )?
                    .source;
                Err(CompilationError::Generic {
                    reason: format!("{message}\n[Source]  \n{source}"),
                    backtrace: BackTrace::capture(),
                })?;
            };
            cudarc::nvrtc::result::get_ptx(program).map_err(|err| CompilationError::Generic {
                reason: format!("{err:?}"),
                backtrace: BackTrace::capture(),
            })?
        };

        let repr = kernel_compiled.repr.unwrap();

        if let Some(cache) = &mut self.ptx_cache {
            let result = cache.insert(
                hash.unwrap(),
                PtxCacheEntry {
                    entrypoint_name: kernel_compiled.entrypoint_name.clone(),
                    shared_mem_bytes: repr.shared_memory_size(),
                    ptx: ptx.clone(),
                },
            );
            if let Err(err) = result {
                log::warn!("Unable to save the ptx {err:?}");
            }
        }

        if let Some(start) = prof_nvrtc {
            crate::compute::arena::NVRTC_NS
                .fetch_add(start.elapsed().as_nanos() as u64, core::sync::atomic::Ordering::Relaxed);
        }
        self.load_ptx(
            ptx,
            kernel_id.clone(),
            kernel_compiled.entrypoint_name,
            cube_dim,
            repr.shared_memory_size(),
        )?;
        Ok(())
    }

    fn load_ptx(
        &mut self,
        ptx: Vec<c_char>,
        kernel_id: KernelId,
        entrypoint_name: String,
        cube_dim: CubeDim,
        shared_mem_bytes: usize,
    ) -> Result<(), CompilationError> {
        let prof_load = crate::compute::arena::prof_enabled().then(std::time::Instant::now);
        let func_name = CString::new(entrypoint_name).unwrap();
        // SAFETY: `ptx` is a valid null-terminated PTX binary from NVRTC. `func_name` is a
        // null-terminated `CString` matching the kernel entry point in the compiled module.
        let func = unsafe {
            let module = cudarc::driver::result::module::load_data(ptx.as_ptr() as *const _)
                .map_err(|err| CompilationError::Generic {
                    reason: format!("Unable to load the PTX: {err:?}"),
                    backtrace: BackTrace::capture(),
                })?;

            cudarc::driver::result::module::get_function(module, func_name).map_err(|err| {
                CompilationError::Generic {
                    reason: format!("Unable to fetch the function from the module: {err:?}"),
                    backtrace: BackTrace::capture(),
                }
            })?
        };

        // LIGHTGBM_RS FORK: the max-dynamic-shared-memory attribute is a property of the
        // function, not of a launch — set it ONCE here instead of on every
        // `cuFuncSetAttribute` in `execute_task` (upstream re-set it per launch, one
        // extra driver call × ~18.5k launches/train). `CUBECL_CUDA_FUNCATTR_EVERY=1`
        // restores the per-launch call (execute_task also sets it then).
        // SAFETY: `func` is the valid function handle just loaded above.
        unsafe {
            cudarc::driver::result::function::set_function_attribute(
                func,
                cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_mem_bytes as i32,
            )
            .map_err(|err| CompilationError::Generic {
                reason: format!("Unable to set the shared-memory attribute: {err:?}"),
                backtrace: BackTrace::capture(),
            })?;
        }

        self.module_names.insert(
            kernel_id.clone(),
            CompiledKernel {
                cube_dim,
                shared_mem_bytes,
                func,
            },
        );
        if let Some(start) = prof_load {
            crate::compute::arena::MODLOAD_NS
                .fetch_add(start.elapsed().as_nanos() as u64, core::sync::atomic::Ordering::Relaxed);
        }

        Ok(())
    }

    /// LIGHTGBM_RS FORK: resolve a compiled kernel's launch data with ONE map lookup.
    /// Upstream hashed the full `KernelId` twice per launch (`contains_key` in
    /// `Command::kernel`, then `get` in `execute_task`); callers now resolve once and
    /// pass the `Copy` data through.
    pub fn resolve_kernel(&self, kernel_id: &KernelId) -> Option<ResolvedKernel> {
        self.module_names.get(kernel_id).map(|k| ResolvedKernel {
            func: k.func,
            cube_dim: k.cube_dim,
            shared_mem_bytes: k.shared_mem_bytes,
        })
    }

    pub fn execute_task(
        &mut self,
        stream: &mut Stream,
        kernel: ResolvedKernel,
        dispatch_count: (u32, u32, u32),
        tensor_maps: &[CUtensorMap],
        resources: &[GpuResource],
        const_info: Option<*mut c_void>,
    ) -> Result<(), LaunchError> {
        // LIGHTGBM_RS FORK: sub-segment timers for the CP3→CP4 teardown (§12 lever 2).
        let prof = crate::compute::arena::prof_enabled();
        let t0 = prof.then(std::time::Instant::now);
        let mut bindings = tensor_maps
            .iter()
            .map(|map| map as *const _ as *mut c_void)
            .collect::<Vec<_>>();
        bindings.extend(resources.iter().map(|memory| memory.binding));
        bindings.extend(const_info);

        let cube_dim = kernel.cube_dim;
        let t1 = prof.then(std::time::Instant::now);
        // SAFETY: `kernel.func` is a valid function handle from a loaded module.
        // `stream.sys` is a valid CUDA stream. `bindings` contains valid device pointers
        // for all kernel arguments. The dispatch and cube dimensions are validated by
        // the caller.
        unsafe {
            // Upstream set the shared-memory attribute on EVERY launch; the fork sets it
            // once at module load (`load_ptx`). `CUBECL_CUDA_FUNCATTR_EVERY=1` restores
            // the per-launch call for A/B.
            if funcattr_every() {
                cudarc::driver::result::function::set_function_attribute(
                    kernel.func,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    kernel.shared_mem_bytes as i32,
                )
                .map_err(|err| LaunchError::Unknown {
                    reason: format!("{err:?}"),
                    backtrace: BackTrace::capture(),
                })?;
            }
            let t2 = prof.then(std::time::Instant::now);
            cudarc::driver::result::launch_kernel(
                kernel.func,
                dispatch_count,
                (cube_dim.x, cube_dim.y, cube_dim.z),
                // Shared memory is collected into a single buffer, with each shared memory being
                // an offset pointer
                kernel.shared_mem_bytes as u32,
                stream.sys,
                &mut bindings,
            )
            .map_err(|err| LaunchError::Unknown {
                reason: format!("{err:?}"),
                backtrace: BackTrace::capture(),
            })?;
            if let (Some(t0), Some(t1), Some(t2)) = (t0, t1, t2) {
                use core::sync::atomic::Ordering;
                let now = std::time::Instant::now();
                crate::compute::arena::KMARSHAL_NS
                    .fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
                crate::compute::arena::KATTR_NS
                    .fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
                crate::compute::arena::KLAUNCH_NS
                    .fetch_add((now - t2).as_nanos() as u64, Ordering::Relaxed);
            }
        };

        Ok(())
    }

    fn validate_shared(&self, repr: &Option<CudaComputeKernel>) -> Result<(), LaunchError> {
        let requested = repr.as_ref().map(|repr| repr.shared_memory_size());
        let max = self.properties.hardware.max_shared_memory_size;
        if let Some(requested) = requested
            && requested > max
        {
            Err(ResourceLimitError::SharedMemory {
                requested,
                max,
                backtrace: BackTrace::capture(),
            }
            .into())
        } else {
            Ok(())
        }
    }
}
