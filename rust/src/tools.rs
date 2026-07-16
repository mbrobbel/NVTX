// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Support for implementing NVTX tools (injection libraries).
//!
//! NVTX instrumentation is a no-op unless a tool attaches to the annotated
//! application. NVTX loads a tool as an *injection library*: a dynamic library
//! whose path is set in the `NVTX_INJECTION64_PATH` (or `NVTX_INJECTION32_PATH`)
//! environment variable. Before the first NVTX call, NVTX loads the library and
//! calls its exported `InitializeInjectionNvtx2` entry point exactly once,
//! passing an accessor for its export tables. The tool then installs its
//! callbacks into per-module function tables; from that point on, every NVTX
//! call in the application invokes the tool's callbacks.
//!
//! This module provides a typed API over that ABI: [`ExportTable`],
//! [`FunctionTable`], and the callback slot markers in [`callback`].
//!
//! # Installing callbacks
//!
//! Callbacks must use the `NVTX_API` calling convention, which is the C
//! convention everywhere except 32-bit x86 Windows, where it is `__stdcall`
//! (declare callbacks `extern "stdcall"` there); [`FunctionTable::set`]
//! type-checks the exact convention against the target's bindings.
//!
//! ```no_run
//! use nvtx::sys::ffi;
//! use nvtx::tools::{callback, Core, ExportTable};
//!
//! #[cfg(all(windows, target_arch = "x86"))]
//! unsafe extern "stdcall" fn on_mark_ascii(message: *const core::ffi::c_char) {
//!     // ...
//! }
//!
//! #[cfg(not(all(windows, target_arch = "x86")))]
//! unsafe extern "C" fn on_mark_ascii(message: *const core::ffi::c_char) {
//!     // ...
//! }
//!
//! fn install(get_export_table: ffi::NvtxGetExportTableFunc_t) -> Option<()> {
//!     // SAFETY: called with the accessor NVTX passed to `InitializeInjectionNvtx2`.
//!     let export = unsafe { ExportTable::new(get_export_table) }?;
//!     let mut core = export.function_table::<Core>().ok()?;
//!     core.set(callback::core::MarkA, Some(on_mark_ascii)).ok()?;
//!     Some(())
//! }
//! ```

mod args;
pub mod callback;
mod export;
mod subscriber;

pub use args::{EventArgs, MessageView, ResourceArgs};
pub use callback::{CallbackId, CallbackModule, Core, Core2, Cuda, CudaRt, OpenCl, Sync};
pub use export::{ExportTable, FunctionTable, SlotError, VersionInfo};
pub use subscriber::{DomainId, RegisteredStringId, ResourceId};

/// Error returned by [`ExportTable::function_table`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AttachError {
    /// The export-table accessor passed by NVTX was null.
    NullGetExportTable,
    /// The callbacks export table was unavailable, too small, or had no
    /// `GetModuleFunctionTable` accessor.
    CallbacksUnavailable,
    /// No function table is available for the requested callback module.
    ModuleTableUnavailable(crate::sys::ffi::NvtxCallbackModule),
}

impl core::fmt::Display for AttachError {
    // `NvtxCallbackModule` only implements `Debug`.
    #[allow(clippy::use_debug)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NullGetExportTable => write!(f, "the NVTX export-table accessor is null"),
            Self::CallbacksUnavailable => {
                write!(f, "the NVTX callbacks export table is unavailable")
            }
            Self::ModuleTableUnavailable(module) => {
                write!(f, "no NVTX function table available for module {module:?}")
            }
        }
    }
}

impl std::error::Error for AttachError {}
