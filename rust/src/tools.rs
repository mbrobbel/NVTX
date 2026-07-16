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
//! This module provides two layers on top of that ABI:
//!
//! * A high-level [`Subscriber`] trait with a default no-op implementation for
//!   every callback, installed via [`attach`]. The crate provides the
//!   `extern "C"` trampolines, argument decoding, and panic containment.
//! * A low-level typed API ([`ExportTable`], [`FunctionTable`], [`callback`])
//!   for tools that install raw `extern "C"` callbacks themselves.
//!
//! # Implementing a tool with [`Subscriber`]
//!
//! Compile a library crate with `crate-type = ["cdylib"]`, implement
//! [`Subscriber`], and export the entry point:
//!
//! ```no_run
//! use core::ffi::{c_int, CStr};
//!
//! struct PrintingTool;
//!
//! impl nvtx::tools::Subscriber for PrintingTool {
//!     fn mark_ascii(&self, message: &CStr) {
//!         eprintln!("mark: {}", message.to_string_lossy());
//!     }
//! }
//!
//! /// NVTX injection entry point, called once by NVTX from the application.
//! /// `extern "system"` matches the `NVTX_API` calling convention on every
//! /// platform (`__stdcall` on 32-bit x86 Windows, the C convention elsewhere).
//! #[no_mangle]
//! pub extern "system" fn InitializeInjectionNvtx2(
//!     get_export_table: nvtx::sys::ffi::NvtxGetExportTableFunc_t,
//! ) -> c_int {
//!     // SAFETY: NVTX passes its export-table accessor to this entry point.
//!     match unsafe { nvtx::tools::attach(get_export_table, PrintingTool) } {
//!         Ok(()) => 1,
//!         Err(_) => 0,
//!     }
//! }
//! ```
//!
//! Run the application with the environment variable pointing at the built
//! library, for example:
//!
//! ```console
//! NVTX_INJECTION64_PATH=/path/to/libtool.so ./app
//! ```
//!
//! # Borrowed arguments
//!
//! Callback arguments ([`EventArgs`], [`ResourceArgs`], `&CStr`, and
//! [`widestring::WideCStr`] references) borrow application memory that is only
//! valid for the duration of the callback. Copy whatever must be retained.
//!
//! # Panics and errors
//!
//! A panic must never unwind across the C ABI into the application. Trampolines
//! catch panics from [`Subscriber`] methods and return a fallback value (`0`,
//! null, or [`NO_PUSH_POP_TRACKING`]). This containment requires the default
//! `panic = "unwind"` strategy; with `panic = "abort"`, a panicking subscriber
//! aborts the application.
//!
//! # Using the low-level layer
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

use core::sync::atomic::{AtomicU64, Ordering};

mod args;
pub mod callback;
mod export;
mod subscriber;
mod trampoline;

pub use args::{EventArgs, MessageView, ResourceArgs};
pub use callback::{CallbackId, CallbackModule, Core, Core2, Cuda, CudaRt, OpenCl, Sync};
pub use export::{ExportTable, FunctionTable, SlotError, VersionInfo};
pub use subscriber::{DomainId, RegisteredStringId, ResourceId, Subscriber};

/// A process-visible range identifier, correlating range start and end.
pub use crate::sys::RangeId;

/// Return value for range push and pop callbacks indicating that the tool does
/// not track push/pop depth.
///
/// This is `NVTX_NO_PUSH_POP_TRACKING` from `nvToolsExt.h`, defined manually
/// because the C definition is a function-like macro.
pub const NO_PUSH_POP_TRACKING: i32 = -2;

/// Return a fresh process-unique nonzero handle value.
///
/// This monotonic counter backs the default implementations of the
/// [`Subscriber`] methods that return handles or identifiers. It starts at `1`;
/// `0` is reserved for null handles and the default domain.
#[must_use]
pub fn next_handle() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Error returned by [`attach`] and [`ExportTable::function_table`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AttachError {
    /// A subscriber is already attached; attaching is one-shot per process.
    AlreadyAttached,
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
            Self::AlreadyAttached => write!(f, "an NVTX subscriber is already attached"),
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

/// Attach `subscriber` as the process-global NVTX subscriber.
///
/// Installs the crate's trampolines into the CORE and CORE2 function tables so
/// every NVTX call in the application is forwarded to `subscriber`. Attaching
/// is one-shot per process, matching NVTX's single `InitializeInjectionNvtx2`
/// call. Slots unavailable in the application's NVTX version (out-of-bounds or
/// null) are skipped best-effort. The tool's NVTX version is reported through
/// the version-info export table when available, as the injection ABI
/// requires.
///
/// Call this from the tool's exported `InitializeInjectionNvtx2` entry point
/// and return `1` on `Ok` and `0` on `Err` (see the [module docs](self) for a
/// complete example).
///
/// # Errors
///
/// Returns [`AttachError`] when the accessor is null, when the export tables
/// are unavailable, or when a subscriber is already attached. On error nothing
/// is installed and no subscriber is stored, so attaching may be retried.
///
/// # Safety
///
/// `get_export_table` must be the accessor NVTX passed to
/// `InitializeInjectionNvtx2`, or a function upholding its contract: any
/// non-null pointer returned for a supported export-table identifier must point
/// to the corresponding valid export table which lives for the rest of the
/// process.
pub unsafe fn attach(
    get_export_table: crate::sys::ffi::NvtxGetExportTableFunc_t,
    subscriber: impl Subscriber + 'static,
) -> Result<(), AttachError> {
    // SAFETY: The caller upholds `ExportTable::new`'s contract.
    let export = unsafe { ExportTable::new(get_export_table) };
    let export = export.ok_or(AttachError::NullGetExportTable)?;
    // Fetch both tables before committing the subscriber so a failed attach
    // stores nothing and can be retried.
    let global_table = export.function_table::<Core>()?;
    let domain_table = export.function_table::<Core2>()?;
    // Store the subscriber before installing any trampoline so a callback can
    // never fire without a subscriber in place.
    trampoline::set_subscriber(Box::new(subscriber))?;
    trampoline::install(global_table, domain_table);
    // Report the tool's NVTX version, as the injection ABI requires.
    if let Some(version_info) = export.version_info() {
        // CAST: `NVTX_VERSION` is a small positive constant.
        version_info.set_injection_nvtx_version(crate::sys::ffi::NVTX_VERSION as u32);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::next_handle;

    #[test]
    fn next_handle_is_nonzero_and_monotonic() {
        let first = next_handle();
        let second = next_handle();
        assert!(first >= 1);
        assert!(second > first);
    }
}
