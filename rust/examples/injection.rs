// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! A minimal NVTX injection library that prints observed events to stderr.
//!
//! Build the cdylib and point an NVTX-annotated application at it:
//!
//! ```console
//! cargo build --features tools --example injection
//! NVTX_INJECTION64_PATH=target/debug/examples/libinjection.so \
//!     cargo run --features color-name --example range
//! ```

#![allow(clippy::use_debug)]

use core::ffi::{c_int, CStr};

use nvtx::tools::{DomainId, EventArgs, RangeId, Subscriber};

struct PrintingTool;

impl Subscriber for PrintingTool {
    fn mark_ex(&self, args: &EventArgs<'_>) {
        eprintln!("[injection] mark {args:?}");
    }

    fn mark_ascii(&self, message: &CStr) {
        eprintln!("[injection] mark {message:?}");
    }

    fn range_push_ex(&self, args: &EventArgs<'_>) -> i32 {
        eprintln!("[injection] range push {args:?}");
        nvtx::tools::NO_PUSH_POP_TRACKING
    }

    fn range_pop(&self) -> i32 {
        eprintln!("[injection] range pop");
        nvtx::tools::NO_PUSH_POP_TRACKING
    }

    fn domain_create_ascii(&self, name: &CStr) -> DomainId {
        let id = DomainId::new(nvtx::tools::next_handle());
        eprintln!("[injection] domain create {name:?} -> {id:?}");
        id
    }

    fn domain_mark_ex(&self, domain: DomainId, args: &EventArgs<'_>) {
        eprintln!("[injection] domain {domain:?} mark {args:?}");
    }

    fn domain_range_start_ex(&self, domain: DomainId, args: &EventArgs<'_>) -> RangeId {
        let id = nvtx::tools::next_handle();
        eprintln!("[injection] domain {domain:?} range start {id} {args:?}");
        id
    }

    fn domain_range_end(&self, domain: DomainId, id: RangeId) {
        eprintln!("[injection] domain {domain:?} range end {id}");
    }

    fn domain_range_push_ex(&self, domain: DomainId, args: &EventArgs<'_>) -> i32 {
        eprintln!("[injection] domain {domain:?} range push {args:?}");
        nvtx::tools::NO_PUSH_POP_TRACKING
    }

    fn domain_range_pop(&self, domain: DomainId) -> i32 {
        eprintln!("[injection] domain {domain:?} range pop");
        nvtx::tools::NO_PUSH_POP_TRACKING
    }
}

/// NVTX injection entry point, called once by NVTX from the application.
/// `extern "system"` matches the `NVTX_API` calling convention on every
/// platform (`__stdcall` on 32-bit x86 Windows, the C convention elsewhere).
#[no_mangle]
pub extern "system" fn InitializeInjectionNvtx2(
    get_export_table: nvtx::sys::ffi::NvtxGetExportTableFunc_t,
) -> c_int {
    // SAFETY: NVTX passes its export-table accessor to this entry point.
    match unsafe { nvtx::tools::attach(get_export_table, PrintingTool) } {
        Ok(()) => 1,
        Err(error) => {
            eprintln!("[injection] failed to attach: {error}");
            0
        }
    }
}
