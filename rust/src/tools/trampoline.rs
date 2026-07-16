// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! The `extern "C"` callbacks installed by [`attach`](crate::tools::attach),
//! forwarding every NVTX call to the process-global [`Subscriber`].
//!
//! Every trampoline decodes its raw arguments and dispatches through
//! [`with_subscriber`], which contains panics so they never unwind into the
//! application: a panicking subscriber method yields the fallback return value
//! (`()`/`0`/null/[`NO_PUSH_POP_TRACKING`]).

use core::ffi::{c_char, c_int, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::OnceLock;

use crate::sys::ffi;
use crate::tools::args::{opt_cstr, opt_wide};
use crate::tools::callback::core as cb_core;
use crate::tools::callback::core2 as cb_core2;
use crate::tools::{
    AttachError, Core, Core2, DomainId, EventArgs, FunctionTable, RegisteredStringId, ResourceArgs,
    ResourceId, Subscriber, NO_PUSH_POP_TRACKING,
};

/// The process-global subscriber, set once by
/// [`attach`](crate::tools::attach).
static SUBSCRIBER: OnceLock<Box<dyn Subscriber>> = OnceLock::new();

/// Store the process-global subscriber.
///
/// # Errors
///
/// Returns [`AttachError::AlreadyAttached`] when a subscriber was stored
/// before.
pub(super) fn set_subscriber(subscriber: Box<dyn Subscriber>) -> Result<(), AttachError> {
    SUBSCRIBER
        .set(subscriber)
        .map_err(|_| AttachError::AlreadyAttached)
}

/// Dispatch to the process-global subscriber, returning `fallback` when no
/// subscriber is attached or when the subscriber panics.
fn with_subscriber<R>(fallback: R, f: impl FnOnce(&dyn Subscriber) -> R) -> R {
    let Some(subscriber) = SUBSCRIBER.get() else {
        return fallback;
    };
    catch_unwind(AssertUnwindSafe(|| f(subscriber.as_ref()))).unwrap_or(fallback)
}

fn domain_id(domain: ffi::nvtxDomainHandle_t) -> DomainId {
    // CAST: domain handles are opaque; capture the pointer's address bits.
    DomainId::new(domain as usize as u64)
}

/// Return `raw` to NVTX as an opaque pointer-typed handle.
///
/// Identifier values must fit the target's pointer width; out-of-range values
/// (only possible on targets with pointers narrower than 64 bits) yield a null
/// handle instead of a silently truncated one that later callbacks could not
/// correlate. This path must stay panic-free: it runs in an `extern "C"`
/// frame outside the [`with_subscriber`] panic guard.
fn opaque_handle<T>(raw: u64) -> *mut T {
    usize::try_from(raw).map_or(core::ptr::null_mut(), |address| address as *mut T)
}

fn domain_handle(id: DomainId) -> ffi::nvtxDomainHandle_t {
    opaque_handle(id.raw())
}

fn string_handle(id: RegisteredStringId) -> ffi::nvtxStringHandle_t {
    opaque_handle(id.raw())
}

fn resource_handle(id: ResourceId) -> ffi::nvtxResourceHandle_t {
    opaque_handle(id.raw())
}

fn resource_id(resource: ffi::nvtxResourceHandle_t) -> ResourceId {
    // CAST: resource handles are opaque; capture the pointer's address bits.
    ResourceId::new(resource as usize as u64)
}

/// Install all CORE and CORE2 trampolines into the fetched function tables.
///
/// Slots rejected by the application's NVTX version (out of bounds or null)
/// are skipped best-effort.
pub(super) fn install(
    mut global_table: FunctionTable<Core>,
    mut domain_table: FunctionTable<Core2>,
) {
    let _ = global_table.set(cb_core::MarkEx, Some(mark_ex));
    let _ = global_table.set(cb_core::MarkA, Some(mark_a));
    let _ = global_table.set(cb_core::MarkW, Some(mark_w));
    let _ = global_table.set(cb_core::RangeStartEx, Some(range_start_ex));
    let _ = global_table.set(cb_core::RangeStartA, Some(range_start_a));
    let _ = global_table.set(cb_core::RangeStartW, Some(range_start_w));
    let _ = global_table.set(cb_core::RangeEnd, Some(range_end));
    let _ = global_table.set(cb_core::RangePushEx, Some(range_push_ex));
    let _ = global_table.set(cb_core::RangePushA, Some(range_push_a));
    let _ = global_table.set(cb_core::RangePushW, Some(range_push_w));
    let _ = global_table.set(cb_core::RangePop, Some(range_pop));
    let _ = global_table.set(cb_core::NameCategoryA, Some(name_category_a));
    let _ = global_table.set(cb_core::NameCategoryW, Some(name_category_w));
    let _ = global_table.set(cb_core::NameOsThreadA, Some(name_os_thread_a));
    let _ = global_table.set(cb_core::NameOsThreadW, Some(name_os_thread_w));

    let _ = domain_table.set(cb_core2::DomainMarkEx, Some(domain_mark_ex));
    let _ = domain_table.set(cb_core2::DomainRangeStartEx, Some(domain_range_start_ex));
    let _ = domain_table.set(cb_core2::DomainRangeEnd, Some(domain_range_end));
    let _ = domain_table.set(cb_core2::DomainRangePushEx, Some(domain_range_push_ex));
    let _ = domain_table.set(cb_core2::DomainRangePop, Some(domain_range_pop));
    let _ = domain_table.set(cb_core2::DomainResourceCreate, Some(domain_resource_create));
    let _ = domain_table.set(
        cb_core2::DomainResourceDestroy,
        Some(domain_resource_destroy),
    );
    let _ = domain_table.set(cb_core2::DomainNameCategoryA, Some(domain_name_category_a));
    let _ = domain_table.set(cb_core2::DomainNameCategoryW, Some(domain_name_category_w));
    let _ = domain_table.set(
        cb_core2::DomainRegisterStringA,
        Some(domain_register_string_a),
    );
    let _ = domain_table.set(
        cb_core2::DomainRegisterStringW,
        Some(domain_register_string_w),
    );
    let _ = domain_table.set(cb_core2::DomainCreateA, Some(domain_create_a));
    let _ = domain_table.set(cb_core2::DomainCreateW, Some(domain_create_w));
    let _ = domain_table.set(cb_core2::DomainDestroy, Some(domain_destroy));
    let _ = domain_table.set(cb_core2::Initialize, Some(initialize));
}

// NVTX callbacks use the `NVTX_API` calling convention: `__stdcall` on 32-bit
// x86 Windows and the C convention everywhere else. This macro emits each
// trampoline with the calling convention bindgen generates for the target, so
// `FunctionTable::set` type-checks the exact ABI.
macro_rules! nvtx_api_callbacks {
    ($(
        unsafe fn $name:ident($($param:ident: $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block
    )+) => {
        $(
            #[cfg(all(windows, target_arch = "x86"))]
            unsafe extern "stdcall" fn $name($($param: $ty),*) $(-> $ret)? $body

            #[cfg(not(all(windows, target_arch = "x86")))]
            unsafe extern "C" fn $name($($param: $ty),*) $(-> $ret)? $body
        )+
    };
}

// The SAFETY comments below rely on the NVTX callback contract: pointer
// arguments are either null or valid for the duration of the call, string
// pointers are NUL-terminated, and attributes structs are readable up to
// their declared `size`.

nvtx_api_callbacks! {
    unsafe fn mark_ex(attrib: *const ffi::nvtxEventAttributes_t) {
        // SAFETY: NVTX callback contract (see above).
        let args = unsafe { EventArgs::decode(attrib) };
        with_subscriber((), |s| s.mark_ex(&args));
    }

    unsafe fn mark_a(message: *const c_char) {
        // SAFETY: NVTX callback contract (see above).
        let Some(message) = (unsafe { opt_cstr(message) }) else {
            return;
        };
        with_subscriber((), |s| s.mark_ascii(message));
    }

    unsafe fn mark_w(message: *const ffi::wchar_t) {
        // SAFETY: NVTX callback contract (see above).
        let Some(message) = (unsafe { opt_wide(message) }) else {
            return;
        };
        with_subscriber((), |s| s.mark_unicode(message));
    }

    unsafe fn range_start_ex(
        attrib: *const ffi::nvtxEventAttributes_t,
    ) -> ffi::nvtxRangeId_t {
        // SAFETY: NVTX callback contract (see above).
        let args = unsafe { EventArgs::decode(attrib) };
        with_subscriber(0, |s| s.range_start_ex(&args))
    }

    unsafe fn range_start_a(message: *const c_char) -> ffi::nvtxRangeId_t {
        // SAFETY: NVTX callback contract (see above).
        let Some(message) = (unsafe { opt_cstr(message) }) else {
            return 0;
        };
        with_subscriber(0, |s| s.range_start_ascii(message))
    }

    unsafe fn range_start_w(message: *const ffi::wchar_t) -> ffi::nvtxRangeId_t {
        // SAFETY: NVTX callback contract (see above).
        let Some(message) = (unsafe { opt_wide(message) }) else {
            return 0;
        };
        with_subscriber(0, |s| s.range_start_unicode(message))
    }

    unsafe fn range_end(id: ffi::nvtxRangeId_t) {
        with_subscriber((), |s| s.range_end(id));
    }

    unsafe fn range_push_ex(attrib: *const ffi::nvtxEventAttributes_t) -> c_int {
        // SAFETY: NVTX callback contract (see above).
        let args = unsafe { EventArgs::decode(attrib) };
        with_subscriber(NO_PUSH_POP_TRACKING, |s| s.range_push_ex(&args))
    }

    unsafe fn range_push_a(message: *const c_char) -> c_int {
        // SAFETY: NVTX callback contract (see above).
        let Some(message) = (unsafe { opt_cstr(message) }) else {
            return NO_PUSH_POP_TRACKING;
        };
        with_subscriber(NO_PUSH_POP_TRACKING, |s| s.range_push_ascii(message))
    }

    unsafe fn range_push_w(message: *const ffi::wchar_t) -> c_int {
        // SAFETY: NVTX callback contract (see above).
        let Some(message) = (unsafe { opt_wide(message) }) else {
            return NO_PUSH_POP_TRACKING;
        };
        with_subscriber(NO_PUSH_POP_TRACKING, |s| s.range_push_unicode(message))
    }

    unsafe fn range_pop() -> c_int {
        with_subscriber(NO_PUSH_POP_TRACKING, |s| s.range_pop())
    }

    unsafe fn name_category_a(category: u32, name: *const c_char) {
        // SAFETY: NVTX callback contract (see above).
        let Some(name) = (unsafe { opt_cstr(name) }) else {
            return;
        };
        with_subscriber((), |s| s.name_category_ascii(category, name));
    }

    unsafe fn name_category_w(category: u32, name: *const ffi::wchar_t) {
        // SAFETY: NVTX callback contract (see above).
        let Some(name) = (unsafe { opt_wide(name) }) else {
            return;
        };
        with_subscriber((), |s| s.name_category_unicode(category, name));
    }

    unsafe fn name_os_thread_a(thread_id: u32, name: *const c_char) {
        // SAFETY: NVTX callback contract (see above).
        let Some(name) = (unsafe { opt_cstr(name) }) else {
            return;
        };
        with_subscriber((), |s| s.name_os_thread_ascii(thread_id, name));
    }

    unsafe fn name_os_thread_w(thread_id: u32, name: *const ffi::wchar_t) {
        // SAFETY: NVTX callback contract (see above).
        let Some(name) = (unsafe { opt_wide(name) }) else {
            return;
        };
        with_subscriber((), |s| s.name_os_thread_unicode(thread_id, name));
    }

    unsafe fn domain_mark_ex(
        domain: ffi::nvtxDomainHandle_t,
        attrib: *const ffi::nvtxEventAttributes_t,
    ) {
        // SAFETY: NVTX callback contract (see above).
        let args = unsafe { EventArgs::decode(attrib) };
        with_subscriber((), |s| s.domain_mark_ex(domain_id(domain), &args));
    }

    unsafe fn domain_range_start_ex(
        domain: ffi::nvtxDomainHandle_t,
        attrib: *const ffi::nvtxEventAttributes_t,
    ) -> ffi::nvtxRangeId_t {
        // SAFETY: NVTX callback contract (see above).
        let args = unsafe { EventArgs::decode(attrib) };
        with_subscriber(0, |s| s.domain_range_start_ex(domain_id(domain), &args))
    }

    unsafe fn domain_range_end(domain: ffi::nvtxDomainHandle_t, id: ffi::nvtxRangeId_t) {
        with_subscriber((), |s| s.domain_range_end(domain_id(domain), id));
    }

    unsafe fn domain_range_push_ex(
        domain: ffi::nvtxDomainHandle_t,
        attrib: *const ffi::nvtxEventAttributes_t,
    ) -> c_int {
        // SAFETY: NVTX callback contract (see above).
        let args = unsafe { EventArgs::decode(attrib) };
        with_subscriber(NO_PUSH_POP_TRACKING, |s| {
            s.domain_range_push_ex(domain_id(domain), &args)
        })
    }

    unsafe fn domain_range_pop(domain: ffi::nvtxDomainHandle_t) -> c_int {
        with_subscriber(NO_PUSH_POP_TRACKING, |s| {
            s.domain_range_pop(domain_id(domain))
        })
    }

    unsafe fn domain_resource_create(
        domain: ffi::nvtxDomainHandle_t,
        attrib: *mut ffi::nvtxResourceAttributes_t,
    ) -> ffi::nvtxResourceHandle_t {
        // SAFETY: NVTX callback contract (see above).
        let args = unsafe { ResourceArgs::decode(attrib.cast_const()) };
        let id = with_subscriber(ResourceId::new(0), |s| {
            s.domain_resource_create(domain_id(domain), &args)
        });
        resource_handle(id)
    }

    unsafe fn domain_resource_destroy(resource: ffi::nvtxResourceHandle_t) {
        with_subscriber((), |s| s.domain_resource_destroy(resource_id(resource)));
    }

    unsafe fn domain_name_category_a(
        domain: ffi::nvtxDomainHandle_t,
        category: u32,
        name: *const c_char,
    ) {
        // SAFETY: NVTX callback contract (see above).
        let Some(name) = (unsafe { opt_cstr(name) }) else {
            return;
        };
        with_subscriber((), |s| {
            s.domain_name_category_ascii(domain_id(domain), category, name);
        });
    }

    unsafe fn domain_name_category_w(
        domain: ffi::nvtxDomainHandle_t,
        category: u32,
        name: *const ffi::wchar_t,
    ) {
        // SAFETY: NVTX callback contract (see above).
        let Some(name) = (unsafe { opt_wide(name) }) else {
            return;
        };
        with_subscriber((), |s| {
            s.domain_name_category_unicode(domain_id(domain), category, name);
        });
    }

    unsafe fn domain_register_string_a(
        domain: ffi::nvtxDomainHandle_t,
        string: *const c_char,
    ) -> ffi::nvtxStringHandle_t {
        // SAFETY: NVTX callback contract (see above).
        let Some(string) = (unsafe { opt_cstr(string) }) else {
            return core::ptr::null_mut();
        };
        let id = with_subscriber(RegisteredStringId::new(0), |s| {
            s.domain_register_string_ascii(domain_id(domain), string)
        });
        string_handle(id)
    }

    unsafe fn domain_register_string_w(
        domain: ffi::nvtxDomainHandle_t,
        string: *const ffi::wchar_t,
    ) -> ffi::nvtxStringHandle_t {
        // SAFETY: NVTX callback contract (see above).
        let Some(string) = (unsafe { opt_wide(string) }) else {
            return core::ptr::null_mut();
        };
        let id = with_subscriber(RegisteredStringId::new(0), |s| {
            s.domain_register_string_unicode(domain_id(domain), string)
        });
        string_handle(id)
    }

    unsafe fn domain_create_a(name: *const c_char) -> ffi::nvtxDomainHandle_t {
        // SAFETY: NVTX callback contract (see above).
        let Some(name) = (unsafe { opt_cstr(name) }) else {
            return core::ptr::null_mut();
        };
        let id = with_subscriber(DomainId::new(0), |s| s.domain_create_ascii(name));
        domain_handle(id)
    }

    unsafe fn domain_create_w(name: *const ffi::wchar_t) -> ffi::nvtxDomainHandle_t {
        // SAFETY: NVTX callback contract (see above).
        let Some(name) = (unsafe { opt_wide(name) }) else {
            return core::ptr::null_mut();
        };
        let id = with_subscriber(DomainId::new(0), |s| s.domain_create_unicode(name));
        domain_handle(id)
    }

    unsafe fn domain_destroy(domain: ffi::nvtxDomainHandle_t) {
        with_subscriber((), |s| s.domain_destroy(domain_id(domain)));
    }

    unsafe fn initialize(reserved: *const c_void) {
        let _ = reserved;
        with_subscriber((), |s| s.initialize());
    }
}
