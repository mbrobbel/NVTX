// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! End-to-end test of `nvtx::tools` against a fake NVTX client: attach a
//! recording subscriber, then invoke the installed callback slots through the
//! NVTX ABI types and assert the subscriber observed decoded values.

#![cfg(feature = "tools")]
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::panic)]
#![allow(clippy::manual_assert)]
#![allow(clippy::missing_transmute_annotations)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::multiple_unsafe_ops_per_block)]
#![allow(clippy::undocumented_unsafe_blocks)]

use core::cell::UnsafeCell;
use core::ffi::{c_int, c_uint, c_void};
use core::mem::{size_of, transmute};
use core::ptr::addr_of;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::ffi::{CStr, CString};
use std::sync::Mutex;

use nvtx::sys::ffi;
use nvtx::tools::{
    self, DomainId, EventArgs, ExportTable, MessageView, RangeId, RegisteredStringId, ResourceArgs,
    ResourceId, Subscriber,
};
use widestring::{WideCStr, WideCString};

const SLOTS: usize = 16;

/// The fake NVTX client's callback slots and slot-pointer tables.
struct Tables {
    core_slots: [ffi::NvtxFunctionPointer; SLOTS],
    core_pointers: [*mut ffi::NvtxFunctionPointer; SLOTS],
    core2_slots: [ffi::NvtxFunctionPointer; SLOTS],
    core2_pointers: [*mut ffi::NvtxFunctionPointer; SLOTS],
}

struct SyncTables(UnsafeCell<Tables>);

// SAFETY: the single test accesses the tables from one thread only.
unsafe impl Sync for SyncTables {}

static TABLES: SyncTables = SyncTables(UnsafeCell::new(Tables {
    core_slots: [None; SLOTS],
    core_pointers: [core::ptr::null_mut(); SLOTS],
    core2_slots: [None; SLOTS],
    core2_pointers: [core::ptr::null_mut(); SLOTS],
}));

// The fake NVTX functions use the `NVTX_API` calling convention (`__stdcall`
// on 32-bit x86 Windows, the C convention everywhere else), matching the
// bindgen-generated function-pointer types for the target. The thin cfg'd
// wrappers below select the convention; the logic lives in plain functions.

unsafe fn get_module_function_table_impl(
    module: ffi::NvtxCallbackModule,
    out_table: *mut ffi::NvtxFunctionTable,
    out_size: *mut c_uint,
) -> c_int {
    let tables = unsafe { &mut *TABLES.0.get() };
    let pointers = match module {
        ffi::NvtxCallbackModule::NVTX_CB_MODULE_CORE => tables.core_pointers.as_mut_ptr(),
        ffi::NvtxCallbackModule::NVTX_CB_MODULE_CORE2 => tables.core2_pointers.as_mut_ptr(),
        _ => return 0,
    };
    unsafe {
        *out_table = pointers;
        // NVTX reports the highest valid callback id (slot count minus one).
        *out_size = (SLOTS - 1) as c_uint;
    }
    1
}

#[cfg(all(windows, target_arch = "x86"))]
unsafe extern "stdcall" fn get_module_function_table(
    module: ffi::NvtxCallbackModule,
    out_table: *mut ffi::NvtxFunctionTable,
    out_size: *mut c_uint,
) -> c_int {
    unsafe { get_module_function_table_impl(module, out_table, out_size) }
}

#[cfg(not(all(windows, target_arch = "x86")))]
unsafe extern "C" fn get_module_function_table(
    module: ffi::NvtxCallbackModule,
    out_table: *mut ffi::NvtxFunctionTable,
    out_size: *mut c_uint,
) -> c_int {
    unsafe { get_module_function_table_impl(module, out_table, out_size) }
}

static CALLBACKS: ffi::NvtxExportTableCallbacks = ffi::NvtxExportTableCallbacks {
    struct_size: size_of::<ffi::NvtxExportTableCallbacks>(),
    GetModuleFunctionTable: Some(get_module_function_table),
};

static REPORTED_VERSION: AtomicU32 = AtomicU32::new(0);

fn set_injection_nvtx_version_impl(version: u32) {
    REPORTED_VERSION.store(version, Ordering::SeqCst);
}

#[cfg(all(windows, target_arch = "x86"))]
unsafe extern "stdcall" fn set_injection_nvtx_version(version: u32) {
    set_injection_nvtx_version_impl(version);
}

#[cfg(not(all(windows, target_arch = "x86")))]
unsafe extern "C" fn set_injection_nvtx_version(version: u32) {
    set_injection_nvtx_version_impl(version);
}

static VERSION_INFO: ffi::NvtxExportTableVersionInfo = ffi::NvtxExportTableVersionInfo {
    struct_size: size_of::<ffi::NvtxExportTableVersionInfo>(),
    version: 3,
    reserved0: 0,
    SetInjectionNvtxVersion: Some(set_injection_nvtx_version),
};

fn get_export_table_impl(id: u32) -> *const c_void {
    match id {
        id if id == ffi::NvtxExportTableID::NVTX_ETID_CALLBACKS as u32 => {
            addr_of!(CALLBACKS).cast()
        }
        id if id == ffi::NvtxExportTableID::NVTX_ETID_VERSIONINFO as u32 => {
            addr_of!(VERSION_INFO).cast()
        }
        _ => core::ptr::null(),
    }
}

#[cfg(all(windows, target_arch = "x86"))]
unsafe extern "stdcall" fn get_export_table(id: u32) -> *const c_void {
    get_export_table_impl(id)
}

#[cfg(not(all(windows, target_arch = "x86")))]
unsafe extern "C" fn get_export_table(id: u32) -> *const c_void {
    get_export_table_impl(id)
}

/// Owned copies of the events observed by [`Recorder`].
#[derive(Debug, PartialEq)]
enum Event {
    MarkEx {
        category: Option<u32>,
        message: Option<String>,
    },
    MarkAscii(String),
    RangeStartAscii(String, u64),
    RangeEnd(u64),
    RangePop,
    NameCategoryUnicode(u32, String),
    NameOsThreadAscii(u32, String),
    DomainCreateAscii(String, u64),
    DomainDestroy(u64),
    DomainMarkEx(u64, Option<String>),
    DomainRangeStartEx(u64, u64),
    DomainRangeEnd(u64, u64),
    DomainRangePushEx(u64),
    DomainRangePop(u64),
    DomainResourceCreate {
        domain: u64,
        identifier: u64,
        name: Option<String>,
        id: u64,
    },
    DomainResourceDestroy(u64),
    DomainRegisterStringAscii(u64, String, u64),
    Initialize,
}

static EVENTS: Mutex<Vec<Event>> = Mutex::new(Vec::new());
static PANIC_NEXT: AtomicBool = AtomicBool::new(false);

fn record(event: Event) {
    EVENTS.lock().unwrap().push(event);
}

fn take_events() -> Vec<Event> {
    core::mem::take(&mut *EVENTS.lock().unwrap())
}

fn owned_message_view(message: MessageView<'_>) -> String {
    match message {
        MessageView::Ascii(s) => s.to_string_lossy().into_owned(),
        MessageView::Unicode(s) => s.to_string_lossy(),
        MessageView::Registered(id) => format!("registered:{}", id.raw()),
        _ => unreachable!("unknown message view variant"),
    }
}

fn owned_message(args: &EventArgs<'_>) -> Option<String> {
    args.message.map(owned_message_view)
}

struct Recorder;

impl Subscriber for Recorder {
    fn mark_ex(&self, args: &EventArgs<'_>) {
        record(Event::MarkEx {
            category: args.category,
            message: owned_message(args),
        });
    }

    fn mark_ascii(&self, message: &CStr) {
        if PANIC_NEXT.swap(false, Ordering::SeqCst) {
            panic!("injected test panic");
        }
        record(Event::MarkAscii(message.to_string_lossy().into_owned()));
    }

    fn range_start_ascii(&self, message: &CStr) -> RangeId {
        let id = tools::next_handle();
        record(Event::RangeStartAscii(
            message.to_string_lossy().into_owned(),
            id,
        ));
        id
    }

    fn range_end(&self, id: RangeId) {
        record(Event::RangeEnd(id));
    }

    // `range_push_ascii` is deliberately not overridden: the default must
    // return `NO_PUSH_POP_TRACKING` without observing an event.

    fn range_pop(&self) -> i32 {
        record(Event::RangePop);
        tools::NO_PUSH_POP_TRACKING
    }

    fn name_category_unicode(&self, category: u32, name: &WideCStr) {
        record(Event::NameCategoryUnicode(category, name.to_string_lossy()));
    }

    fn name_os_thread_ascii(&self, thread_id: u32, name: &CStr) {
        record(Event::NameOsThreadAscii(
            thread_id,
            name.to_string_lossy().into_owned(),
        ));
    }

    fn domain_create_ascii(&self, name: &CStr) -> DomainId {
        let id = DomainId::new(tools::next_handle());
        record(Event::DomainCreateAscii(
            name.to_string_lossy().into_owned(),
            id.raw(),
        ));
        id
    }

    fn domain_destroy(&self, domain: DomainId) {
        record(Event::DomainDestroy(domain.raw()));
    }

    fn domain_mark_ex(&self, domain: DomainId, args: &EventArgs<'_>) {
        record(Event::DomainMarkEx(domain.raw(), owned_message(args)));
    }

    fn domain_range_start_ex(&self, domain: DomainId, _args: &EventArgs<'_>) -> RangeId {
        let id = tools::next_handle();
        record(Event::DomainRangeStartEx(domain.raw(), id));
        id
    }

    fn domain_range_end(&self, domain: DomainId, id: RangeId) {
        record(Event::DomainRangeEnd(domain.raw(), id));
    }

    fn domain_range_push_ex(&self, domain: DomainId, _args: &EventArgs<'_>) -> i32 {
        record(Event::DomainRangePushEx(domain.raw()));
        tools::NO_PUSH_POP_TRACKING
    }

    fn domain_range_pop(&self, domain: DomainId) -> i32 {
        record(Event::DomainRangePop(domain.raw()));
        tools::NO_PUSH_POP_TRACKING
    }

    fn domain_resource_create(&self, domain: DomainId, args: &ResourceArgs<'_>) -> ResourceId {
        let id = ResourceId::new(tools::next_handle());
        record(Event::DomainResourceCreate {
            domain: domain.raw(),
            identifier: args.identifier,
            name: args.message.map(owned_message_view),
            id: id.raw(),
        });
        id
    }

    fn domain_resource_destroy(&self, resource: ResourceId) {
        record(Event::DomainResourceDestroy(resource.raw()));
    }

    fn domain_register_string_ascii(&self, domain: DomainId, string: &CStr) -> RegisteredStringId {
        let id = RegisteredStringId::new(tools::next_handle());
        record(Event::DomainRegisterStringAscii(
            domain.raw(),
            string.to_string_lossy().into_owned(),
            id.raw(),
        ));
        id
    }

    fn initialize(&self) {
        record(Event::Initialize);
    }
}

fn core_slot(id: u32) -> ffi::NvtxFunctionPointer {
    let tables = unsafe { &*TABLES.0.get() };
    tables.core_slots[id as usize]
}

fn core2_slot(id: u32) -> ffi::NvtxFunctionPointer {
    let tables = unsafe { &*TABLES.0.get() };
    tables.core2_slots[id as usize]
}

fn event_attributes(message: &CStr) -> ffi::nvtxEventAttributes_v2 {
    ffi::nvtxEventAttributes_v2 {
        version: 2,
        size: size_of::<ffi::nvtxEventAttributes_v2>() as u16,
        category: 7,
        colorType: 0,
        color: 0,
        payloadType: 0,
        reserved0: 0,
        payload: ffi::nvtxEventAttributes_v2_payload_t { ullValue: 0 },
        messageType: i32::from(ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII),
        message: ffi::nvtxMessageValue_t {
            ascii: message.as_ptr(),
        },
    }
}

// All attach-dependent assertions live in this single test: `attach` is
// process-global and one-shot, so ordering must be deterministic.
#[test]
fn attach_installs_working_trampolines() {
    // Point the slot-pointer tables at the slots, leaving `MarkW` (id 3)
    // deliberately null to exercise best-effort slot skipping.
    {
        let tables = unsafe { &mut *TABLES.0.get() };
        for id in 0..SLOTS {
            tables.core_pointers[id] = core::ptr::from_mut(&mut tables.core_slots[id]);
            tables.core2_pointers[id] = core::ptr::from_mut(&mut tables.core2_slots[id]);
        }
        tables.core_pointers[3] = core::ptr::null_mut();
    }

    // SAFETY: `get_export_table` upholds the NVTX export-table contract.
    unsafe { tools::attach(Some(get_export_table), Recorder) }.unwrap();
    // Attaching is one-shot.
    assert_eq!(
        // SAFETY: as above.
        unsafe { tools::attach(Some(get_export_table), Recorder) },
        Err(tools::AttachError::AlreadyAttached)
    );

    // Every slot is installed except the null `MarkW` slot.
    for id in 1..SLOTS as u32 {
        assert_eq!(core_slot(id).is_none(), id == 3, "core slot {id}");
        assert!(core2_slot(id).is_some(), "core2 slot {id}");
    }
    assert!(core_slot(0).is_none());
    assert!(core2_slot(0).is_none());

    // Attaching reported the tool's NVTX version to the application.
    assert_eq!(REPORTED_VERSION.load(Ordering::SeqCst), 3);

    // The version-info export table is reachable through the low-level API.
    // SAFETY: as above.
    let export = unsafe { ExportTable::new(Some(get_export_table)) }.unwrap();
    assert_eq!(export.version_info().unwrap().version(), 3);

    // CORE: mark with an ASCII message.
    let mark_a: ffi::nvtxMarkA_impl_fntype = unsafe { transmute(core_slot(2)) };
    let mark_a = mark_a.unwrap();
    let message = CString::new("mark").unwrap();
    unsafe { mark_a(message.as_ptr()) };
    // A null message is skipped.
    unsafe { mark_a(core::ptr::null()) };
    assert_eq!(take_events(), [Event::MarkAscii("mark".to_owned())]);

    // CORE: mark with attributes, full-size and truncated.
    let mark_ex: ffi::nvtxMarkEx_impl_fntype = unsafe { transmute(core_slot(1)) };
    let mark_ex = mark_ex.unwrap();
    let message = CString::new("attributed").unwrap();
    let mut attributes = event_attributes(&message);
    unsafe { mark_ex(&attributes) };
    // Truncate the declared size to just past `category`: the message (and
    // its dangling-looking pointer bits) must be ignored.
    attributes.size =
        (core::mem::offset_of!(ffi::nvtxEventAttributes_v2, category) + size_of::<u32>()) as u16;
    unsafe { mark_ex(&attributes) };
    unsafe { mark_ex(core::ptr::null()) };
    assert_eq!(
        take_events(),
        [
            Event::MarkEx {
                category: Some(7),
                message: Some("attributed".to_owned()),
            },
            Event::MarkEx {
                category: Some(7),
                message: None,
            },
            Event::MarkEx {
                category: None,
                message: None,
            },
        ]
    );

    // CORE: range start/end round-trips the subscriber-returned id.
    let range_start_a: ffi::nvtxRangeStartA_impl_fntype = unsafe { transmute(core_slot(5)) };
    let range_end: ffi::nvtxRangeEnd_impl_fntype = unsafe { transmute(core_slot(7)) };
    let message = CString::new("range").unwrap();
    let range_id = unsafe { range_start_a.unwrap()(message.as_ptr()) };
    assert_ne!(range_id, 0);
    unsafe { range_end.unwrap()(range_id) };
    assert_eq!(
        take_events(),
        [
            Event::RangeStartAscii("range".to_owned(), range_id),
            Event::RangeEnd(range_id),
        ]
    );

    // CORE: the default (not overridden) push returns NO_PUSH_POP_TRACKING
    // and the overridden pop still records.
    let range_push_a: ffi::nvtxRangePushA_impl_fntype = unsafe { transmute(core_slot(9)) };
    let range_pop: ffi::nvtxRangePop_impl_fntype = unsafe { transmute(core_slot(11)) };
    let message = CString::new("push").unwrap();
    assert_eq!(
        unsafe { range_push_a.unwrap()(message.as_ptr()) },
        tools::NO_PUSH_POP_TRACKING
    );
    assert_eq!(unsafe { range_pop.unwrap()() }, tools::NO_PUSH_POP_TRACKING);
    assert_eq!(take_events(), [Event::RangePop]);

    // CORE: category naming decodes wide strings; thread naming ASCII.
    let name_category_w: ffi::nvtxNameCategoryW_impl_fntype = unsafe { transmute(core_slot(13)) };
    let name_os_thread_a: ffi::nvtxNameOsThreadA_impl_fntype = unsafe { transmute(core_slot(14)) };
    let wide = WideCString::from_str("wide category").unwrap();
    unsafe { name_category_w.unwrap()(11, wide.as_ptr().cast()) };
    let name = CString::new("worker").unwrap();
    unsafe { name_os_thread_a.unwrap()(42, name.as_ptr()) };
    assert_eq!(
        take_events(),
        [
            Event::NameCategoryUnicode(11, "wide category".to_owned()),
            Event::NameOsThreadAscii(42, "worker".to_owned()),
        ]
    );

    // CORE2: domain lifecycle round-trips the subscriber-returned handle.
    let domain_create_a: ffi::nvtxDomainCreateA_impl_fntype = unsafe { transmute(core2_slot(12)) };
    let domain_destroy: ffi::nvtxDomainDestroy_impl_fntype = unsafe { transmute(core2_slot(14)) };
    let name = CString::new("domain").unwrap();
    let domain = unsafe { domain_create_a.unwrap()(name.as_ptr()) };
    assert!(!domain.is_null());
    let domain_raw = domain as usize as u64;

    // CORE2: registered strings surface as `MessageView::Registered` later.
    let register_string_a: ffi::nvtxDomainRegisterStringA_impl_fntype =
        unsafe { transmute(core2_slot(10)) };
    let string = CString::new("registered").unwrap();
    let registered = unsafe { register_string_a.unwrap()(domain, string.as_ptr()) };
    assert!(!registered.is_null());
    let registered_raw = registered as usize as u64;

    let domain_mark_ex: ffi::nvtxDomainMarkEx_impl_fntype = unsafe { transmute(core2_slot(1)) };
    let mut attributes = event_attributes(&string);
    attributes.messageType = i32::from(ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_REGISTERED);
    attributes.message = ffi::nvtxMessageValue_t { registered };
    unsafe { domain_mark_ex.unwrap()(domain, &attributes) };

    // CORE2: ranges, push/pop, resources, and initialize.
    let domain_range_start_ex: ffi::nvtxDomainRangeStartEx_impl_fntype =
        unsafe { transmute(core2_slot(2)) };
    let domain_range_end: ffi::nvtxDomainRangeEnd_impl_fntype = unsafe { transmute(core2_slot(3)) };
    let domain_range_id = unsafe { domain_range_start_ex.unwrap()(domain, core::ptr::null()) };
    assert_ne!(domain_range_id, 0);
    unsafe { domain_range_end.unwrap()(domain, domain_range_id) };

    let domain_range_push_ex: ffi::nvtxDomainRangePushEx_impl_fntype =
        unsafe { transmute(core2_slot(4)) };
    let domain_range_pop: ffi::nvtxDomainRangePop_impl_fntype = unsafe { transmute(core2_slot(5)) };
    assert_eq!(
        unsafe { domain_range_push_ex.unwrap()(domain, core::ptr::null()) },
        tools::NO_PUSH_POP_TRACKING
    );
    assert_eq!(
        unsafe { domain_range_pop.unwrap()(domain) },
        tools::NO_PUSH_POP_TRACKING
    );

    let resource_create: ffi::nvtxDomainResourceCreate_impl_fntype =
        unsafe { transmute(core2_slot(6)) };
    let resource_destroy: ffi::nvtxDomainResourceDestroy_impl_fntype =
        unsafe { transmute(core2_slot(7)) };
    let resource_name = CString::new("resource").unwrap();
    let mut resource_attributes = ffi::nvtxResourceAttributes_v0 {
        version: 0,
        size: size_of::<ffi::nvtxResourceAttributes_v0>() as u16,
        identifierType: 65538,
        identifier: ffi::nvtxResourceAttributes_v0_identifier_t { ullValue: 0x1234 },
        messageType: i32::from(ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII),
        message: ffi::nvtxMessageValue_t {
            ascii: resource_name.as_ptr(),
        },
    };
    let resource = unsafe { resource_create.unwrap()(domain, &mut resource_attributes) };
    assert!(!resource.is_null());
    let resource_raw = resource as usize as u64;
    unsafe { resource_destroy.unwrap()(resource) };

    let initialize: ffi::nvtxInitialize_impl_fntype = unsafe { transmute(core2_slot(15)) };
    unsafe { initialize.unwrap()(core::ptr::null()) };

    unsafe { domain_destroy.unwrap()(domain) };

    assert_eq!(
        take_events(),
        [
            Event::DomainCreateAscii("domain".to_owned(), domain_raw),
            Event::DomainRegisterStringAscii(domain_raw, "registered".to_owned(), registered_raw),
            Event::DomainMarkEx(domain_raw, Some(format!("registered:{registered_raw}"))),
            Event::DomainRangeStartEx(domain_raw, domain_range_id),
            Event::DomainRangeEnd(domain_raw, domain_range_id),
            Event::DomainRangePushEx(domain_raw),
            Event::DomainRangePop(domain_raw),
            Event::DomainResourceCreate {
                domain: domain_raw,
                identifier: 0x1234,
                name: Some("resource".to_owned()),
                id: resource_raw,
            },
            Event::DomainResourceDestroy(resource_raw),
            Event::Initialize,
            Event::DomainDestroy(domain_raw),
        ]
    );

    // A panicking subscriber never unwinds into the caller, and later
    // callbacks still work.
    PANIC_NEXT.store(true, Ordering::SeqCst);
    let message = CString::new("panics").unwrap();
    unsafe { mark_a(message.as_ptr()) };
    let message = CString::new("after panic").unwrap();
    unsafe { mark_a(message.as_ptr()) };
    assert_eq!(take_events(), [Event::MarkAscii("after panic".to_owned())]);
}
