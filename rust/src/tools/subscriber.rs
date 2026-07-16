// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! The high-level [`Subscriber`] trait implemented by NVTX tools.

use core::ffi::CStr;

use widestring::WideCStr;

use crate::tools::{
    next_handle, EventArgs, RangeId, ResourceArgs, SyncUserArgs, NO_PUSH_POP_TRACKING,
};

/// Identifies a domain created by [`Subscriber::domain_create_ascii`] or
/// [`Subscriber::domain_create_unicode`].
///
/// [`DomainId::DEFAULT`] (`0`) identifies the default (global) domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DomainId(u64);

impl DomainId {
    /// The default (global) domain, represented by a null domain handle.
    pub const DEFAULT: Self = Self(0);

    /// Wrap a raw identifier value.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw identifier value.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Identifies a string registered by
/// [`Subscriber::domain_register_string_ascii`] or
/// [`Subscriber::domain_register_string_unicode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegisteredStringId(u64);

impl RegisteredStringId {
    /// Wrap a raw identifier value.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw identifier value.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Identifies a resource created by [`Subscriber::domain_resource_create`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(u64);

impl ResourceId {
    /// Wrap a raw identifier value.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw identifier value.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Identifies a user-defined synchronization object created by
/// [`Subscriber::domain_syncuser_create`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SyncUserId(u64);

impl SyncUserId {
    /// Wrap a raw identifier value.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw identifier value.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// An NVTX tool: receives every NVTX call made by the application.
///
/// Every method has a default implementation, so a tool only implements the
/// callbacks it cares about. Methods returning identifiers default to fresh
/// values from [`next_handle`]; NVTX treats these values as opaque handles and
/// hands them back verbatim in later callbacks, so tools may override them
/// with any nonzero value (for example a pointer to tool-owned state, like the
/// C++ `tools/sample-injection` reference does). [`DomainId`],
/// [`RegisteredStringId`], and [`ResourceId`] values cross the ABI as
/// pointer-typed handles and must fit the target's pointer width; wider values
/// surface to the application as null handles.
///
/// Borrowed arguments (`&CStr`, [`WideCStr`] references, [`EventArgs`], and
/// [`ResourceArgs`]) are only valid for the duration of the callback; copy
/// whatever must be retained.
///
/// Callbacks run on the application thread making the NVTX call, possibly
/// concurrently; hence the `Send + Sync` requirement. Blocking or heavy work
/// directly affects the application.
#[allow(unused_variables)]
pub trait Subscriber: Send + Sync {
    /// `nvtxMarkEx`: an instantaneous event.
    fn mark_ex(&self, args: &EventArgs<'_>) {}

    /// `nvtxMarkA`: an instantaneous event with an ASCII message.
    fn mark_ascii(&self, message: &CStr) {}

    /// `nvtxMarkW`: an instantaneous event with a Unicode message.
    fn mark_unicode(&self, message: &WideCStr) {}

    /// `nvtxRangeStartEx`: start a process-visible range.
    ///
    /// The returned identifier is handed back in [`Subscriber::range_end`].
    fn range_start_ex(&self, args: &EventArgs<'_>) -> RangeId {
        next_handle()
    }

    /// `nvtxRangeStartA`: start a process-visible range with an ASCII message.
    ///
    /// The returned identifier is handed back in [`Subscriber::range_end`].
    fn range_start_ascii(&self, message: &CStr) -> RangeId {
        next_handle()
    }

    /// `nvtxRangeStartW`: start a process-visible range with a Unicode
    /// message.
    ///
    /// The returned identifier is handed back in [`Subscriber::range_end`].
    fn range_start_unicode(&self, message: &WideCStr) -> RangeId {
        next_handle()
    }

    /// `nvtxRangeEnd`: end the process-visible range identified by `id`.
    fn range_end(&self, id: RangeId) {}

    /// `nvtxRangePushEx`: push a thread-visible range.
    ///
    /// Returns the range's zero-based depth on the current thread, or
    /// [`NO_PUSH_POP_TRACKING`] (the default) when depth is not tracked.
    fn range_push_ex(&self, args: &EventArgs<'_>) -> i32 {
        NO_PUSH_POP_TRACKING
    }

    /// `nvtxRangePushA`: push a thread-visible range with an ASCII message.
    ///
    /// See [`Subscriber::range_push_ex`] for the return value.
    fn range_push_ascii(&self, message: &CStr) -> i32 {
        NO_PUSH_POP_TRACKING
    }

    /// `nvtxRangePushW`: push a thread-visible range with a Unicode message.
    ///
    /// See [`Subscriber::range_push_ex`] for the return value.
    fn range_push_unicode(&self, message: &WideCStr) -> i32 {
        NO_PUSH_POP_TRACKING
    }

    /// `nvtxRangePop`: pop the innermost thread-visible range.
    ///
    /// Returns the popped range's zero-based depth on the current thread, or
    /// [`NO_PUSH_POP_TRACKING`] (the default) when depth is not tracked.
    fn range_pop(&self) -> i32 {
        NO_PUSH_POP_TRACKING
    }

    /// `nvtxNameCategoryA`: name a category with an ASCII string.
    fn name_category_ascii(&self, category: u32, name: &CStr) {}

    /// `nvtxNameCategoryW`: name a category with a Unicode string.
    fn name_category_unicode(&self, category: u32, name: &WideCStr) {}

    /// `nvtxNameOsThreadA`: name an OS thread with an ASCII string.
    fn name_os_thread_ascii(&self, thread_id: u32, name: &CStr) {}

    /// `nvtxNameOsThreadW`: name an OS thread with a Unicode string.
    fn name_os_thread_unicode(&self, thread_id: u32, name: &WideCStr) {}

    /// `nvtxDomainCreateA`: create a domain with an ASCII name.
    ///
    /// The returned identifier is handed back as the `domain` of later
    /// domain-scoped callbacks and must be nonzero ([`DomainId::DEFAULT`]
    /// identifies the default domain).
    fn domain_create_ascii(&self, name: &CStr) -> DomainId {
        DomainId::new(next_handle())
    }

    /// `nvtxDomainCreateW`: create a domain with a Unicode name.
    ///
    /// See [`Subscriber::domain_create_ascii`] for the return value.
    fn domain_create_unicode(&self, name: &WideCStr) -> DomainId {
        DomainId::new(next_handle())
    }

    /// `nvtxDomainDestroy`: destroy a domain.
    fn domain_destroy(&self, domain: DomainId) {}

    /// `nvtxDomainMarkEx`: an instantaneous event within a domain.
    fn domain_mark_ex(&self, domain: DomainId, args: &EventArgs<'_>) {}

    /// `nvtxDomainRangeStartEx`: start a process-visible range within a
    /// domain.
    ///
    /// The returned identifier is handed back in
    /// [`Subscriber::domain_range_end`].
    fn domain_range_start_ex(&self, domain: DomainId, args: &EventArgs<'_>) -> RangeId {
        next_handle()
    }

    /// `nvtxDomainRangeEnd`: end the process-visible range identified by `id`
    /// within a domain.
    fn domain_range_end(&self, domain: DomainId, id: RangeId) {}

    /// `nvtxDomainRangePushEx`: push a thread-visible range within a domain.
    ///
    /// See [`Subscriber::range_push_ex`] for the return value.
    fn domain_range_push_ex(&self, domain: DomainId, args: &EventArgs<'_>) -> i32 {
        NO_PUSH_POP_TRACKING
    }

    /// `nvtxDomainRangePop`: pop the innermost thread-visible range within a
    /// domain.
    ///
    /// See [`Subscriber::range_pop`] for the return value.
    fn domain_range_pop(&self, domain: DomainId) -> i32 {
        NO_PUSH_POP_TRACKING
    }

    /// `nvtxDomainResourceCreate`: associate a resource with a handle within a
    /// domain.
    ///
    /// The returned identifier is handed back in
    /// [`Subscriber::domain_resource_destroy`].
    fn domain_resource_create(&self, domain: DomainId, args: &ResourceArgs<'_>) -> ResourceId {
        ResourceId::new(next_handle())
    }

    /// `nvtxDomainResourceDestroy`: release a resource identifier.
    fn domain_resource_destroy(&self, resource: ResourceId) {}

    /// `nvtxDomainNameCategoryA`: name a category within a domain with an
    /// ASCII string.
    fn domain_name_category_ascii(&self, domain: DomainId, category: u32, name: &CStr) {}

    /// `nvtxDomainNameCategoryW`: name a category within a domain with a
    /// Unicode string.
    fn domain_name_category_unicode(&self, domain: DomainId, category: u32, name: &WideCStr) {}

    /// `nvtxDomainRegisterStringA`: register an ASCII string with a domain.
    ///
    /// The string value is only provided here; later callbacks refer to it via
    /// the returned identifier (see [`MessageView::Registered`]), so tools
    /// that resolve registered strings must copy it now.
    ///
    /// [`MessageView::Registered`]: crate::tools::MessageView::Registered
    fn domain_register_string_ascii(&self, domain: DomainId, string: &CStr) -> RegisteredStringId {
        RegisteredStringId::new(next_handle())
    }

    /// `nvtxDomainRegisterStringW`: register a Unicode string with a domain.
    ///
    /// See [`Subscriber::domain_register_string_ascii`].
    fn domain_register_string_unicode(
        &self,
        domain: DomainId,
        string: &WideCStr,
    ) -> RegisteredStringId {
        RegisteredStringId::new(next_handle())
    }

    /// `nvtxInitialize`: explicit NVTX initialization by the application.
    fn initialize(&self) {}

    // CUDA driver, CUDA runtime, and OpenCL resource-naming callbacks receive
    // the application's opaque handles; pointer-typed handles surface as their
    // raw address bits.

    /// `nvtxNameCuDeviceA`: name a CUDA device with an ASCII string.
    fn name_cudevice_ascii(&self, device: i32, name: &CStr) {}

    /// `nvtxNameCuDeviceW`: name a CUDA device with a Unicode string.
    fn name_cudevice_unicode(&self, device: i32, name: &WideCStr) {}

    /// `nvtxNameCuContextA`: name a CUDA context with an ASCII string.
    fn name_cucontext_ascii(&self, context: u64, name: &CStr) {}

    /// `nvtxNameCuContextW`: name a CUDA context with a Unicode string.
    fn name_cucontext_unicode(&self, context: u64, name: &WideCStr) {}

    /// `nvtxNameCuStreamA`: name a CUDA stream with an ASCII string.
    fn name_custream_ascii(&self, stream: u64, name: &CStr) {}

    /// `nvtxNameCuStreamW`: name a CUDA stream with a Unicode string.
    fn name_custream_unicode(&self, stream: u64, name: &WideCStr) {}

    /// `nvtxNameCuEventA`: name a CUDA event with an ASCII string.
    fn name_cuevent_ascii(&self, event: u64, name: &CStr) {}

    /// `nvtxNameCuEventW`: name a CUDA event with a Unicode string.
    fn name_cuevent_unicode(&self, event: u64, name: &WideCStr) {}

    /// `nvtxNameCudaDeviceA`: name a CUDA runtime device with an ASCII string.
    fn name_cuda_device_ascii(&self, device: i32, name: &CStr) {}

    /// `nvtxNameCudaDeviceW`: name a CUDA runtime device with a Unicode
    /// string.
    fn name_cuda_device_unicode(&self, device: i32, name: &WideCStr) {}

    /// `nvtxNameCudaStreamA`: name a CUDA runtime stream with an ASCII string.
    fn name_cuda_stream_ascii(&self, stream: u64, name: &CStr) {}

    /// `nvtxNameCudaStreamW`: name a CUDA runtime stream with a Unicode
    /// string.
    fn name_cuda_stream_unicode(&self, stream: u64, name: &WideCStr) {}

    /// `nvtxNameCudaEventA`: name a CUDA runtime event with an ASCII string.
    fn name_cuda_event_ascii(&self, event: u64, name: &CStr) {}

    /// `nvtxNameCudaEventW`: name a CUDA runtime event with a Unicode string.
    fn name_cuda_event_unicode(&self, event: u64, name: &WideCStr) {}

    /// `nvtxNameClDeviceA`: name an `OpenCL` device with an ASCII string.
    fn name_cl_device_ascii(&self, device: u64, name: &CStr) {}

    /// `nvtxNameClDeviceW`: name an `OpenCL` device with a Unicode string.
    fn name_cl_device_unicode(&self, device: u64, name: &WideCStr) {}

    /// `nvtxNameClContextA`: name an `OpenCL` context with an ASCII string.
    fn name_cl_context_ascii(&self, context: u64, name: &CStr) {}

    /// `nvtxNameClContextW`: name an `OpenCL` context with a Unicode string.
    fn name_cl_context_unicode(&self, context: u64, name: &WideCStr) {}

    /// `nvtxNameClCommandQueueA`: name an `OpenCL` command queue with an ASCII
    /// string.
    fn name_cl_command_queue_ascii(&self, command_queue: u64, name: &CStr) {}

    /// `nvtxNameClCommandQueueW`: name an `OpenCL` command queue with a
    /// Unicode string.
    fn name_cl_command_queue_unicode(&self, command_queue: u64, name: &WideCStr) {}

    /// `nvtxNameClMemObjectA`: name an `OpenCL` memory object with an ASCII
    /// string.
    fn name_cl_mem_object_ascii(&self, mem_object: u64, name: &CStr) {}

    /// `nvtxNameClMemObjectW`: name an `OpenCL` memory object with a Unicode
    /// string.
    fn name_cl_mem_object_unicode(&self, mem_object: u64, name: &WideCStr) {}

    /// `nvtxNameClSamplerA`: name an `OpenCL` sampler with an ASCII string.
    fn name_cl_sampler_ascii(&self, sampler: u64, name: &CStr) {}

    /// `nvtxNameClSamplerW`: name an `OpenCL` sampler with a Unicode string.
    fn name_cl_sampler_unicode(&self, sampler: u64, name: &WideCStr) {}

    /// `nvtxNameClProgramA`: name an `OpenCL` program with an ASCII string.
    fn name_cl_program_ascii(&self, program: u64, name: &CStr) {}

    /// `nvtxNameClProgramW`: name an `OpenCL` program with a Unicode string.
    fn name_cl_program_unicode(&self, program: u64, name: &WideCStr) {}

    /// `nvtxNameClEventA`: name an `OpenCL` event with an ASCII string.
    fn name_cl_event_ascii(&self, event: u64, name: &CStr) {}

    /// `nvtxNameClEventW`: name an `OpenCL` event with a Unicode string.
    fn name_cl_event_unicode(&self, event: u64, name: &WideCStr) {}

    /// `nvtxDomainSyncUserCreate`: create a user-defined synchronization
    /// object within a domain.
    ///
    /// The returned identifier is handed back in the other `domain_syncuser_*`
    /// callbacks.
    fn domain_syncuser_create(&self, domain: DomainId, args: &SyncUserArgs<'_>) -> SyncUserId {
        SyncUserId::new(next_handle())
    }

    /// `nvtxDomainSyncUserDestroy`: destroy a user-defined synchronization
    /// object.
    fn domain_syncuser_destroy(&self, handle: SyncUserId) {}

    /// `nvtxDomainSyncUserAcquireStart`: the synchronization object started to
    /// acquire.
    fn domain_syncuser_acquire_start(&self, handle: SyncUserId) {}

    /// `nvtxDomainSyncUserAcquireFailed`: the acquisition failed.
    fn domain_syncuser_acquire_failed(&self, handle: SyncUserId) {}

    /// `nvtxDomainSyncUserAcquireSuccess`: the acquisition succeeded.
    fn domain_syncuser_acquire_success(&self, handle: SyncUserId) {}

    /// `nvtxDomainSyncUserReleasing`: the synchronization object is released.
    fn domain_syncuser_releasing(&self, handle: SyncUserId) {}
}
