// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! The high-level [`Subscriber`] trait implemented by NVTX tools.

use core::ffi::CStr;

use widestring::WideCStr;

use crate::tools::{next_handle, EventArgs, RangeId, ResourceArgs, NO_PUSH_POP_TRACKING};

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
}
