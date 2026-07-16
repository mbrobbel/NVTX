// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Identifier types for the values NVTX tools hand back to the application.

/// Identifies a domain created by a domain-create callback.
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

/// Identifies a string registered by a register-string callback.
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

/// Identifies a resource created by a resource-create callback.
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
