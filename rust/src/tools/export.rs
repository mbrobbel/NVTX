// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Low-level typed access to the NVTX export tables.

use core::ffi::c_uint;
use core::marker::PhantomData;

use crate::sys::ffi;
use crate::tools::callback::{CallbackId, CallbackModule};
use crate::tools::AttachError;

/// Accessor for the export tables NVTX passes to `InitializeInjectionNvtx2`.
#[derive(Debug, Clone, Copy)]
pub struct ExportTable {
    /// The accessor, non-null by construction (see [`ExportTable::new`]).
    get: ffi::NvtxGetExportTableFunc_t,
}

impl ExportTable {
    /// Wrap the export-table accessor NVTX passed to
    /// `InitializeInjectionNvtx2`. Returns `None` for a null accessor.
    ///
    /// # Safety
    ///
    /// `get_export_table` must be the accessor NVTX passed to
    /// `InitializeInjectionNvtx2`, or a function upholding its contract: any
    /// non-null pointer returned for a supported export-table identifier must
    /// point to the corresponding valid export table which lives for the rest
    /// of the process.
    #[must_use]
    pub unsafe fn new(get_export_table: ffi::NvtxGetExportTableFunc_t) -> Option<Self> {
        get_export_table.is_some().then_some(Self {
            get: get_export_table,
        })
    }

    /// Fetch a pointer to the export table with the given identifier, or
    /// `None` if NVTX does not provide it.
    fn table_ptr<T>(&self, id: ffi::NvtxExportTableID) -> Option<*const T> {
        let get = self.get?;
        // CAST: `NvtxExportTableID` is a `repr(u32)` enum.
        // SAFETY: `get` is NVTX's export-table accessor (see the
        // `ExportTable::new` contract).
        let ptr = unsafe { get(id as u32) }.cast::<T>();
        (!ptr.is_null()).then_some(ptr)
    }

    /// Fetch the callbacks export table, validating that its declared
    /// `struct_size` covers the fields read.
    fn callbacks(&self) -> Option<ffi::NvtxExportTableCallbacks> {
        let ptr = self.table_ptr::<ffi::NvtxExportTableCallbacks>(
            ffi::NvtxExportTableID::NVTX_ETID_CALLBACKS,
        )?;
        // SAFETY: `ptr` is non-null and `struct_size` is the first field of
        // every export table (see the `ExportTable::new` contract).
        let struct_size = unsafe { (*ptr).struct_size };
        if struct_size < core::mem::size_of::<ffi::NvtxExportTableCallbacks>() {
            return None;
        }
        // SAFETY: `struct_size` declares that the full struct is present.
        Some(unsafe { *ptr })
    }

    /// Fetch the function table of callback module `M`.
    ///
    /// # Errors
    ///
    /// Returns [`AttachError::CallbacksUnavailable`] when the callbacks export
    /// table is unavailable, too small, or has no `GetModuleFunctionTable`
    /// accessor, and [`AttachError::ModuleTableUnavailable`] when NVTX declines
    /// to provide the module's table.
    pub fn function_table<M: CallbackModule>(&self) -> Result<FunctionTable<M>, AttachError> {
        let callbacks = self.callbacks().ok_or(AttachError::CallbacksUnavailable)?;
        let get_module_table = callbacks
            .GetModuleFunctionTable
            .ok_or(AttachError::CallbacksUnavailable)?;
        let mut table: ffi::NvtxFunctionTable = core::ptr::null_mut();
        let mut max_id: c_uint = 0;
        // SAFETY: `get_module_table` comes from the validated callbacks export
        // table and is called with valid out-pointers.
        let ret = unsafe { get_module_table(M::MODULE, &mut table, &mut max_id) };
        if ret == 0 || table.is_null() {
            return Err(AttachError::ModuleTableUnavailable(M::MODULE));
        }
        Ok(FunctionTable {
            table,
            max_id,
            _module: PhantomData,
        })
    }

    /// Fetch the version-info export table, or `None` if NVTX does not provide
    /// it.
    #[must_use]
    pub fn version_info(&self) -> Option<VersionInfo> {
        let ptr = self.table_ptr::<ffi::NvtxExportTableVersionInfo>(
            ffi::NvtxExportTableID::NVTX_ETID_VERSIONINFO,
        )?;
        // SAFETY: `ptr` is non-null and `struct_size` is the first field of
        // every export table (see the `ExportTable::new` contract).
        let struct_size = unsafe { (*ptr).struct_size };
        if struct_size < core::mem::size_of::<ffi::NvtxExportTableVersionInfo>() {
            return None;
        }
        // SAFETY: `struct_size` declares that the full struct is present.
        let info = unsafe { *ptr };
        Some(VersionInfo { info })
    }
}

/// The NVTX version information export table.
#[derive(Debug, Clone, Copy)]
pub struct VersionInfo {
    info: ffi::NvtxExportTableVersionInfo,
}

impl VersionInfo {
    /// The NVTX API version of the headers the application was built against.
    #[must_use]
    pub fn version(&self) -> u32 {
        self.info.version
    }

    /// Report the tool's NVTX version back to the application.
    ///
    /// NVTX currently ignores this value; it exists so applications can detect
    /// problematic tool versions in emergency situations.
    pub fn set_injection_nvtx_version(&self, version: u32) {
        if let Some(set_version) = self.info.SetInjectionNvtxVersion {
            // SAFETY: `set_version` comes from the validated version-info
            // export table.
            unsafe { set_version(version) }
        }
    }
}

/// Error returned by [`FunctionTable::set`] and [`FunctionTable::clear`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SlotError {
    /// The callback identifier exceeds the table provided by the application's
    /// NVTX version.
    OutOfBounds {
        /// The rejected callback identifier.
        id: u32,
        /// The highest valid callback identifier in the table.
        max_id: u32,
    },
    /// The slot pointer for the callback identifier is null.
    NullSlot {
        /// The rejected callback identifier.
        id: u32,
    },
}

impl core::fmt::Display for SlotError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfBounds { id, max_id } => {
                write!(
                    f,
                    "callback id {id} exceeds the table's highest id {max_id}"
                )
            }
            Self::NullSlot { id } => write!(f, "callback id {id} has a null slot pointer"),
        }
    }
}

impl std::error::Error for SlotError {}

/// The function table of callback module `M`, holding one type-erased callback
/// slot per callback identifier.
#[derive(Debug)]
pub struct FunctionTable<M> {
    table: ffi::NvtxFunctionTable,
    /// The highest valid callback identifier: NVTX reports the slot count
    /// minus one, excluding the reserved slot 0 (see
    /// `nvtxEtiGetModuleFunctionTable` in `nvtxDetail/nvtxImpl.h`).
    max_id: c_uint,
    _module: PhantomData<M>,
}

impl<M: CallbackModule> FunctionTable<M> {
    /// The highest valid callback identifier in this table.
    #[must_use]
    pub fn max_id(&self) -> u32 {
        self.max_id
    }

    /// Install `f` in the slot of callback `C`.
    ///
    /// The function-pointer type is checked at compile time against the slot's
    /// ABI signature. Passing `None` resets the slot (see [`Self::clear`]).
    ///
    /// # Errors
    ///
    /// See [`SlotError`]; out-of-bounds identifiers occur when the application
    /// uses an older NVTX version with a smaller table.
    pub fn set<C: CallbackId<Module = M>>(
        &mut self,
        callback: C,
        f: C::Fn,
    ) -> Result<(), SlotError> {
        let _ = callback;
        self.write(C::ID, C::erase(f))
    }

    /// Reset the slot of callback `C` to no callback.
    ///
    /// # Errors
    ///
    /// See [`SlotError`].
    pub fn clear<C: CallbackId<Module = M>>(&mut self, callback: C) -> Result<(), SlotError> {
        let _ = callback;
        self.write(C::ID, None)
    }

    fn write(&mut self, id: u32, f: ffi::NvtxFunctionPointer) -> Result<(), SlotError> {
        if id == 0 || id > self.max_id {
            return Err(SlotError::OutOfBounds {
                id,
                max_id: self.max_id,
            });
        }
        // SAFETY: `id` is bounds-checked above and NVTX keeps the table alive
        // for the rest of the process.
        let slot_ptr = unsafe { self.table.add(id as usize) };
        // SAFETY: `slot_ptr` is within the table per the bounds check above.
        let slot = unsafe { *slot_ptr };
        if slot.is_null() {
            return Err(SlotError::NullSlot { id });
        }
        // SAFETY: `slot` is non-null and points to a callback slot owned by
        // NVTX for the rest of the process.
        unsafe { *slot = f };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::tools::callback::core as core_cb;
    use crate::tools::callback::core2 as core2_cb;
    use crate::tools::{Core, Core2};

    unsafe extern "C" fn noop_mark_a(_message: *const ::core::ffi::c_char) {}

    fn function_table<M: CallbackModule>(
        slots: &mut [ffi::NvtxFunctionPointer],
        pointers: &mut Vec<*mut ffi::NvtxFunctionPointer>,
        max_id: u32,
    ) -> FunctionTable<M> {
        *pointers = slots.iter_mut().map(::core::ptr::from_mut).collect();
        FunctionTable {
            table: pointers.as_mut_ptr(),
            max_id,
            _module: PhantomData,
        }
    }

    #[test]
    fn set_writes_bounds_checked_slots() {
        let mut slots: [ffi::NvtxFunctionPointer; 16] = [None; 16];
        let mut pointers = Vec::new();
        let mut table = function_table::<Core>(&mut slots, &mut pointers, 15);

        table.set(core_cb::MarkA, Some(noop_mark_a)).unwrap();
        assert!(slots[core_cb::MarkA::ID as usize].is_some());
        table.clear(core_cb::MarkA).unwrap();
        assert!(slots[core_cb::MarkA::ID as usize].is_none());
        // The highest valid callback identifier is accepted.
        table.set(core_cb::NameOsThreadW, None).unwrap();
    }

    #[test]
    fn set_rejects_out_of_bounds_ids() {
        let mut slots: [ffi::NvtxFunctionPointer; 15] = [None; 15];
        let mut pointers = Vec::new();
        // A table from an older NVTX version without the last CORE slot.
        let mut table = function_table::<Core>(&mut slots, &mut pointers, 14);

        assert_eq!(
            table.set(core_cb::NameOsThreadW, None),
            Err(SlotError::OutOfBounds { id: 15, max_id: 14 })
        );
        assert_eq!(
            table.write(0, None),
            Err(SlotError::OutOfBounds { id: 0, max_id: 14 })
        );
    }

    #[test]
    fn set_rejects_null_slot_pointers() {
        let mut slots: [ffi::NvtxFunctionPointer; 16] = [None; 16];
        let mut pointers = Vec::new();
        let mut table = function_table::<Core2>(&mut slots, &mut pointers, 15);
        pointers[core2_cb::DomainMarkEx::ID as usize] = ::core::ptr::null_mut();

        assert_eq!(
            table.set(core2_cb::DomainMarkEx, None),
            Err(SlotError::NullSlot { id: 1 })
        );
    }
}
