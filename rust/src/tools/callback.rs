// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Typed identifiers for NVTX callback modules and callback slots.
//!
//! Every NVTX callback slot is identified by a module (for example
//! [`Core`] or [`Core2`]) and a callback identifier within that module. The
//! zero-sized marker types in [`core`] and [`crate::tools::callback::core2`]
//! tie each slot to its exact ABI function-pointer type, so
//! [`FunctionTable::set`](crate::tools::FunctionTable::set) is type-checked at
//! compile time.

use crate::sys::ffi;

mod private {
    /// Seals [`super::CallbackModule`] and [`super::CallbackId`].
    pub trait Sealed {}
}

/// An NVTX callback module, identifying one per-module function table.
///
/// This trait is sealed and implemented by the module marker types in this
/// module.
pub trait CallbackModule: private::Sealed {
    /// The `NvtxCallbackModule` value identifying this module's function table.
    const MODULE: ffi::NvtxCallbackModule;
}

/// The CORE callback module: global (default-domain) marks, ranges, and
/// category/thread naming.
#[derive(Debug, Clone, Copy)]
pub struct Core;

impl private::Sealed for Core {}

impl CallbackModule for Core {
    const MODULE: ffi::NvtxCallbackModule = ffi::NvtxCallbackModule::NVTX_CB_MODULE_CORE;
}

/// The CORE2 callback module: domain-scoped marks, ranges, resources,
/// registered strings, and domain management.
#[derive(Debug, Clone, Copy)]
pub struct Core2;

impl private::Sealed for Core2 {}

impl CallbackModule for Core2 {
    const MODULE: ffi::NvtxCallbackModule = ffi::NvtxCallbackModule::NVTX_CB_MODULE_CORE2;
}

/// The CUDA callback module: CUDA driver resource naming.
#[derive(Debug, Clone, Copy)]
pub struct Cuda;

impl private::Sealed for Cuda {}

impl CallbackModule for Cuda {
    const MODULE: ffi::NvtxCallbackModule = ffi::NvtxCallbackModule::NVTX_CB_MODULE_CUDA;
}

/// The CUDART callback module: CUDA runtime resource naming.
#[derive(Debug, Clone, Copy)]
pub struct CudaRt;

impl private::Sealed for CudaRt {}

impl CallbackModule for CudaRt {
    const MODULE: ffi::NvtxCallbackModule = ffi::NvtxCallbackModule::NVTX_CB_MODULE_CUDART;
}

/// The OPENCL callback module: `OpenCL` resource naming.
#[derive(Debug, Clone, Copy)]
pub struct OpenCl;

impl private::Sealed for OpenCl {}

impl CallbackModule for OpenCl {
    const MODULE: ffi::NvtxCallbackModule = ffi::NvtxCallbackModule::NVTX_CB_MODULE_OPENCL;
}

/// The SYNC callback module: user-defined synchronization objects.
#[derive(Debug, Clone, Copy)]
pub struct Sync;

impl private::Sealed for Sync {}

impl CallbackModule for Sync {
    const MODULE: ffi::NvtxCallbackModule = ffi::NvtxCallbackModule::NVTX_CB_MODULE_SYNC;
}

/// An NVTX callback slot within a [`CallbackModule`]'s function table.
///
/// This trait is sealed and implemented by the zero-sized marker types in
/// [`core`] and [`crate::tools::callback::core2`].
pub trait CallbackId: private::Sealed + Copy {
    /// The module whose function table holds this callback slot.
    type Module: CallbackModule;
    /// The exact ABI function-pointer type stored in this callback slot.
    type Fn;
    /// The callback identifier (index into the module's function table).
    const ID: u32;
    /// Type-erase `f` for storage in the NVTX function table.
    fn erase(f: Self::Fn) -> ffi::NvtxFunctionPointer;
}

macro_rules! define_callback_ids {
    ($(
        $(#[$mod_meta:meta])*
        mod $mod_name:ident ($module:ident, $cbid_enum:ident) {
            $($(#[$meta:meta])* $name:ident = $variant:ident => $fn_type:ident,)+
        }
    )+) => {
        $(
            $(#[$mod_meta])*
            pub mod $mod_name {
                use super::{private, CallbackId, $module};
                use crate::sys::ffi;

                $(
                    $(#[$meta])*
                    #[derive(Debug, Clone, Copy)]
                    pub struct $name;

                    impl private::Sealed for $name {}

                    impl CallbackId for $name {
                        type Module = $module;
                        type Fn = ffi::$fn_type;
                        // CAST: `NvtxCallbackId*` are `repr(u32)` enums; the
                        // discriminant is the function-table index.
                        const ID: u32 = ffi::$cbid_enum::$variant as u32;

                        fn erase(f: Self::Fn) -> ffi::NvtxFunctionPointer {
                            // SAFETY: Both types are an `Option` of an
                            // `unsafe extern "C"` function pointer with
                            // identical (pointer-sized, null-niche) layout.
                            // NVTX stores slots type-erased and casts back to
                            // this slot's documented signature before calling.
                            unsafe {
                                ::core::mem::transmute::<
                                    ffi::$fn_type,
                                    ffi::NvtxFunctionPointer,
                                >(f)
                            }
                        }
                    }
                )+
            }
        )+
    };
}

define_callback_ids! {
    /// Callback slots of the [`Core`] module.
    mod core (Core, NvtxCallbackIdCore) {
        /// `nvtxMarkEx` slot.
        MarkEx = NVTX_CBID_CORE_MarkEx => nvtxMarkEx_impl_fntype,
        /// `nvtxMarkA` slot.
        MarkA = NVTX_CBID_CORE_MarkA => nvtxMarkA_impl_fntype,
        /// `nvtxMarkW` slot.
        MarkW = NVTX_CBID_CORE_MarkW => nvtxMarkW_impl_fntype,
        /// `nvtxRangeStartEx` slot.
        RangeStartEx = NVTX_CBID_CORE_RangeStartEx => nvtxRangeStartEx_impl_fntype,
        /// `nvtxRangeStartA` slot.
        RangeStartA = NVTX_CBID_CORE_RangeStartA => nvtxRangeStartA_impl_fntype,
        /// `nvtxRangeStartW` slot.
        RangeStartW = NVTX_CBID_CORE_RangeStartW => nvtxRangeStartW_impl_fntype,
        /// `nvtxRangeEnd` slot.
        RangeEnd = NVTX_CBID_CORE_RangeEnd => nvtxRangeEnd_impl_fntype,
        /// `nvtxRangePushEx` slot.
        RangePushEx = NVTX_CBID_CORE_RangePushEx => nvtxRangePushEx_impl_fntype,
        /// `nvtxRangePushA` slot.
        RangePushA = NVTX_CBID_CORE_RangePushA => nvtxRangePushA_impl_fntype,
        /// `nvtxRangePushW` slot.
        RangePushW = NVTX_CBID_CORE_RangePushW => nvtxRangePushW_impl_fntype,
        /// `nvtxRangePop` slot.
        RangePop = NVTX_CBID_CORE_RangePop => nvtxRangePop_impl_fntype,
        /// `nvtxNameCategoryA` slot.
        NameCategoryA = NVTX_CBID_CORE_NameCategoryA => nvtxNameCategoryA_impl_fntype,
        /// `nvtxNameCategoryW` slot.
        NameCategoryW = NVTX_CBID_CORE_NameCategoryW => nvtxNameCategoryW_impl_fntype,
        /// `nvtxNameOsThreadA` slot.
        NameOsThreadA = NVTX_CBID_CORE_NameOsThreadA => nvtxNameOsThreadA_impl_fntype,
        /// `nvtxNameOsThreadW` slot.
        NameOsThreadW = NVTX_CBID_CORE_NameOsThreadW => nvtxNameOsThreadW_impl_fntype,
    }
    /// Callback slots of the [`Core2`] module.
    mod core2 (Core2, NvtxCallbackIdCore2) {
        /// `nvtxDomainMarkEx` slot.
        DomainMarkEx = NVTX_CBID_CORE2_DomainMarkEx => nvtxDomainMarkEx_impl_fntype,
        /// `nvtxDomainRangeStartEx` slot.
        DomainRangeStartEx =
            NVTX_CBID_CORE2_DomainRangeStartEx => nvtxDomainRangeStartEx_impl_fntype,
        /// `nvtxDomainRangeEnd` slot.
        DomainRangeEnd = NVTX_CBID_CORE2_DomainRangeEnd => nvtxDomainRangeEnd_impl_fntype,
        /// `nvtxDomainRangePushEx` slot.
        DomainRangePushEx = NVTX_CBID_CORE2_DomainRangePushEx => nvtxDomainRangePushEx_impl_fntype,
        /// `nvtxDomainRangePop` slot.
        DomainRangePop = NVTX_CBID_CORE2_DomainRangePop => nvtxDomainRangePop_impl_fntype,
        /// `nvtxDomainResourceCreate` slot.
        DomainResourceCreate =
            NVTX_CBID_CORE2_DomainResourceCreate => nvtxDomainResourceCreate_impl_fntype,
        /// `nvtxDomainResourceDestroy` slot.
        DomainResourceDestroy =
            NVTX_CBID_CORE2_DomainResourceDestroy => nvtxDomainResourceDestroy_impl_fntype,
        /// `nvtxDomainNameCategoryA` slot.
        DomainNameCategoryA =
            NVTX_CBID_CORE2_DomainNameCategoryA => nvtxDomainNameCategoryA_impl_fntype,
        /// `nvtxDomainNameCategoryW` slot.
        DomainNameCategoryW =
            NVTX_CBID_CORE2_DomainNameCategoryW => nvtxDomainNameCategoryW_impl_fntype,
        /// `nvtxDomainRegisterStringA` slot.
        DomainRegisterStringA =
            NVTX_CBID_CORE2_DomainRegisterStringA => nvtxDomainRegisterStringA_impl_fntype,
        /// `nvtxDomainRegisterStringW` slot.
        DomainRegisterStringW =
            NVTX_CBID_CORE2_DomainRegisterStringW => nvtxDomainRegisterStringW_impl_fntype,
        /// `nvtxDomainCreateA` slot.
        DomainCreateA = NVTX_CBID_CORE2_DomainCreateA => nvtxDomainCreateA_impl_fntype,
        /// `nvtxDomainCreateW` slot.
        DomainCreateW = NVTX_CBID_CORE2_DomainCreateW => nvtxDomainCreateW_impl_fntype,
        /// `nvtxDomainDestroy` slot.
        DomainDestroy = NVTX_CBID_CORE2_DomainDestroy => nvtxDomainDestroy_impl_fntype,
        /// `nvtxInitialize` slot.
        Initialize = NVTX_CBID_CORE2_Initialize => nvtxInitialize_impl_fntype,
    }
    /// Callback slots of the [`Cuda`] module.
    mod cuda (Cuda, NvtxCallbackIdCuda) {
        /// `nvtxNameCuDeviceA` slot.
        NameCuDeviceA = NVTX_CBID_CUDA_NameCuDeviceA => nvtxNameCuDeviceA_fakeimpl_fntype,
        /// `nvtxNameCuDeviceW` slot.
        NameCuDeviceW = NVTX_CBID_CUDA_NameCuDeviceW => nvtxNameCuDeviceW_fakeimpl_fntype,
        /// `nvtxNameCuContextA` slot.
        NameCuContextA = NVTX_CBID_CUDA_NameCuContextA => nvtxNameCuContextA_fakeimpl_fntype,
        /// `nvtxNameCuContextW` slot.
        NameCuContextW = NVTX_CBID_CUDA_NameCuContextW => nvtxNameCuContextW_fakeimpl_fntype,
        /// `nvtxNameCuStreamA` slot.
        NameCuStreamA = NVTX_CBID_CUDA_NameCuStreamA => nvtxNameCuStreamA_fakeimpl_fntype,
        /// `nvtxNameCuStreamW` slot.
        NameCuStreamW = NVTX_CBID_CUDA_NameCuStreamW => nvtxNameCuStreamW_fakeimpl_fntype,
        /// `nvtxNameCuEventA` slot.
        NameCuEventA = NVTX_CBID_CUDA_NameCuEventA => nvtxNameCuEventA_fakeimpl_fntype,
        /// `nvtxNameCuEventW` slot.
        NameCuEventW = NVTX_CBID_CUDA_NameCuEventW => nvtxNameCuEventW_fakeimpl_fntype,
    }
    /// Callback slots of the [`CudaRt`] module.
    mod cudart (CudaRt, NvtxCallbackIdCudaRt) {
        /// `nvtxNameCudaDeviceA` slot.
        NameCudaDeviceA = NVTX_CBID_CUDART_NameCudaDeviceA => nvtxNameCudaDeviceA_fakeimpl_fntype,
        /// `nvtxNameCudaDeviceW` slot.
        NameCudaDeviceW = NVTX_CBID_CUDART_NameCudaDeviceW => nvtxNameCudaDeviceW_fakeimpl_fntype,
        /// `nvtxNameCudaStreamA` slot.
        NameCudaStreamA = NVTX_CBID_CUDART_NameCudaStreamA => nvtxNameCudaStreamA_fakeimpl_fntype,
        /// `nvtxNameCudaStreamW` slot.
        NameCudaStreamW = NVTX_CBID_CUDART_NameCudaStreamW => nvtxNameCudaStreamW_fakeimpl_fntype,
        /// `nvtxNameCudaEventA` slot.
        NameCudaEventA = NVTX_CBID_CUDART_NameCudaEventA => nvtxNameCudaEventA_fakeimpl_fntype,
        /// `nvtxNameCudaEventW` slot.
        NameCudaEventW = NVTX_CBID_CUDART_NameCudaEventW => nvtxNameCudaEventW_fakeimpl_fntype,
    }
    /// Callback slots of the [`OpenCl`] module.
    mod opencl (OpenCl, NvtxCallbackIdOpenCL) {
        /// `nvtxNameClDeviceA` slot.
        NameClDeviceA = NVTX_CBID_OPENCL_NameClDeviceA => nvtxNameClDeviceA_fakeimpl_fntype,
        /// `nvtxNameClDeviceW` slot.
        NameClDeviceW = NVTX_CBID_OPENCL_NameClDeviceW => nvtxNameClDeviceW_fakeimpl_fntype,
        /// `nvtxNameClContextA` slot.
        NameClContextA = NVTX_CBID_OPENCL_NameClContextA => nvtxNameClContextA_fakeimpl_fntype,
        /// `nvtxNameClContextW` slot.
        NameClContextW = NVTX_CBID_OPENCL_NameClContextW => nvtxNameClContextW_fakeimpl_fntype,
        /// `nvtxNameClCommandQueueA` slot.
        NameClCommandQueueA =
            NVTX_CBID_OPENCL_NameClCommandQueueA => nvtxNameClCommandQueueA_fakeimpl_fntype,
        /// `nvtxNameClCommandQueueW` slot.
        NameClCommandQueueW =
            NVTX_CBID_OPENCL_NameClCommandQueueW => nvtxNameClCommandQueueW_fakeimpl_fntype,
        /// `nvtxNameClMemObjectA` slot.
        NameClMemObjectA =
            NVTX_CBID_OPENCL_NameClMemObjectA => nvtxNameClMemObjectA_fakeimpl_fntype,
        /// `nvtxNameClMemObjectW` slot.
        NameClMemObjectW =
            NVTX_CBID_OPENCL_NameClMemObjectW => nvtxNameClMemObjectW_fakeimpl_fntype,
        /// `nvtxNameClSamplerA` slot.
        NameClSamplerA = NVTX_CBID_OPENCL_NameClSamplerA => nvtxNameClSamplerA_fakeimpl_fntype,
        /// `nvtxNameClSamplerW` slot.
        NameClSamplerW = NVTX_CBID_OPENCL_NameClSamplerW => nvtxNameClSamplerW_fakeimpl_fntype,
        /// `nvtxNameClProgramA` slot.
        NameClProgramA = NVTX_CBID_OPENCL_NameClProgramA => nvtxNameClProgramA_fakeimpl_fntype,
        /// `nvtxNameClProgramW` slot.
        NameClProgramW = NVTX_CBID_OPENCL_NameClProgramW => nvtxNameClProgramW_fakeimpl_fntype,
        /// `nvtxNameClEventA` slot.
        NameClEventA = NVTX_CBID_OPENCL_NameClEventA => nvtxNameClEventA_fakeimpl_fntype,
        /// `nvtxNameClEventW` slot.
        NameClEventW = NVTX_CBID_OPENCL_NameClEventW => nvtxNameClEventW_fakeimpl_fntype,
    }
    /// Callback slots of the [`Sync`] module.
    mod sync (Sync, NvtxCallbackIdSync) {
        /// `nvtxDomainSyncUserCreate` slot.
        DomainSyncUserCreate =
            NVTX_CBID_SYNC_DomainSyncUserCreate => nvtxDomainSyncUserCreate_fakeimpl_fntype,
        /// `nvtxDomainSyncUserDestroy` slot.
        DomainSyncUserDestroy =
            NVTX_CBID_SYNC_DomainSyncUserDestroy => nvtxDomainSyncUserDestroy_fakeimpl_fntype,
        /// `nvtxDomainSyncUserAcquireStart` slot.
        DomainSyncUserAcquireStart = NVTX_CBID_SYNC_DomainSyncUserAcquireStart
            => nvtxDomainSyncUserAcquireStart_fakeimpl_fntype,
        /// `nvtxDomainSyncUserAcquireFailed` slot.
        DomainSyncUserAcquireFailed = NVTX_CBID_SYNC_DomainSyncUserAcquireFailed
            => nvtxDomainSyncUserAcquireFailed_fakeimpl_fntype,
        /// `nvtxDomainSyncUserAcquireSuccess` slot.
        DomainSyncUserAcquireSuccess = NVTX_CBID_SYNC_DomainSyncUserAcquireSuccess
            => nvtxDomainSyncUserAcquireSuccess_fakeimpl_fntype,
        /// `nvtxDomainSyncUserReleasing` slot.
        DomainSyncUserReleasing = NVTX_CBID_SYNC_DomainSyncUserReleasing
            => nvtxDomainSyncUserReleasing_fakeimpl_fntype,
    }
}
