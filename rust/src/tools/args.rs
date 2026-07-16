// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Decoded views of NVTX event and resource attributes.
//!
//! NVTX attribute structs carry an explicit `size` field so applications built
//! against older NVTX headers can pass smaller structs. The decoders here only
//! read the fields the caller's declared size covers.

use core::ffi::{c_char, CStr};
use core::mem::{offset_of, size_of};

use widestring::WideCStr;

use crate::sys::ffi;
use crate::tools::RegisteredStringId;
use crate::{Color, Payload};

/// A decoded message from NVTX event or resource attributes.
///
/// Borrowed variants are only valid for the duration of the callback.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum MessageView<'a> {
    /// An ASCII message.
    Ascii(&'a CStr),
    /// A Unicode (wide-string) message.
    Unicode(&'a WideCStr),
    /// A registered string, previously observed by
    /// [`Subscriber::domain_register_string_ascii`] or
    /// [`Subscriber::domain_register_string_unicode`].
    ///
    /// [`Subscriber::domain_register_string_ascii`]: crate::tools::Subscriber::domain_register_string_ascii
    /// [`Subscriber::domain_register_string_unicode`]: crate::tools::Subscriber::domain_register_string_unicode
    Registered(RegisteredStringId),
}

/// Decoded `nvtxEventAttributes_t`: the attributes of a mark or range event.
///
/// Fields the application did not provide (or provided with an unknown type
/// discriminator) are `None`. Borrowed fields are only valid for the duration
/// of the callback.
#[derive(Debug, Clone, Copy, Default)]
pub struct EventArgs<'a> {
    /// The `version` field declared by the application (`0` if absent).
    pub version: u16,
    /// The event's category identifier; `None` when absent or `0`.
    pub category: Option<u32>,
    /// The event's color; `None` when absent or of unknown type.
    pub color: Option<Color>,
    /// The event's payload value; `None` when absent or of unknown type.
    pub payload: Option<Payload>,
    /// The event's message; `None` when absent, null, or of unknown type.
    pub message: Option<MessageView<'a>>,
}

/// Decoded `nvtxResourceAttributes_t`: the attributes of a named resource.
///
/// Borrowed fields are only valid for the duration of the callback.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResourceArgs<'a> {
    /// The `version` field declared by the application (`0` if absent).
    pub version: u16,
    /// The raw resource identifier type (see
    /// [`resource_type`](crate::sys::resource_type)); `0` if absent.
    pub identifier_type: i32,
    /// The raw resource identifier bits (pointer identifiers surface as their
    /// address); `0` if absent.
    pub identifier: u64,
    /// The resource's name; `None` when absent, null, or of unknown type.
    pub message: Option<MessageView<'a>>,
}

/// Decoded `nvtxSyncUserAttributes_t`: the attributes of a user-defined
/// synchronization object.
///
/// Borrowed fields are only valid for the duration of the callback.
#[derive(Debug, Clone, Copy, Default)]
pub struct SyncUserArgs<'a> {
    /// The `version` field declared by the application (`0` if absent).
    pub version: u16,
    /// The synchronization object's name; `None` when absent, null, or of
    /// unknown type.
    pub message: Option<MessageView<'a>>,
}

/// Borrow a NUL-terminated C string, or `None` when `ptr` is null.
///
/// # Safety
///
/// A non-null `ptr` must point to a NUL-terminated string valid for `'a`.
pub(super) unsafe fn opt_cstr<'a>(ptr: *const c_char) -> Option<&'a CStr> {
    if ptr.is_null() {
        None
    } else {
        // SAFETY: `ptr` is non-null and the caller guarantees it points to a
        // NUL-terminated string valid for `'a`.
        Some(unsafe { CStr::from_ptr(ptr) })
    }
}

/// Borrow a NUL-terminated wide C string, or `None` when `ptr` is null.
///
/// # Safety
///
/// A non-null `ptr` must point to a NUL-terminated wide string valid for `'a`.
pub(super) unsafe fn opt_wide<'a>(ptr: *const ffi::wchar_t) -> Option<&'a WideCStr> {
    if ptr.is_null() {
        None
    } else {
        // SAFETY: `ptr` is non-null and the caller guarantees it points to a
        // NUL-terminated wide string valid for `'a`; `wchar_t` has the width
        // of the platform's wide character.
        Some(unsafe { WideCStr::from_ptr_str(ptr.cast()) })
    }
}

/// `true` when a struct of `len` bytes contains the field at `offset` of
/// `size` bytes.
fn has_field(len: usize, offset: usize, size: usize) -> bool {
    len >= offset + size
}

/// Copy at most `size_of::<T>()` of the `declared` bytes at `source` over
/// `target`, leaving the rest of `target` untouched.
///
/// # Safety
///
/// `source` must be valid for reads of `min(declared, size_of::<T>())` bytes.
unsafe fn copy_declared<T>(source: *const T, declared: usize, target: &mut T) -> usize {
    let len = declared.min(size_of::<T>());
    // SAFETY: `source` is valid for `len` reads per this function's contract
    // and `target` is a valid, exclusive destination of at least `len` bytes.
    unsafe {
        core::ptr::copy_nonoverlapping(
            source.cast::<u8>(),
            core::ptr::from_mut(target).cast(),
            len,
        );
    }
    len
}

/// Read the `u16` `size` field every NVTX attributes struct starts with.
///
/// # Safety
///
/// `ptr` must be valid for reads covering the struct's `version` and `size`
/// header fields.
unsafe fn declared_size<T>(ptr: *const T, size_offset: usize) -> usize {
    // SAFETY: `size_offset` is within the always-present header per this
    // function's contract.
    let size_ptr = unsafe { ptr.cast::<u8>().add(size_offset) };
    // SAFETY: `size_ptr` points to the `u16` `size` field; an unaligned read
    // is used since `ptr`'s alignment is not guaranteed by the ABI.
    usize::from(unsafe { size_ptr.cast::<u16>().read_unaligned() })
}

fn decode_color(color_type: i32, color: u32) -> Option<Color> {
    if color_type == i32::from(ffi::nvtxColorType_t::NVTX_COLOR_ARGB) {
        let [a, r, g, b] = color.to_be_bytes();
        Some(Color::new(r, g, b, a))
    } else {
        None
    }
}

fn decode_payload(
    payload_type: i32,
    value: ffi::nvtxEventAttributes_v2_payload_t,
) -> Option<Payload> {
    // CAST: `nvtxPayloadType_t` is a `repr(u32)` enum with values that fit in
    // `i32`, compared against the struct's `i32` type discriminator.
    const UNSIGNED_INT64: i32 = ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 as i32;
    const INT64: i32 = ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_INT64 as i32;
    const DOUBLE: i32 = ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_DOUBLE as i32;
    const UNSIGNED_INT32: i32 = ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_UNSIGNED_INT32 as i32;
    const INT32: i32 = ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_INT32 as i32;
    const FLOAT: i32 = ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_FLOAT as i32;

    match payload_type {
        // SAFETY: `payload_type` declares the matching union member active.
        UNSIGNED_INT64 => Some(Payload::Uint64(unsafe { value.ullValue })),
        // SAFETY: See above.
        INT64 => Some(Payload::Int64(unsafe { value.llValue })),
        // SAFETY: See above.
        DOUBLE => Some(Payload::Double(unsafe { value.dValue })),
        // SAFETY: See above.
        UNSIGNED_INT32 => Some(Payload::Uint32(unsafe { value.uiValue })),
        // SAFETY: See above.
        INT32 => Some(Payload::Int32(unsafe { value.iValue })),
        // SAFETY: See above.
        FLOAT => Some(Payload::Float(unsafe { value.fValue })),
        _ => None,
    }
}

/// Decode a message union by its type discriminator.
///
/// # Safety
///
/// String pointers in `value` (when declared active by `message_type`) must be
/// NUL-terminated and valid for `'a`.
unsafe fn decode_message<'a>(
    message_type: i32,
    value: ffi::nvtxMessageValue_t,
) -> Option<MessageView<'a>> {
    // CAST: `nvtxMessageType_t` is a `repr(u32)` enum with values that fit in
    // `i32`, compared against the struct's `i32` type discriminator.
    const ASCII: i32 = ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII as i32;
    const UNICODE: i32 = ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_UNICODE as i32;
    const REGISTERED: i32 = ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_REGISTERED as i32;

    match message_type {
        ASCII => {
            // SAFETY: `message_type` declares the `ascii` member active.
            let ptr = unsafe { value.ascii };
            // SAFETY: The caller guarantees the string is valid for `'a`.
            unsafe { opt_cstr(ptr) }.map(MessageView::Ascii)
        }
        UNICODE => {
            // SAFETY: `message_type` declares the `unicode` member active.
            let ptr = unsafe { value.unicode };
            // SAFETY: The caller guarantees the string is valid for `'a`.
            unsafe { opt_wide(ptr) }.map(MessageView::Unicode)
        }
        REGISTERED => {
            // SAFETY: `message_type` declares the `registered` member active.
            let handle = unsafe { value.registered };
            // CAST: registered-string handles are opaque values; capture the
            // pointer's address bits.
            (!handle.is_null())
                .then(|| MessageView::Registered(RegisteredStringId::new(handle as usize as u64)))
        }
        _ => None,
    }
}

impl EventArgs<'_> {
    /// Decode event attributes, reading only the fields covered by the
    /// struct's declared `size`. A null pointer decodes to the default (all
    /// `None`) value.
    ///
    /// # Safety
    ///
    /// A non-null `attrib` must point to an `nvtxEventAttributes_t` valid for
    /// reads of its declared `size`, whose message string pointers (if any)
    /// are valid for `'a`.
    #[must_use]
    pub unsafe fn decode(attrib: *const ffi::nvtxEventAttributes_t) -> Self {
        const EMPTY: ffi::nvtxEventAttributes_v2 = ffi::nvtxEventAttributes_v2 {
            version: 0,
            size: 0,
            category: 0,
            colorType: 0,
            color: 0,
            payloadType: 0,
            reserved0: 0,
            payload: ffi::nvtxEventAttributes_v2_payload_t { ullValue: 0 },
            messageType: 0,
            message: ffi::nvtxMessageValue_t {
                ascii: core::ptr::null(),
            },
        };

        if attrib.is_null() {
            return Self::default();
        }
        // SAFETY: `attrib` is non-null and the `version`/`size` header is
        // always present.
        let declared =
            unsafe { declared_size(attrib, offset_of!(ffi::nvtxEventAttributes_v2, size)) };
        let mut raw = EMPTY;
        // SAFETY: The caller guarantees `attrib` is valid for reads of
        // `declared` bytes.
        let len = unsafe { copy_declared(attrib, declared, &mut raw) };

        let category = (has_field(
            len,
            offset_of!(ffi::nvtxEventAttributes_v2, category),
            size_of::<u32>(),
        ) && raw.category != 0)
            .then_some(raw.category);
        let color = has_field(
            len,
            offset_of!(ffi::nvtxEventAttributes_v2, color),
            size_of::<u32>(),
        )
        .then(|| decode_color(raw.colorType, raw.color))
        .flatten();
        let payload = has_field(
            len,
            offset_of!(ffi::nvtxEventAttributes_v2, payload),
            size_of::<ffi::nvtxEventAttributes_v2_payload_t>(),
        )
        .then(|| decode_payload(raw.payloadType, raw.payload))
        .flatten();
        let message = if has_field(
            len,
            offset_of!(ffi::nvtxEventAttributes_v2, message),
            size_of::<ffi::nvtxMessageValue_t>(),
        ) {
            // SAFETY: The caller guarantees message strings are valid for `'a`.
            unsafe { decode_message(raw.messageType, raw.message) }
        } else {
            None
        };

        Self {
            version: raw.version,
            category,
            color,
            payload,
            message,
        }
    }
}

impl ResourceArgs<'_> {
    /// Decode resource attributes, reading only the fields covered by the
    /// struct's declared `size`. A null pointer decodes to the default value.
    ///
    /// # Safety
    ///
    /// A non-null `attrib` must point to an `nvtxResourceAttributes_t` valid
    /// for reads of its declared `size`, whose message string pointers (if
    /// any) are valid for `'a`.
    #[must_use]
    pub unsafe fn decode(attrib: *const ffi::nvtxResourceAttributes_t) -> Self {
        const EMPTY: ffi::nvtxResourceAttributes_v0 = ffi::nvtxResourceAttributes_v0 {
            version: 0,
            size: 0,
            identifierType: 0,
            identifier: ffi::nvtxResourceAttributes_v0_identifier_t { ullValue: 0 },
            messageType: 0,
            message: ffi::nvtxMessageValue_t {
                ascii: core::ptr::null(),
            },
        };

        if attrib.is_null() {
            return Self::default();
        }
        // SAFETY: `attrib` is non-null and the `version`/`size` header is
        // always present.
        let declared =
            unsafe { declared_size(attrib, offset_of!(ffi::nvtxResourceAttributes_v0, size)) };
        let mut raw = EMPTY;
        // SAFETY: The caller guarantees `attrib` is valid for reads of
        // `declared` bytes.
        let len = unsafe { copy_declared(attrib, declared, &mut raw) };

        let identifier_type = if has_field(
            len,
            offset_of!(ffi::nvtxResourceAttributes_v0, identifierType),
            size_of::<i32>(),
        ) {
            raw.identifierType
        } else {
            0
        };
        let identifier = if has_field(
            len,
            offset_of!(ffi::nvtxResourceAttributes_v0, identifier),
            size_of::<ffi::nvtxResourceAttributes_v0_identifier_t>(),
        ) {
            // SAFETY: The identifier union is captured as its raw bits.
            unsafe { raw.identifier.ullValue }
        } else {
            0
        };
        let message = if has_field(
            len,
            offset_of!(ffi::nvtxResourceAttributes_v0, message),
            size_of::<ffi::nvtxMessageValue_t>(),
        ) {
            // SAFETY: The caller guarantees message strings are valid for `'a`.
            unsafe { decode_message(raw.messageType, raw.message) }
        } else {
            None
        };

        Self {
            version: raw.version,
            identifier_type,
            identifier,
            message,
        }
    }
}

impl SyncUserArgs<'_> {
    /// Decode user-defined synchronization attributes, reading only the fields
    /// covered by the struct's declared `size`. A null pointer decodes to the
    /// default value.
    ///
    /// # Safety
    ///
    /// A non-null `attrib` must point to an `nvtxSyncUserAttributes_t` valid
    /// for reads of its declared `size`, whose message string pointers (if
    /// any) are valid for `'a`.
    #[must_use]
    pub unsafe fn decode(attrib: *const ffi::nvtxSyncUserAttributes_t) -> Self {
        const EMPTY: ffi::nvtxSyncUserAttributes_v0 = ffi::nvtxSyncUserAttributes_v0 {
            version: 0,
            size: 0,
            messageType: 0,
            message: ffi::nvtxMessageValue_t {
                ascii: core::ptr::null(),
            },
        };

        if attrib.is_null() {
            return Self::default();
        }
        // SAFETY: `attrib` is non-null and the `version`/`size` header is
        // always present.
        let declared =
            unsafe { declared_size(attrib, offset_of!(ffi::nvtxSyncUserAttributes_v0, size)) };
        let mut raw = EMPTY;
        // SAFETY: The caller guarantees `attrib` is valid for reads of
        // `declared` bytes.
        let len = unsafe { copy_declared(attrib, declared, &mut raw) };

        let message = if has_field(
            len,
            offset_of!(ffi::nvtxSyncUserAttributes_v0, message),
            size_of::<ffi::nvtxMessageValue_t>(),
        ) {
            // SAFETY: The caller guarantees message strings are valid for `'a`.
            unsafe { decode_message(raw.messageType, raw.message) }
        } else {
            None
        };

        Self {
            version: raw.version,
            message,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::ffi::CString;

    use widestring::WideCString;

    use super::*;

    // CAST: field offsets and sizes fit in `u16` for these small structs.
    const EVENT_ATTRIBUTES_SIZE: u16 = size_of::<ffi::nvtxEventAttributes_v2>() as u16;

    fn event_attributes(message: &CString) -> ffi::nvtxEventAttributes_v2 {
        ffi::nvtxEventAttributes_v2 {
            version: 2,
            size: EVENT_ATTRIBUTES_SIZE,
            category: 42,
            colorType: i32::from(ffi::nvtxColorType_t::NVTX_COLOR_ARGB),
            color: 0x8011_2233,
            payloadType: i32::from(ffi::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_UNSIGNED_INT64),
            reserved0: 0,
            payload: ffi::nvtxEventAttributes_v2_payload_t {
                ullValue: 0xCAFE_F00D,
            },
            messageType: i32::from(ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII),
            message: ffi::nvtxMessageValue_t {
                ascii: message.as_ptr(),
            },
        }
    }

    #[test]
    fn decodes_full_event_attributes() {
        let message = CString::new("event").unwrap();
        let attrib = event_attributes(&message);
        // SAFETY: `attrib` is a valid attributes struct backed by `message`.
        let args = unsafe { EventArgs::decode(&attrib) };

        assert_eq!(args.version, 2);
        assert_eq!(args.category, Some(42));
        assert_eq!(args.color, Some(Color::new(0x11, 0x22, 0x33, 0x80)));
        assert!(matches!(args.payload, Some(Payload::Uint64(0xCAFE_F00D))));
        assert!(matches!(args.message, Some(MessageView::Ascii(s)) if s.to_str() == Ok("event")));
    }

    #[test]
    fn decodes_unicode_and_registered_messages() {
        let wide = WideCString::from_str("wide").unwrap();
        let mut attrib = event_attributes(&CString::new("unused").unwrap());
        attrib.messageType = i32::from(ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_UNICODE);
        attrib.message = ffi::nvtxMessageValue_t {
            unicode: wide.as_ptr().cast(),
        };
        // SAFETY: `attrib` is a valid attributes struct backed by `wide`.
        let args = unsafe { EventArgs::decode(&attrib) };
        assert!(
            matches!(args.message, Some(MessageView::Unicode(s)) if s.to_string().unwrap() == "wide")
        );

        attrib.messageType = i32::from(ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_REGISTERED);
        attrib.message = ffi::nvtxMessageValue_t {
            // CAST: a synthesized registered-string handle value.
            registered: 7_usize as ffi::nvtxStringHandle_t,
        };
        // SAFETY: `attrib` is a valid attributes struct.
        let args = unsafe { EventArgs::decode(&attrib) };
        assert!(matches!(
            args.message,
            Some(MessageView::Registered(id)) if id.raw() == 7
        ));
    }

    #[test]
    fn ignores_fields_beyond_declared_size() {
        let message = CString::new("event").unwrap();
        let mut attrib = event_attributes(&message);
        // Declare only the fields up to and including `color` as present.
        // CAST: field offsets fit in `u16`.
        attrib.size = (offset_of!(ffi::nvtxEventAttributes_v2, color) + size_of::<u32>()) as u16;
        // SAFETY: `attrib` is a valid attributes struct.
        let args = unsafe { EventArgs::decode(&attrib) };

        assert_eq!(args.category, Some(42));
        assert!(args.color.is_some());
        assert!(args.payload.is_none());
        assert!(args.message.is_none());
    }

    #[test]
    fn decodes_null_zero_sized_and_unknown_typed_attributes_to_default() {
        // SAFETY: a null pointer is decoded to the default value.
        let args = unsafe { EventArgs::decode(core::ptr::null()) };
        assert!(args.category.is_none());

        let message = CString::new("event").unwrap();
        let mut attrib = event_attributes(&message);
        attrib.size = 0;
        // SAFETY: `attrib` is a valid attributes struct.
        let args = unsafe { EventArgs::decode(&attrib) };
        assert_eq!(args.version, 0);
        assert!(args.category.is_none());
        assert!(args.message.is_none());

        let mut attrib = event_attributes(&message);
        attrib.colorType = 99;
        attrib.payloadType = 99;
        attrib.messageType = 99;
        // SAFETY: `attrib` is a valid attributes struct.
        let args = unsafe { EventArgs::decode(&attrib) };
        assert!(args.color.is_none());
        assert!(args.payload.is_none());
        assert!(args.message.is_none());
    }

    #[test]
    fn decodes_null_message_pointer_to_none() {
        let message = CString::new("unused").unwrap();
        let mut attrib = event_attributes(&message);
        attrib.message = ffi::nvtxMessageValue_t {
            ascii: core::ptr::null(),
        };
        // SAFETY: `attrib` is a valid attributes struct.
        let args = unsafe { EventArgs::decode(&attrib) };
        assert!(args.message.is_none());
    }

    #[test]
    fn decodes_sync_user_attributes() {
        let name = CString::new("mutex").unwrap();
        let mut attrib = ffi::nvtxSyncUserAttributes_v0 {
            version: 0,
            // CAST: the struct size fits in `u16`.
            size: size_of::<ffi::nvtxSyncUserAttributes_v0>() as u16,
            messageType: i32::from(ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII),
            message: ffi::nvtxMessageValue_t {
                ascii: name.as_ptr(),
            },
        };
        // SAFETY: `attrib` is a valid sync-user-attributes struct backed by
        // `name`.
        let args = unsafe { SyncUserArgs::decode(&attrib) };
        assert!(matches!(args.message, Some(MessageView::Ascii(s)) if s.to_str() == Ok("mutex")));

        // Truncate the declared size to the header: the message is ignored.
        attrib.size = 4;
        // SAFETY: `attrib` is a valid sync-user-attributes struct.
        let args = unsafe { SyncUserArgs::decode(&attrib) };
        assert!(args.message.is_none());

        // SAFETY: a null pointer is decoded to the default value.
        let args = unsafe { SyncUserArgs::decode(core::ptr::null()) };
        assert!(args.message.is_none());
    }

    #[test]
    fn decodes_resource_attributes() {
        let name = CString::new("resource").unwrap();
        let attrib = ffi::nvtxResourceAttributes_v0 {
            version: 0,
            // CAST: the struct size fits in `u16`.
            size: size_of::<ffi::nvtxResourceAttributes_v0>() as u16,
            identifierType: 65538,
            identifier: ffi::nvtxResourceAttributes_v0_identifier_t { ullValue: 0x1234 },
            messageType: i32::from(ffi::nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII),
            message: ffi::nvtxMessageValue_t {
                ascii: name.as_ptr(),
            },
        };
        // SAFETY: `attrib` is a valid resource-attributes struct backed by
        // `name`.
        let args = unsafe { ResourceArgs::decode(&attrib) };

        assert_eq!(args.identifier_type, 65538);
        assert_eq!(args.identifier, 0x1234);
        assert!(
            matches!(args.message, Some(MessageView::Ascii(s)) if s.to_str() == Ok("resource"))
        );
    }
}
