//! Windows GPU detection through DXCore.
//!
//! `dxcore.dll` and its `DXCoreCreateAdapterFactory` export are looked up at
//! runtime with `libloading`, never linked: a machine without DXCore (or without
//! the export) must still launch Regent, it just has no GPUs to report.
//!
//! All `unsafe` lives here, and every block carries a `// SAFETY:` note.

use std::ffi::c_void;
use std::ptr;

use libloading::Library;
use windows::core::{Interface, GUID, HRESULT};
use windows::Win32::Graphics::DXCore::{
    DXCoreAdapterProperty, DXCoreHardwareID, DedicatedAdapterMemory, DriverDescription, HardwareID,
    IDXCoreAdapter, IDXCoreAdapterFactory, IDXCoreAdapterList, IsHardware, IsIntegrated,
    DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS,
};

use super::vendor_name;
use crate::hardware::{GpuInfo, GpuKind, GpuReport};

/// The documented C signature of `DXCoreCreateAdapterFactory`.
type CreateAdapterFactory = unsafe extern "system" fn(*const GUID, *mut *mut c_void) -> HRESULT;

/// The DLL and export DXCore ships in.
const DXCORE_DLL: &str = "dxcore.dll";
const CREATE_FACTORY_SYMBOL: &[u8] = b"DXCoreCreateAdapterFactory\0";

/// `DriverDescription` is a NUL-terminated UTF-8 string; DXCore documents 256
/// bytes as enough for every driver.
const DESCRIPTION_BUFFER: usize = 256;

/// Detects the machine's graphics adapters.
pub(super) fn detect() -> GpuReport {
    // SAFETY: loading a system DLL has no preconditions. The handle outlives
    // every use below, because it is dropped only when `detect` returns.
    let Ok(library) = (unsafe { Library::new(DXCORE_DLL) }) else {
        return GpuReport::default();
    };
    // SAFETY: the symbol is documented to have this ABI and stays valid while
    // `library` is alive.
    let Ok(create) = (unsafe { library.get::<CreateAdapterFactory>(CREATE_FACTORY_SYMBOL) }) else {
        return GpuReport::default();
    };

    let mut factory_ptr: *mut c_void = ptr::null_mut();
    // SAFETY: `create` is the real export; the IID and out-pointer are valid,
    // and DXCore either writes a factory or leaves the pointer null.
    let created = unsafe { create(&IDXCoreAdapterFactory::IID, &mut factory_ptr) };
    if created.is_err() || factory_ptr.is_null() {
        return GpuReport::default();
    }
    // SAFETY: on success DXCore handed over one owned reference to a
    // `IDXCoreAdapterFactory`; taking it here releases it exactly once.
    let factory = unsafe { IDXCoreAdapterFactory::from_raw(factory_ptr) };

    let attributes = [DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS];
    // SAFETY: `factory` is a live DXCore factory; the attribute list and its
    // count describe the same slice.
    let Ok(list) = (unsafe { factory.CreateAdapterList::<IDXCoreAdapterList>(&attributes) }) else {
        return GpuReport::default();
    };
    // SAFETY: a plain accessor on a live adapter list.
    let count = unsafe { list.GetAdapterCount() };

    let mut gpus = Vec::new();
    for index in 0..count {
        // SAFETY: `index` is below `count`, so DXCore returns a valid adapter.
        let Ok(adapter) = (unsafe { list.GetAdapter::<IDXCoreAdapter>(index) }) else {
            continue;
        };
        if let Some(gpu) = read_adapter(&adapter) {
            gpus.push(gpu);
        }
    }

    GpuReport {
        gpus,
        unified_memory: None,
    }
}

/// Reads one adapter, or `None` when it is not a hardware device.
fn read_adapter(adapter: &IDXCoreAdapter) -> Option<GpuInfo> {
    if !bool_property(adapter, IsHardware).unwrap_or(false) {
        return None;
    }

    let integrated = bool_property(adapter, IsIntegrated).unwrap_or(false);
    let vendor = hardware_id(adapter).map(vendor_name);
    let name = description(adapter)
        .or_else(|| vendor.clone())
        .unwrap_or_else(|| "Unknown GPU".to_string());

    let (kind, dedicated_vram) = if integrated {
        (GpuKind::Integrated, None)
    } else {
        (
            GpuKind::Dedicated,
            u64_property(adapter, DedicatedAdapterMemory),
        )
    };

    Some(GpuInfo {
        name,
        vendor,
        kind,
        dedicated_vram,
    })
}

/// Reads a DXCore boolean property, or `None` when it is unsupported.
fn bool_property(adapter: &IDXCoreAdapter, property: DXCoreAdapterProperty) -> Option<bool> {
    // DXCore documents boolean properties as a single byte.
    let mut value: u8 = 0;
    // SAFETY: `value` is one byte, the buffer size passed matches it exactly, and
    // the property is a value type DXCore writes as a boolean.
    let ok = unsafe { adapter.GetProperty(property, 1, (&mut value as *mut u8).cast::<c_void>()) };
    ok.is_ok().then_some(value != 0)
}

/// Reads a DXCore 64-bit property, or `None` when it is unsupported.
fn u64_property(adapter: &IDXCoreAdapter, property: DXCoreAdapterProperty) -> Option<u64> {
    let mut value: u64 = 0;
    // SAFETY: the buffer is exactly eight bytes, matching the property's type.
    let ok = unsafe { adapter.GetProperty(property, 8, (&mut value as *mut u64).cast::<c_void>()) };
    ok.is_ok().then_some(value)
}

/// Reads the driver's description of the adapter, if it provides one.
fn description(adapter: &IDXCoreAdapter) -> Option<String> {
    let mut buffer = [0u8; DESCRIPTION_BUFFER];
    // SAFETY: the buffer is `DESCRIPTION_BUFFER` bytes and the size passed is
    // the same, so DXCore cannot write past its end.
    let ok = unsafe {
        adapter.GetProperty(
            DriverDescription,
            DESCRIPTION_BUFFER,
            buffer.as_mut_ptr().cast::<c_void>(),
        )
    };
    if ok.is_err() {
        return None;
    }

    let end = buffer
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(DESCRIPTION_BUFFER);
    let text = String::from_utf8_lossy(&buffer[..end]).trim().to_string();
    (!text.is_empty()).then_some(text)
}

/// Reads the adapter's PCI vendor ID.
fn hardware_id(adapter: &IDXCoreAdapter) -> Option<u32> {
    // SAFETY: `DXCoreHardwareID` is four `u32`s with no invalid bit patterns.
    let mut id: DXCoreHardwareID = unsafe { std::mem::zeroed() };
    // SAFETY: the buffer size is exactly `size_of::<DXCoreHardwareID>()`, which
    // is the property's documented type.
    let ok = unsafe {
        adapter.GetProperty(
            HardwareID,
            std::mem::size_of::<DXCoreHardwareID>(),
            (&mut id as *mut DXCoreHardwareID).cast::<c_void>(),
        )
    };
    ok.is_ok().then_some(id.vendorID)
}
