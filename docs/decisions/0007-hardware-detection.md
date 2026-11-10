# 0007 — Hardware detection: GPUs and VRAM

## Status

Accepted, 2026-11-10

## Context

The memory forecast (§5.2) runs before any model loads, so it needs the machine's
RAM, CPU and — where one exists — a GPU's dedicated VRAM. Two spec rules shape
every choice here (§2.6, §5.2):

- Integrated or unknown GPUs contribute no VRAM, and a figure is never guessed.
- A GPU path must never stop the app from launching on a machine without that GPU
  or driver.

`crates/regent-core/src/hardware/gpu/` therefore has one implementation per OS,
picked at compile time by `cfg(target_os)`. Every other OS, and every failure
inside a detection path, returns `GpuReport::default()` (no GPUs) or a partial
report. Nothing in the module panics; this is covered by tests that run on a
machine with no GPU at all.

## Method per OS

**Windows — DXCore.** `dxcore.dll` is opened with `libloading` and the
`DXCoreCreateAdapterFactory` export is resolved by name. The `windows` crate
supplies the DXCore types only (`Win32_Graphics_DXCore`), and the raw C signature
is declared locally. A `IDXCoreAdapterFactory` is created, an adapter list is
requested for `DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS`, and each adapter is
read with `GetProperty`:

| Property | Use |
|---|---|
| `IsHardware` | adapters that answer false (software) are skipped |
| `IsIntegrated` | chooses `Integrated` versus `Dedicated` |
| `DedicatedAdapterMemory` | `dedicated_vram`, only for `Dedicated` adapters |
| `DriverDescription` | the adapter name |
| `HardwareID` | the PCI vendor ID, mapped by `vendor_name` |

The vendor mapping is 0x10DE NVIDIA, 0x1002 AMD, 0x8086 Intel; any other ID is
shown as hex, so the report never invents a name. All `unsafe` lives in
`windows.rs`, each block with a `// SAFETY:` note.

**Linux — sysfs plus NVML.** `detect_sysfs` reads `/sys/class/drm/card<N>`:
`card` followed only by digits, so connector entries such as `card0-HDMI-A-1`
and `renderD128` are ignored. AMD entries are dedicated when
`device/mem_info_vram_total` is 2 GiB or more, integrated otherwise; NVIDIA
entries are skipped because NVML covers them with a real model name; every other
vendor is `Unknown` with no VRAM. NVML comes from `nvml-wrapper`, which loads
`libnvidia-ml.so.1` at runtime: a machine without the NVIDIA driver reports no
NVIDIA entries instead of failing. `detect_sysfs` is compiled on every OS — its
tests run on Windows CI too — and only called on Linux.

**macOS — Metal.** `MTLCreateSystemDefaultDevice` gives the default GPU. When it
reports `hasUnifiedMemory` the GPU is `Integrated` with no VRAM and the report
carries `UnifiedMemory { gpu_working_set_limit: recommendedMaxWorkingSetSize }`;
otherwise the GPU is `Dedicated` and that same limit is its `dedicated_vram`. No
device means an empty report, which also covers Intel Macs without Metal.

## Why everything is loaded at runtime

- `dxcore.dll` is only present from Windows 10 2004 onwards. Linking it would
  turn a missing DLL into a launch failure.
- The NVIDIA driver library is absent on most machines, including every Mac and
  any Linux box without a discrete NVIDIA card.
- The spec's §2 dependency check (`dumpbin /dependents`, `otool -L`, `ldd`) must
  show only OS libraries for the shipped binaries. Runtime loading keeps
  `dxcore.dll` and `libnvidia-ml.so.1` out of the import table.

Metal and the `windows` crate's DXCore types are header-only at the binary level:
they define interfaces, not link-time imports, so they add no dependency either.

## The AMD 2 GiB rule

AMD APUs expose a small slice of system RAM through
`mem_info_vram_total` — a firmware carve-out, typically 512 MiB, not memory of
its own. Counting it as dedicated VRAM would move model layers onto a pool that
is really RAM, and the forecast would double-count it. Below 2 GiB an AMD card
is therefore treated as integrated with no VRAM; at or above it the figure is a
real dedicated pool (even a 2 GiB card reports close to its full size).

## Unknown GPUs get 0 VRAM

`GpuKind::Unknown` and `GpuKind::Integrated` always carry
`dedicated_vram: None`. `compute_budget` ignores them, so a machine whose GPU
cannot be classified forecasts as if it were CPU-only. The alternative —
believing a guessed number — would promise the user a fit the machine cannot
deliver, which §5.2 forbids ("Never guess a figure").

## Verification

- `detect_sysfs` is tested against fake `tempfile` DRM trees: AMD 8 GiB →
  `Dedicated`, AMD 512 MiB → `Integrated`, an NVIDIA entry skipped, Intel →
  `Unknown`, connector folders ignored, missing files and missing roots without a
  panic.
- `crates/regent-core/examples/hardware.rs` prints the detected RAM, CPU, GPU
  report and the budget with no overrides, so detection can be compared with
  Task Manager → Performance → GPU.
- The Linux and macOS paths are type-checked from a Windows host with
  `cargo check -p regent-core --target <triple>`.

## Consequences

- Only the largest dedicated GPU's VRAM reaches the budget; multi-GPU machines
  are not pooled, matching §5.2.
- Windows detection needs DXCore (Windows 10 2004+). Older machines simply
  report no GPUs and forecast as CPU-only.
- Bumping `windows`, `nvml-wrapper`, `objc2` or `objc2-metal` must be re-checked
  against all three targets, since each dependency is compiled only on its own
  OS.
