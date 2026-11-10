//! Prints what Regent detects on this machine.
//!
//! Run it with `cargo run -p regent-core --example hardware`. It never panics,
//! even on a machine where every probe fails.

use regent_core::hardware::{
    compute_budget, BudgetOverrides, GpuKind, GpuReport, HardwareProbe, SystemProbe,
};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn main() {
    let probe = SystemProbe::new();
    let ram = probe.ram();
    let cpu = probe.cpu();
    let gpus = probe.gpus();

    println!("RAM");
    println!("  total      {}", bytes(ram.total));
    println!("  available  {}", bytes(ram.available));

    println!();
    println!("CPU");
    println!("  name       {}", cpu.name);
    println!("  logical    {}", cpu.logical_threads);
    println!(
        "  physical   {}",
        cpu.physical_cores
            .map(|cores| cores.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    );

    println!();
    println!("GPUs");
    if gpus.gpus.is_empty() {
        println!("  none detected");
    }
    for gpu in &gpus.gpus {
        println!(
            "  {} ({}, {:?})",
            gpu.name,
            gpu.vendor.as_deref().unwrap_or("unknown vendor"),
            gpu.kind
        );
        println!(
            "    dedicated VRAM  {}",
            gpu.dedicated_vram
                .map(bytes)
                .unwrap_or_else(|| "none".to_string())
        );
    }
    match &gpus.unified_memory {
        Some(memory) => println!(
            "  unified memory, GPU working-set limit {}",
            memory
                .gpu_working_set_limit
                .map(bytes)
                .unwrap_or_else(|| "unknown".to_string())
        ),
        None => println!("  no unified memory"),
    }
    println!(
        "  dedicated VRAM total {}",
        bytes(total_dedicated_vram(&gpus))
    );
    println!(
        "  {}",
        match largest_dedicated_vram(&gpus) {
            Some(vram) => format!("largest dedicated GPU {}", bytes(vram)),
            None => "no dedicated VRAM to use".to_string(),
        }
    );

    let budget = compute_budget(&ram, &gpus, &BudgetOverrides::default());
    println!();
    println!("Budget (no overrides)");
    println!("  RAM   {} ({:?})", bytes(budget.ram), budget.ram_source);
    println!("  VRAM  {} ({:?})", bytes(budget.vram), budget.vram_source);
    println!("  unified memory: {}", budget.unified);
}

fn bytes(value: u64) -> String {
    format!("{:.2} GiB ({} bytes)", value as f64 / GIB, value)
}

fn total_dedicated_vram(gpus: &GpuReport) -> u64 {
    gpus.gpus
        .iter()
        .filter(|gpu| gpu.kind == GpuKind::Dedicated)
        .filter_map(|gpu| gpu.dedicated_vram)
        .sum()
}

fn largest_dedicated_vram(gpus: &GpuReport) -> Option<u64> {
    gpus.gpus
        .iter()
        .filter(|gpu| gpu.kind == GpuKind::Dedicated)
        .filter_map(|gpu| gpu.dedicated_vram)
        .max()
}
