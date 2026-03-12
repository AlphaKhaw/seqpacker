//! Table formatting for benchmark output.

/// Print the column header row and separator line.
pub fn print_header() {
    println!(
        "{:<20} {:>8}  {:<8} {:>6}  {:>6}  {:>6}  {:>9}  {:>10}  {:>12}",
        "Dataset", "Capacity", "Algo", "Bins", "Eff%", "Pad%", "Avg/Pack", "Time(ms)", "Throughput"
    );
    println!("{}", "-".repeat(105));
}
