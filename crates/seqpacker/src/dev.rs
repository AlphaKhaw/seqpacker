//! Development helpers and debugging macros.

/// Remove the first occurrence of `&` from a type name,
/// preserving everything that follows (e.g. lifetimes and `mut`).
pub fn peel_one_ref(type_name: &str) -> &str {
    if let Some(rest) = type_name.strip_prefix('&') {
        rest.trim_start()
    } else {
        type_name
    }
}

/// Print a value's type and content for debugging.
///
/// Usage: `dbg_type!(my_variable);`
///
/// Output: `[file.rs:42] my_variable = <value> (type: MyType)`
#[macro_export]
macro_rules! dbg_type {
    ($val:expr) => {{
        let r = &$val;
        let ty_full = std::any::type_name_of_val(r);
        let ty_one = $crate::dev::peel_one_ref(ty_full);
        eprintln!(
            "[{}:{}] {} = {:?} (type: {})",
            file!(),
            line!(),
            stringify!($val),
            r,
            ty_one
        );
    }};
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peel_one_ref() {
        assert_eq!(peel_one_ref("&i32"), "i32");
        assert_eq!(peel_one_ref("&mut i32"), "mut i32");
        assert_eq!(peel_one_ref("&&i32"), "&i32");
        assert_eq!(peel_one_ref("i32"), "i32");
        assert_eq!(peel_one_ref("&std::vec::Vec<i32>"), "std::vec::Vec<i32>");
    }
}
