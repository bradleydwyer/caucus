fn main() {
    // Scope the macOS undefined-symbol behavior to the Python extension.
    // Applying this as a workspace rustflag would weaken link checking for
    // unrelated binaries such as the Caucus CLI.
    pyo3_build_config::add_extension_module_link_args();
}
