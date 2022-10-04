use cc;

fn main() {

println!("cargo:rustc-link-lib=dylib=cuda");
println!("cargo:rustc-link-lib=dylib=cudart");
println!("cargo:rustc-link-lib=dylib=curand");
println!("cargo:rerun-if-changed=src/wrapper.cuh");

cc::Build::new().cuda(true)
  .file("src/primitives.cu")
  .compile("libprimitives.a");
}