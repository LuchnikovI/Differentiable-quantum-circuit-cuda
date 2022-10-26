use cc;

macro_rules! cuda_flags {
    ($builder:expr) => {
      $builder
        .flag("-arch=sm_86")
        .file("src/primitives.cu")
        .compile("libprimitives.a");
    };
}

fn main() {
  println!("cargo:rerun-if-changed=src/primitives.cu");
  println!("cargo:rustc-link-lib=dylib=cuda");
  println!("cargo:rustc-link-lib=dylib=cudart");
  println!("cargo:rustc-link-lib=dylib=cublas");
  #[cfg(not(feature = "f64"))]
  cuda_flags!(cc::Build::new().cuda(true));
  #[cfg(feature = "f64")]
  cuda_flags!(cc::Build::new().cuda(true).flag("-DF64"));
}
