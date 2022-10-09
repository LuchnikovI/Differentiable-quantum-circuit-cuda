use cc;
use std::env;

macro_rules! cuda_flags {
    ($builder:expr) => {
      $builder
        .flag("-arch=sm_80")
        .file("src/primitives.cu")
        .compile("libprimitives.a");
    };
}

fn main() {
  if let Ok(cuda_path) = env::var("CUDA_HOME") {
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
  } else {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
  }
  println!("cargo:rustc-link-lib=dylib=cuda");
  println!("cargo:rustc-link-lib=dylib=cudart");

  #[cfg(not(feature = "f64"))]
  cuda_flags!(cc::Build::new().cuda(true));

  #[cfg(feature = "f64")]
  cuda_flags!(cc::Build::new().cuda(true).flag("-DF64"));
}
