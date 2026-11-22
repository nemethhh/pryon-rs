use pryon_testkit::{synth, Clip};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["synth", name, out] => {
            let Some(clip) = Clip::from_name(name) else {
                eprintln!("unknown clip {name}");
                return ExitCode::from(2);
            };
            let bytes: Vec<u8> = synth(clip).iter().flat_map(|s| s.to_le_bytes()).collect();
            if let Err(e) = std::fs::write(out, bytes) {
                eprintln!("write {out}: {e}");
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("usage: pryon-cli synth <name> OUT.raw");
            ExitCode::from(2)
        }
    }
}
