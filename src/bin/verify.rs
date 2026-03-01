//! ZK proof verifier for sonoluminescence simulation.
//!
//! Verifies a proof that a bubble collapse was computed correctly.
//!
//! Usage:
//!   cargo run --release --bin verify
//!   cargo run --release --bin verify -- --proof output/proof.bin

use halo2_proofs::{
    halo2curves::bn256::{Bn256, G1Affine},
    plonk::{verify_proof, VerifyingKey},
    poly::{
        commitment::Params,
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::VerifierSHPLONK,
            strategy::SingleStrategy,
        },
    },
    transcript::{Blake2bRead, Challenge255, TranscriptReadBuffer},
    SerdeFormat,
};
use std::{fs, path::PathBuf, time::Instant};

use zk_physics::circuits::sonoluminescence::SonoluminescenceCircuit;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut proof_file = PathBuf::from("output/proof.bin");
    let mut params_file = PathBuf::from("output/params.bin");
    let mut vk_file = PathBuf::from("output/vk.bin");
    let mut pi_file: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--proof" => {
                i += 1;
                proof_file = PathBuf::from(&args[i]);
            }
            "--params" => {
                i += 1;
                params_file = PathBuf::from(&args[i]);
            }
            "--vk" => {
                i += 1;
                vk_file = PathBuf::from(&args[i]);
            }
            "--public" => {
                i += 1;
                pi_file = Some(PathBuf::from(&args[i]));
            }
            _ => {
                eprintln!("Unknown arg: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Default public inputs file from proof path
    let pi_file = pi_file.unwrap_or_else(|| proof_file.with_extension("json"));

    println!("========================================");
    println!("ZK VERIFIER — SONOLUMINESCENCE");
    println!("========================================");

    // Load params
    println!("\n[1/4] Loading params...");
    let params_bytes = fs::read(&params_file).expect("Failed to read params file");
    let params: ParamsKZG<Bn256> = ParamsKZG::read(&mut &params_bytes[..])
        .expect("Failed to deserialize params");
    println!("  Params loaded ({} bytes)", params_bytes.len());

    // Load VK
    println!("\n[2/4] Loading verifying key...");
    let vk_bytes = fs::read(&vk_file).expect("Failed to read VK file");

    // We need a dummy circuit to read the VK (halo2 needs the circuit shape)
    // The VK read requires knowing the circuit's configure output.
    // We use read_custom which takes the raw bytes.
    let vk: VerifyingKey<G1Affine> = VerifyingKey::read::<
        &[u8],
        SonoluminescenceCircuit,
    >(&mut &vk_bytes[..], SerdeFormat::RawBytes)
        .expect("Failed to deserialize VK");
    println!("  VK loaded ({} bytes)", vk_bytes.len());

    // Load public inputs
    println!("\n[3/4] Loading public inputs...");
    let pi_json: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&pi_file).expect("Failed to read public inputs file"),
    )
    .expect("Failed to parse public inputs JSON");

    let public_inputs: Vec<halo2_proofs::halo2curves::bn256::Fr> = pi_json["public_inputs"]
        .as_array()
        .expect("public_inputs not an array")
        .iter()
        .map(|v| {
            let s = v.as_str().expect("public input not a string");
            // Parse the hex field element (format: "0x...")
            use ff::PrimeField;
            let bytes = hex_to_bytes(s);
            halo2_proofs::halo2curves::bn256::Fr::from_repr(
                bytes.try_into().expect("Wrong byte length for Fr"),
            )
            .expect("Invalid field element")
        })
        .collect();

    let steps = pi_json["steps"].as_u64().unwrap_or(0);
    let sono = pi_json["sonoluminescence"].as_bool().unwrap_or(false);
    let peak_t = pi_json["peak_temperature_K"].as_f64().unwrap_or(0.0);

    let labels = pi_json["public_input_labels"]
        .as_array()
        .map(|a| a.iter().map(|v| v.as_str().unwrap_or("?").to_string()).collect::<Vec<_>>())
        .unwrap_or_else(|| vec!["R0".into(), "final_temperature".into(), "total_emission".into()]);

    println!("  Steps: {}", steps);
    println!("  Peak temperature: {:.0} K", peak_t);
    println!(
        "  Sonoluminescence: {}",
        if sono { "YES" } else { "NO" }
    );
    println!("  Public inputs ({}): {}", public_inputs.len(), labels.join(", "));

    // Load and verify proof
    println!("\n[4/4] Verifying proof...");
    let proof = fs::read(&proof_file).expect("Failed to read proof file");
    println!("  Proof size: {} bytes", proof.len());

    let t0 = Instant::now();
    let mut transcript =
        Blake2bRead::<&[u8], G1Affine, Challenge255<G1Affine>>::init(&proof[..]);

    let result = verify_proof::<
        KZGCommitmentScheme<Bn256>,
        VerifierSHPLONK<Bn256>,
        Challenge255<G1Affine>,
        Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
        SingleStrategy<Bn256>,
    >(
        &params,
        &vk,
        SingleStrategy::new(&params),
        &[&[&public_inputs]],
        &mut transcript,
    );

    let elapsed = t0.elapsed();

    println!("\n========================================");
    match result {
        Ok(_) => {
            println!("RESULT: VALID");
            println!("  Verification time: {:.2?}", elapsed);
            if sono {
                println!("  The proof confirms sonoluminescence");
                println!("  was computed correctly.");
            }
        }
        Err(e) => {
            println!("RESULT: INVALID");
            println!("  Error: {:?}", e);
            println!("  Verification time: {:.2?}", elapsed);
            std::process::exit(1);
        }
    }
    println!("========================================");
}

/// Parse a field element debug string (e.g., "0x1234...abcd") to 32 bytes (little-endian).
fn hex_to_bytes(s: &str) -> [u8; 32] {
    let s = s.trim();
    let hex_str = if s.starts_with("0x") || s.starts_with("0X") {
        &s[2..]
    } else {
        s
    };

    // Pad to 64 hex chars (32 bytes)
    let padded = format!("{:0>64}", hex_str);
    let mut bytes = [0u8; 32];
    for i in 0..32 {
        bytes[31 - i] =
            u8::from_str_radix(&padded[i * 2..i * 2 + 2], 16).expect("Invalid hex in public input");
    }
    bytes
}
