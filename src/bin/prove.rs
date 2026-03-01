//! ZK proof generator for sonoluminescence simulation.
//!
//! Generates a proof that a bubble collapse was computed correctly.
//!
//! Usage:
//!   cargo run --release --bin prove -- --steps 100
//!   cargo run --release --bin prove -- --steps 100 --output output/proof.bin

use halo2_proofs::{
    halo2curves::bn256::{Bn256, Fr, G1Affine},
    plonk::{create_proof, keygen_pk, keygen_vk},
    poly::{
        commitment::{Params, ParamsProver},
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::ProverSHPLONK,
        },
    },
    transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
};
use rand::rngs::OsRng;
use std::{fs, path::PathBuf, time::Instant};

use zk_physics::{
    circuits::sonoluminescence::{compute_public_inputs, SonoluminescenceCircuit},
    witness::SimulationWitness,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut steps = 100;
    let mut output = PathBuf::from("output/proof.bin");
    let params_file = PathBuf::from("output/params.bin");
    let vk_file = PathBuf::from("output/vk.bin");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                i += 1;
                steps = args[i].parse().expect("Invalid steps");
            }
            "--output" => {
                i += 1;
                output = PathBuf::from(&args[i]);
            }
            _ => {
                eprintln!("Unknown arg: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    println!("========================================");
    println!("ZK PROOF OF SONOLUMINESCENCE");
    println!("========================================");
    println!("Steps: {}", steps);

    // Generate witness
    let t0 = Instant::now();
    println!("\n[1/4] Generating witness...");
    let witness = SimulationWitness::collapse_preset(steps);
    let peak_t = witness.compute_peak_temperature();
    let peak_t_float = peak_t as f64 / 1e30;
    println!("  Peak temperature: {:.0} K", peak_t_float);
    println!("  Min radius: {:.4} um", witness.min_radius() as f64 / 1e30 * 1e6);
    if peak_t_float > 5000.0 {
        println!("  SONOLUMINESCENCE DETECTED");
    }
    println!("  Witness generated in {:.2?}", t0.elapsed());

    // Compute public inputs
    let public_inputs = compute_public_inputs::<Fr>(&witness);
    println!("  Public inputs: R0, final_temperature, total_emission");

    // Circuit setup
    let k = SonoluminescenceCircuit::min_k(steps);
    println!("\n[2/4] Setting up circuit (k={}, 2^k={} rows)...", k, 1u64 << k);

    let t1 = Instant::now();
    let params: ParamsKZG<Bn256> = ParamsKZG::new(k);
    println!("  KZG params generated in {:.2?}", t1.elapsed());

    let circuit = SonoluminescenceCircuit::new(witness);
    let t2 = Instant::now();
    let vk = keygen_vk(&params, &circuit).expect("keygen_vk failed");
    let pk = keygen_pk(&params, vk.clone(), &circuit).expect("keygen_pk failed");
    println!("  Keys generated in {:.2?}", t2.elapsed());

    // Generate proof
    println!("\n[3/4] Generating proof...");
    let t3 = Instant::now();

    let mut transcript = Blake2bWrite::<Vec<u8>, G1Affine, Challenge255<G1Affine>>::init(vec![]);
    create_proof::<
        KZGCommitmentScheme<Bn256>,
        ProverSHPLONK<Bn256>,
        Challenge255<G1Affine>,
        OsRng,
        Blake2bWrite<Vec<u8>, G1Affine, Challenge255<G1Affine>>,
        SonoluminescenceCircuit,
    >(
        &params,
        &pk,
        &[circuit],
        &[&[&public_inputs]],
        OsRng,
        &mut transcript,
    )
    .expect("Proof generation failed");

    let proof = transcript.finalize();
    println!("  Proof generated in {:.2?}", t3.elapsed());
    println!("  Proof size: {} bytes", proof.len());

    // Save outputs
    println!("\n[4/4] Saving outputs...");
    fs::create_dir_all(output.parent().unwrap()).ok();

    fs::write(&output, &proof).expect("Failed to write proof");
    println!("  Proof: {}", output.display());

    // Save public inputs as JSON
    let pi_file = output.with_extension("json");
    let pi_json = serde_json::json!({
        "steps": steps,
        "k": k,
        "public_inputs": public_inputs.iter().map(|f| format!("{:?}", f)).collect::<Vec<_>>(),
        "peak_temperature_K": peak_t_float,
        "sonoluminescence": peak_t_float > 5000.0,
        "public_input_labels": ["R0", "final_temperature", "total_emission"],
    });
    fs::write(&pi_file, serde_json::to_string_pretty(&pi_json).unwrap())
        .expect("Failed to write public inputs");
    println!("  Public inputs: {}", pi_file.display());

    // Save params and VK for verifier
    let mut params_buf = vec![];
    params.write(&mut params_buf).expect("Failed to serialize params");
    fs::write(&params_file, &params_buf).expect("Failed to write params");
    println!("  Params: {}", params_file.display());

    let mut vk_buf = vec![];
    vk.write(&mut vk_buf, halo2_proofs::SerdeFormat::RawBytes)
        .expect("Failed to serialize VK");
    fs::write(&vk_file, &vk_buf).expect("Failed to write VK");
    println!("  VK: {}", vk_file.display());

    let total = t0.elapsed();
    println!("\n========================================");
    println!("PROOF COMPLETE in {:.2?}", total);
    println!("========================================");
}
