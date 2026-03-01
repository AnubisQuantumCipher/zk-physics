//! Benchmark for ZK sonoluminescence proof system.
//!
//! Measures constraint count, prove time, verify time, and proof size
//! at various step counts.
//!
//! Usage:
//!   cargo run --release --bin benchmark
//!   cargo run --release --bin benchmark -- --steps 10,50,100

use halo2_proofs::{
    halo2curves::bn256::{Bn256, Fr, G1Affine},
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof},
    poly::{
        commitment::ParamsProver,
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverSHPLONK, VerifierSHPLONK},
            strategy::SingleStrategy,
        },
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use rand::rngs::OsRng;
use std::time::Instant;

use zk_physics::{
    circuits::sonoluminescence::{compute_public_inputs, SonoluminescenceCircuit},
    witness::SimulationWitness,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut step_counts: Vec<usize> = vec![10, 50, 100, 500, 1000];

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                i += 1;
                step_counts = args[i]
                    .split(',')
                    .map(|s| s.trim().parse().expect("Invalid step count"))
                    .collect();
            }
            _ => {
                eprintln!("Unknown arg: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    println!("========================================");
    println!("ZK SONOLUMINESCENCE BENCHMARK");
    println!("========================================");
    println!(
        "Step counts: {:?}",
        step_counts
    );
    println!();

    println!(
        "{:>6} {:>4} {:>8} {:>12} {:>12} {:>12} {:>10}",
        "Steps", "k", "Rows", "Witness(ms)", "Prove(ms)", "Verify(ms)", "Proof(B)"
    );
    println!("{}", "-".repeat(72));

    for &steps in &step_counts {
        benchmark_steps(steps);
    }

    println!("\n========================================");
    println!("BENCHMARK COMPLETE");
    println!("========================================");
}

fn benchmark_steps(steps: usize) {
    // Witness generation
    let t0 = Instant::now();
    let witness = SimulationWitness::collapse_preset(steps);
    let public_inputs = compute_public_inputs::<Fr>(&witness);
    let witness_ms = t0.elapsed().as_millis();

    let k = SonoluminescenceCircuit::min_k(steps);
    let rows = 1u64 << k;

    // Setup
    let params: ParamsKZG<Bn256> = ParamsKZG::new(k);
    let circuit = SonoluminescenceCircuit::new(witness);
    let vk = keygen_vk(&params, &circuit).expect("keygen_vk failed");
    let pk = keygen_pk(&params, vk.clone(), &circuit).expect("keygen_pk failed");

    // Prove
    let t1 = Instant::now();
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
    let prove_ms = t1.elapsed().as_millis();

    // Verify
    let t2 = Instant::now();
    let mut transcript =
        Blake2bRead::<&[u8], G1Affine, Challenge255<G1Affine>>::init(&proof[..]);
    verify_proof::<
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
    )
    .expect("Verification failed");
    let verify_ms = t2.elapsed().as_millis();

    println!(
        "{:>6} {:>4} {:>8} {:>12} {:>12} {:>12} {:>10}",
        steps, k, rows, witness_ms, prove_ms, verify_ms, proof.len()
    );
}
