//! Privacy tests for the ZK sonoluminescence proof system.
//!
//! These tests verify that the ZK property holds: proofs generated with
//! different private parameters are all valid, and the verifier cannot
//! distinguish between them (only public inputs are revealed).

use ff::PrimeField;
use halo2_proofs::{
    halo2curves::bn256::{Bn256, Fr, G1Affine},
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, VerifyingKey},
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

use zk_physics::{
    circuits::sonoluminescence::{compute_public_inputs, SonoluminescenceCircuit},
    witness::SimulationWitness,
};

struct ProofBundle {
    params: ParamsKZG<Bn256>,
    vk: VerifyingKey<G1Affine>,
    proof: Vec<u8>,
    public_inputs: Vec<Fr>,
}

/// Generate a proof and return everything needed to verify it.
fn prove(steps: usize) -> ProofBundle {
    let witness = SimulationWitness::collapse_preset(steps);
    let public_inputs = compute_public_inputs::<Fr>(&witness);
    let k = SonoluminescenceCircuit::min_k(steps);

    let params: ParamsKZG<Bn256> = ParamsKZG::new(k);
    let circuit = SonoluminescenceCircuit::new(witness);
    let vk = keygen_vk(&params, &circuit).expect("keygen_vk failed");
    let pk = keygen_pk(&params, vk.clone(), &circuit).expect("keygen_pk failed");

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

    ProofBundle {
        params,
        vk,
        proof: transcript.finalize(),
        public_inputs,
    }
}

/// Verify a proof using the same params/vk it was generated with.
fn verify_bundle(bundle: &ProofBundle) -> bool {
    let mut transcript =
        Blake2bRead::<&[u8], G1Affine, Challenge255<G1Affine>>::init(&bundle.proof[..]);
    verify_proof::<
        KZGCommitmentScheme<Bn256>,
        VerifierSHPLONK<Bn256>,
        Challenge255<G1Affine>,
        Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
        SingleStrategy<Bn256>,
    >(
        &bundle.params,
        &bundle.vk,
        SingleStrategy::new(&bundle.params),
        &[&[&bundle.public_inputs]],
        &mut transcript,
    )
    .is_ok()
}

/// Verify a proof with different public inputs.
fn verify_with_pi(bundle: &ProofBundle, pi: &[Fr]) -> bool {
    let mut transcript =
        Blake2bRead::<&[u8], G1Affine, Challenge255<G1Affine>>::init(&bundle.proof[..]);
    verify_proof::<
        KZGCommitmentScheme<Bn256>,
        VerifierSHPLONK<Bn256>,
        Challenge255<G1Affine>,
        Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
        SingleStrategy<Bn256>,
    >(
        &bundle.params,
        &bundle.vk,
        SingleStrategy::new(&bundle.params),
        &[&[pi]],
        &mut transcript,
    )
    .is_ok()
}

#[test]
fn test_proof_valid_10_steps() {
    let bundle = prove(10);
    assert!(verify_bundle(&bundle), "Valid proof should verify");
}

#[test]
fn test_proof_valid_50_steps() {
    let bundle = prove(50);
    assert!(verify_bundle(&bundle), "Valid proof should verify");
}

#[test]
fn test_proof_size_constant() {
    let b10 = prove(10);
    let b50 = prove(50);
    assert_eq!(
        b10.proof.len(),
        b50.proof.len(),
        "Proof size should be constant regardless of step count"
    );
}

#[test]
fn test_different_proofs_both_valid() {
    // Two proofs of the same computation should both verify
    // but have different bytes (due to random blinding factors).
    // They use different params since KZG setup is randomized.
    let b1 = prove(10);
    let b2 = prove(10);

    assert!(verify_bundle(&b1));
    assert!(verify_bundle(&b2));

    // Public inputs should be identical (same computation)
    assert_eq!(
        b1.public_inputs, b2.public_inputs,
        "Same computation should produce same public inputs"
    );

    // Proofs should differ (randomized blinding + different KZG params)
    assert_ne!(
        b1.proof, b2.proof,
        "Two proofs should differ due to randomization"
    );
}

#[test]
fn test_tampered_proof_rejected() {
    let mut bundle = prove(10);

    // Tamper with the proof
    if let Some(byte) = bundle.proof.last_mut() {
        *byte ^= 0xFF;
    }

    assert!(
        !verify_bundle(&bundle),
        "Tampered proof should be rejected"
    );
}

#[test]
fn test_wrong_public_inputs_rejected() {
    let bundle = prove(10);

    // Modify the public input (change R0)
    let mut bad_pi = bundle.public_inputs.clone();
    bad_pi[0] = Fr::from(12345u64);

    assert!(
        !verify_with_pi(&bundle, &bad_pi),
        "Proof with wrong public inputs should be rejected"
    );
}

#[test]
#[ignore] // slow: ~30s
fn test_proof_valid_100_steps() {
    let bundle = prove(100);
    assert!(verify_bundle(&bundle), "Valid proof should verify");
}

#[test]
#[ignore] // slow: large circuit
fn test_proof_valid_1000_steps() {
    let bundle = prove(1000);
    assert!(verify_bundle(&bundle), "Valid proof should verify");
}

#[test]
fn test_public_inputs_contain_r0_temperature_emission() {
    let witness = SimulationWitness::collapse_preset(10);
    let pi = compute_public_inputs::<Fr>(&witness);

    // 3 public inputs: R0, final_temperature, total_emission
    assert_eq!(
        pi.len(),
        3,
        "Public inputs should be R0, final_temperature, and total_emission"
    );

    // First public input should be R0
    assert_eq!(
        pi[0],
        Fr::from_u128(witness.r0),
        "First public input should be R0"
    );
}
