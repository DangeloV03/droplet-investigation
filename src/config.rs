//! Simulation configuration for the NVT droplet investigation.
//! Parsed from a TOML file passed as the first command-line argument.

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub lattice: LatticeConfig,
    pub chain: ChainConfig,
    pub initial_condition: InitialConditionConfig,
    pub simulation: SimulationConfig,
    pub output: OutputConfig,
}

#[derive(Debug, Deserialize)]
pub struct LatticeConfig {
    pub lx: usize,
    pub ly: usize,
}

#[derive(Debug, Deserialize)]
pub struct ChainConfig {
    pub bond_energy: f64,
    pub delta_f: f64,
    pub delta_mu: f64,
    pub eta: f64,
    pub lambda: f64,
    pub scheme: String,
}

#[derive(Debug, Deserialize)]
pub struct InitialConditionConfig {
    pub density: f64,
    pub bonding_fraction: f64,
}

#[derive(Debug, Deserialize)]
pub struct SimulationConfig {
    pub equilibration_time: f64,
    pub chunk_time: f64,
    pub num_chunks: usize,
    pub seed: u64,
}

#[derive(Debug, Deserialize)]
pub struct OutputConfig {
    pub outdir: String,
}

impl Config {
    pub fn from_file(path: &str) -> Self {
        let contents = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Could not read config file '{}': {}", path, e));
        toml::from_str(&contents)
            .unwrap_or_else(|e| panic!("Could not parse config file '{}': {}", path, e))
    }
}