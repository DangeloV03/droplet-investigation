mod config;

use lattice_gas::markov_chain::{BONDING, EMPTY, INERT};
use ndarray::Array2;
use rand::prelude::*;
use ndarray_npy::write_npy;
use std::io::Write;


fn main() {
    // Parse config
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        panic!("Usage: {} <config.toml>", args[0]);
    }
    let config = config::Config::from_file(&args[1]);

    // Print parameters
    println!("\n{:<25} {:<15}", "Parameter", "Value");
    println!("{:<25} {:<15}", "-".repeat(25), "-".repeat(15));
    println!("{:<25} {:<15}", "Lx", config.lattice.lx);
    println!("{:<25} {:<15}", "Ly", config.lattice.ly);
    println!("{:<25} {:<15}", "bond_energy (ε)", config.chain.bond_energy);
    println!("{:<25} {:<15}", "delta_f (Δf)", config.chain.delta_f);
    println!("{:<25} {:<15}", "delta_mu (Δμ)", config.chain.delta_mu);
    println!("{:<25} {:<15}", "eta (η)", config.chain.eta);
    println!("{:<25} {:<15}", "lambda (Λ)", config.chain.lambda);
    println!("{:<25} {:<15}", "scheme", config.chain.scheme);
    println!("{:<25} {:<15}", "density", config.initial_condition.density);
    println!("{:<25} {:<15}", "bonding_fraction", config.initial_condition.bonding_fraction);
    println!("{:<25} {:<15}", "equilibration_time", config.simulation.equilibration_time);
    println!("{:<25} {:<15}", "chunk_time", config.simulation.chunk_time);
    println!("{:<25} {:<15}", "num_chunks", config.simulation.num_chunks);
    println!("{:<25} {:<15}", "seed", config.simulation.seed);
    println!("{:<25} {:<15}", "outdir", config.output.outdir);
    println!("{:<25} {:<15}", "-".repeat(25), "-".repeat(15));

    // Build rng
    let mut rng = StdRng::seed_from_u64(config.simulation.seed);

    // Build initial state
    let total_sites = config.lattice.lx * config.lattice.ly;
    let total_particles = (total_sites as f64 * config.initial_condition.density).round() as usize;
    let num_bonding = (total_particles as f64 * config.initial_condition.bonding_fraction).round() as usize;
    let num_inert = total_particles - num_bonding;

    println!("Total particles: {}", total_particles);
    println!("Bonding (B):     {}", num_bonding);
    println!("Inert (I):       {}", num_inert);

    let mut indices: Vec<(usize, usize)> = (0..config.lattice.lx)
        .flat_map(|i| (0..config.lattice.ly).map(move |j| (i, j)))
        .collect();
    indices.shuffle(&mut rng);

    let mut state: Array2<u32> = Array2::from_elem(
        (config.lattice.lx, config.lattice.ly),
        EMPTY,
    );
    for (idx, (i, j)) in indices.iter().enumerate() {
        if idx < num_bonding {
            state[[*i, *j]] = BONDING;
        } else if idx < num_bonding + num_inert {
            state[[*i, *j]] = INERT;
        } else {
            break;
        }
    }
    println!("✅ Initial state built.");

    let actual_b = state.iter().filter(|&&s| s == BONDING).count(); 

    let actual_i = state.iter().filter(|&&s| s == INERT).count();
    let actual_e = state.iter().filter(|&&s| s == EMPTY).count();
    println!("State check — B: {}, I: {}, E: {}, total sites: {}", 
        actual_b, actual_i, actual_e, actual_b + actual_i + actual_e);


    // Build chain and boundary
    let chain = lattice_gas::markov_chain::HeteroNVTDrivenChain::new(
        1.0,
        config.chain.bond_energy,
        config.chain.delta_f,
        config.chain.delta_mu,
        config.chain.eta,
        config.chain.lambda,
        config.chain.scheme.clone(),
    );
    let boundary = lattice_gas::boundary_condition::Periodic;

    // Equilibration run
    println!("⏳ Starting equilibration for time {}...", config.simulation.equilibration_time);
    let now = std::time::Instant::now();

    let eq_rng = StdRng::seed_from_u64(config.simulation.seed);
    let ending_criterion_eq = lattice_gas::ending_criterion::Time::new(
        config.simulation.equilibration_time,
    );

    let sim_after_eq = lattice_gas::simulate::simulate(
        state,
        Box::new(boundary),
        Box::new(chain.clone()),
        vec![],
        vec![Box::new(ending_criterion_eq)],
        eq_rng,
    );

    println!(
        "✅ Equilibration done. KMC time: {:.1}. Wall time: {:?}",
        sim_after_eq.time,
        now.elapsed()
    );

    let eq_b = sim_after_eq.state.iter().filter(|&&s| s == BONDING).count();
    let eq_i = sim_after_eq.state.iter().filter(|&&s| s == INERT).count();
    let eq_e = sim_after_eq.state.iter().filter(|&&s| s == EMPTY).count();
    println!("After EQ — B: {}, I: {}, E: {}", eq_b, eq_i, eq_e);


    // Production loop
    let mut current_state = sim_after_eq.state.clone();
    let mut total_time = sim_after_eq.time;
    let mut density_rows: Vec<(usize, f64, f64, f64, f64)> = Vec::new();
    let total_sites_f64 = (config.lattice.lx * config.lattice.ly) as f64;

    for chunk_idx in 0..config.simulation.num_chunks {
        if chunk_idx % 10 == 0 {
            println!("⏱️  Running chunk {}/{}...", chunk_idx + 1, config.simulation.num_chunks);
        }

        let chunk_rng = StdRng::seed_from_u64(config.simulation.seed + chunk_idx as u64 + 1);
        let chunk_criterion = lattice_gas::ending_criterion::Time::new(
            config.simulation.chunk_time,
        );

        let chunk_sim = lattice_gas::simulate::simulate(
            current_state,
            Box::new(boundary),
            Box::new(chain.clone()),
            vec![],
            vec![Box::new(chunk_criterion)],
            chunk_rng,
        );

        let raw_b = chunk_sim.state.iter().filter(|&&s| s == BONDING).count();
        let raw_i = chunk_sim.state.iter().filter(|&&s| s == INERT).count();
        


        total_time += chunk_sim.time;

        let rho_b = chunk_sim.state.iter().filter(|&&s| s == BONDING).count() as f64 / total_sites_f64;
        let rho_i = chunk_sim.state.iter().filter(|&&s| s == INERT).count() as f64 / total_sites_f64;
        let rho_e = chunk_sim.state.iter().filter(|&&s| s == EMPTY).count() as f64 / total_sites_f64;
        density_rows.push((chunk_idx + 1, total_time, rho_b, rho_i, rho_e));

        current_state = chunk_sim.state;

        if chunk_idx % 10 == 0 {
        let prod_b = current_state.iter().filter(|&&s| s == BONDING).count();
        let prod_i = current_state.iter().filter(|&&s| s == INERT).count();
        let prod_e = current_state.iter().filter(|&&s| s == EMPTY).count();
        println!("Chunk {} — B: {}, I: {}, E: {}, total particles: {}", 
            chunk_idx + 1, prod_b, prod_i, prod_e, prod_b + prod_i);
}

    }

        // Save final lattice state
    let lattice_path = format!("{}/final_lattice.npy", config.output.outdir);
    write_npy(&lattice_path, &current_state)
        .unwrap_or_else(|e| panic!("Could not write lattice: {}", e));
    println!("✅ Final lattice saved to {}", lattice_path);


    println!("✅ Production run complete. Total KMC time: {:.1}", total_time);



    

    write_density_csv(&config.output.outdir, &density_rows)
        .unwrap_or_else(|e| panic!("Could not write CSV: {}", e));

    println!("✅ CSV written to {}/density_series.csv", config.output.outdir);
}

fn write_density_csv(
    outdir: &str,
    rows: &[(usize, f64, f64, f64, f64)],
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(outdir)?;
    let path = std::path::Path::new(outdir).join("density_series.csv");
    let file = std::fs::File::create(&path)?;
    let mut writer = std::io::BufWriter::new(file);
    writeln!(writer, "chunk,time,rho_bonding,rho_inert,rho_empty")?;
    for (chunk, time, rb, ri, re) in rows {
        writeln!(writer, "{},{:.6},{:.6},{:.6},{:.6}", chunk, time, rb, ri, re)?;
    }
    Ok(())
}



