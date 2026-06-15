mod config;

use lattice_gas::markov_chain::{EMPTY, INERT};
use lattice_gas::boundary_condition::Periodic;
use lattice_gas::ending_criterion::Time;
use lattice_gas::simulate::simulate;
use ndarray::Array2;
use rand::prelude::*;
use std::io::Write;

/// Finds the single particle's (row, col) on the lattice.
fn find_particle(state: &Array2<u32>) -> (usize, usize) {
    for ((i, j), &v) in state.indexed_iter() {
        if v != EMPTY {
            return (i, j);
        }
    }
    panic!("No particle found on lattice");
}

/// Minimum-image displacement for one coordinate, periodic length `l`.
fn min_image(delta: isize, l: usize) -> isize {
    let l = l as isize;
    if delta > l / 2 {
        delta - l
    } else if delta < -(l / 2) {
        delta + l
    } else {
        delta
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        panic!("Usage: {} <config.toml>", args[0]);
    }
    let config = config::Config::from_file(&args[1]);

    let lx = config.lattice.lx;
    let ly = config.lattice.ly;
    let num_steps = config.simulation.num_chunks;
    let n_traj = config.simulation.num_trajectories;

    println!("Test 2: Diffusion (single-particle MSD)");
    println!("Lambda: {}", config.chain.lambda);
    println!("Lattice: {}x{}", lx, ly);
    println!("Trajectories: {}", n_traj);
    println!("Steps per trajectory: {}", num_steps);
    println!("Step time: {}", config.simulation.chunk_time);

    let mut msd_sum = vec![0.0_f64; num_steps];
    let mut time_sum = vec![0.0_f64; num_steps];
    let mut dx_sum = vec![0.0_f64; num_steps];
    let mut dy_sum = vec![0.0_f64; num_steps];

    for traj in 0..n_traj {
        let seed = config.simulation.seed + traj as u64 * 1_000_003;

        let mut state: Array2<u32> = Array2::from_elem((lx, ly), EMPTY);
        let start = (lx / 2, ly / 2);
        state[[start.0, start.1]] = INERT;

        let chain = lattice_gas::markov_chain::HeteroNVTDrivenChain::new(
            1.0,
            config.chain.bond_energy,
            config.chain.delta_f,
            config.chain.delta_mu,
            config.chain.eta,
            config.chain.lambda,
            config.chain.scheme.clone(),
        );
        let boundary = Periodic;

        let mut current_state = state;
        let mut prev_pos = (start.0 as isize, start.1 as isize);
        let mut unwrapped = (0.0_f64, 0.0_f64);
        let mut total_time = 0.0_f64;

        for step in 0..num_steps {
            let step_rng = StdRng::seed_from_u64(seed + step as u64 + 1);
            let ending = Time::new(config.simulation.chunk_time);

            let sim = simulate(
                current_state,
                Box::new(boundary),
                Box::new(chain.clone()),
                vec![],
                vec![Box::new(ending)],
                step_rng,
            );

            total_time += sim.time;

            let (i, j) = find_particle(&sim.state);
            let raw_dx = i as isize - prev_pos.0;
            let raw_dy = j as isize - prev_pos.1;
            let dx = min_image(raw_dx, lx);
            let dy = min_image(raw_dy, ly);

            unwrapped.0 += dx as f64;
            unwrapped.1 += dy as f64;
            dx_sum[step] += unwrapped.0;
            dy_sum[step] += unwrapped.1;

            let r2 = unwrapped.0 * unwrapped.0 + unwrapped.1 * unwrapped.1;
            msd_sum[step] += r2;
            time_sum[step] += total_time;

            prev_pos = (i as isize, j as isize);
            current_state = sim.state;
        }

        if traj % 5 == 0 {
            println!("  trajectory {}/{} done", traj + 1, n_traj);
        }
    }

    std::fs::create_dir_all(&config.output.outdir).unwrap();
    let path = std::path::Path::new(&config.output.outdir).join("msd.csv");
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = std::io::BufWriter::new(file);
    writeln!(writer, "step,time,msd,mean_dx,mean_dy,lambda").unwrap();
    for step in 0..num_steps {
        let t = time_sum[step] / n_traj as f64;
        let msd = msd_sum[step] / n_traj as f64;
        let mdx = dx_sum[step] / n_traj as f64;
        let mdy = dy_sum[step] / n_traj as f64;
        writeln!(writer, "{},{:.6},{:.6},{:.6},{:.6},{}", step + 1, t, msd, mdx, mdy, config.chain.lambda).unwrap();
    }
    println!("✅ MSD data written to {}", path.display());
}