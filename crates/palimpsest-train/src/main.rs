mod backend;
mod cmd;
mod stats;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "palimpsest-train", about = "Training entrypoints for the Palimpsest phases")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Phase 1: train the Drafter standalone on the KV toy task.
    Drafter(cmd::drafter::Args),
    /// Phase 2: produce ablation-derived weight labels from a trained Drafter.
    Ablate(cmd::ablate::Args),
    /// Phase 3: train the Critic on ablation labels; validate rank correlation.
    Critic(cmd::critic::Args),
    /// Phase 4: train the Router on Drafter activations against Critic targets.
    Router(cmd::router::Args),
    /// Phase 5: train the Editor on corrupt-and-repair pairs.
    Editor(cmd::editor::Args),
    /// Phase 6: run the full self-editing loop and dump trajectories.
    RunLoop(cmd::run_loop::Args),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Drafter(args) => cmd::drafter::run(args),
        Command::Ablate(args) => cmd::ablate::run(args),
        Command::Critic(args) => cmd::critic::run(args),
        Command::Router(args) => cmd::router::run(args),
        Command::Editor(args) => cmd::editor::run(args),
        Command::RunLoop(args) => cmd::run_loop::run(args),
    }
}
