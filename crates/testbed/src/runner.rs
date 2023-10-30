use std::{path::Path, process::Stdio};

use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use tokio::{io::AsyncReadExt, process::Command};

#[derive(Deserialize, Serialize)]
struct TestSuiteResult {
    r#type: String,
    event: String,
    passed: u32,
    failed: u32,
    ignored: u32,
    measured: u32,
    filtered_out: u32,
    exec_time: f64,
}

async fn pytest_runner(
    override_cmd: &Option<String>,
    extra_args: &mut Vec<String>,
    repo_path: &Path,
) -> anyhow::Result<f32> {
    let cmd = if let Some(cmd) = override_cmd {
        cmd
    } else {
        "python3"
    };
    let mut args = vec![
        "-m".to_owned(),
        "pytest".to_owned(),
        "tests".to_owned(),
        "-q".to_owned(),
        "--disable-warnings".to_owned(),
        "--no-header".to_owned(),
    ];
    args.append(extra_args);
    let mut child = Command::new(cmd)
        .args(args)
        .current_dir(repo_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let mut stdout = String::new();
    child
        .stdout
        .take()
        .ok_or(anyhow!("failed to take stdout"))?
        .read_to_string(&mut stdout)
        .await?;

    // XXX: the pytest command can still fail even after the compilation check
    // the above check should prevent an error, but better safe than sorry
    let lines = stdout.split_terminator('\n');
    let result = match lines.last() {
        Some(line) => line.replace('=', "").trim().to_owned(),
        None => return Ok(0f32),
    };
    let mut passed = 0f32;
    let mut failed = 0f32;
    for res in result.split(", ") {
        if res.contains("passed") {
            let passed_str = res.replace(" passed", "");
            passed = passed_str.parse::<u32>()? as f32;
        } else if res.contains("failed") && !res.contains("xfailed") {
            let failed_str = res.replace(" failed", "");
            failed = failed_str.parse::<u32>()? as f32;
        }
    }

    Ok(passed / (passed + failed))
}

async fn cargo_runner(override_cmd: &Option<String>, repo_path: &Path) -> anyhow::Result<f32> {
    let cmd = if let Some(cmd) = override_cmd {
        cmd
    } else {
        "cargo"
    };
    let mut child = Command::new(cmd)
        .arg("test")
        .args(["--", "-Z", "unstable-options", "--format", "json"])
        .current_dir(repo_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;

    let mut stdout = String::new();
    child
        .stdout
        .take()
        .ok_or(anyhow!("failed to take stdout"))?
        .read_to_string(&mut stdout)
        .await?;
    let lines = stdout.split_terminator('\n');
    let test_suite_result = serde_json::from_str::<TestSuiteResult>(lines.last().unwrap())?;

    Ok(test_suite_result.passed as f32
        / (test_suite_result.passed as f32 + test_suite_result.failed as f32))
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Runner {
    Cargo,
    Pytest,
}

pub async fn run_test(
    runner: Runner,
    override_cmd: &Option<String>,
    extra_args: &mut Vec<String>,
    repo_path: &Path,
) -> anyhow::Result<f32> {
    match runner {
        Runner::Cargo => cargo_runner(override_cmd, repo_path).await,
        Runner::Pytest => pytest_runner(override_cmd, extra_args, repo_path).await,
    }
}
