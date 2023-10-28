use std::{
    collections::HashMap,
    io::Read,
    path::Path,
    process::{Command, Stdio},
};

use anyhow::anyhow;
use serde::{Deserialize, Serialize};

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

fn hf_hub_test_runner(override_cmd: &Option<String>, repo_path: &Path) -> anyhow::Result<f32> {
    let cmd = if let Some(cmd) = override_cmd {
        &cmd
    } else {
        "python3"
    };
    let mut child = Command::new(cmd)
        .args([
            "-m",
            "pytest",
            "tests",
            "-k",
            "_utils_ and not _utils_cache and not _utils_http and not paginate and not git",
        ])
        .current_dir(repo_path)
        .stdout(Stdio::piped())
        .spawn()?;
    let mut stdout = String::new();
    child
        .stdout
        .take()
        .ok_or(anyhow!("failed to take stdout"))?
        .read_to_string(&mut stdout)?;
    let status = child.wait()?;
    if !status.success() {
        return Ok(0f32);
    }
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
        } else if res.contains("failed") {
            let failed_str = res.replace(" failed", "");
            failed = failed_str.parse::<u32>()? as f32;
        }
    }

    Ok(passed / (passed + failed))
}

fn simple_test_runner(override_cmd: &Option<String>, repo_path: &Path) -> anyhow::Result<f32> {
    let cmd = if let Some(cmd) = override_cmd {
        &cmd
    } else {
        "cargo"
    };
    let mut child = Command::new(cmd)
        .arg("test")
        .args(["--", "-Z", "unstable-options", "--format", "json"])
        .current_dir(repo_path)
        .stdout(Stdio::piped())
        .spawn()?;

    let mut stdout = String::new();
    child
        .stdout
        .take()
        .ok_or(anyhow!("failed to take stdout"))?
        .read_to_string(&mut stdout)?;
    let lines = stdout.split_terminator('\n');
    let test_suite_result = serde_json::from_str::<TestSuiteResult>(lines.last().unwrap())?;

    Ok(test_suite_result.passed as f32
        / (test_suite_result.passed as f32 + test_suite_result.failed as f32))
}

type Runner = fn(&Option<String>, &Path) -> anyhow::Result<f32>;

pub fn get_runner_map() -> HashMap<String, Runner> {
    let mut runner_map = HashMap::new();
    runner_map.insert(
        "simple_test_runner".to_owned(),
        simple_test_runner as Runner,
    );
    runner_map.insert(
        "hf_hub_test_runner".to_owned(),
        hf_hub_test_runner as Runner,
    );
    runner_map
}
