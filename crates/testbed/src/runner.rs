use std::{path::Path, process::Stdio};

use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use tokio::{io::AsyncReadExt, process::Command};
use tracing::debug;

use crate::parse_env;

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
    debug!("running pytest tests: {cmd} {}", args.join(" "));
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
    child.wait().await?;

    // XXX: the pytest command can still fail even after the compilation check
    // the above check should prevent an error, but better safe than sorry
    let lines = stdout.split_terminator('\n');
    let result = match lines.last() {
        Some(line) => line.replace('=', "").trim().to_owned(),
        None => return Ok(0f32),
    };
    let mut passed = 0f32;
    let mut failed = 0f32;
    let without_time = &result[0..result.find("in").unwrap_or(result.len())].trim();
    for res in without_time.split(", ") {
        if res.contains("passed") {
            let passed_str = res.replace(" passed", "");
            passed = passed_str.parse::<u32>()? as f32;
        } else if res.contains("failed") && !res.contains("xfailed") {
            let failed_str = res.replace(" failed", "");
            failed = failed_str.parse::<u32>()? as f32;
        } else if res.contains("error") {
            return Ok(0f32);
        }
    }
    if passed == 0f32 && failed == 0f32 {
        return Ok(0f32);
    }

    Ok(passed / (passed + failed))
}

async fn cargo_runner(
    override_cmd: &Option<String>,
    extra_args: &mut Vec<String>,
    env: &Option<Vec<String>>,
    repo_path: &Path,
) -> anyhow::Result<f32> {
    let cmd = if let Some(cmd) = override_cmd {
        cmd
    } else {
        "cargo"
    };
    let mut args = vec![];
    args.append(extra_args);
    if !args.contains(&"--".to_owned()) {
        args.push("--".to_owned());
    }
    args.extend([
        "-Z".to_owned(),
        "unstable-options".to_owned(),
        "--format".to_owned(),
        "json".to_owned(),
    ]);
    debug!("running cargo tests: {cmd} test {}", args.join(" "));
    let parsed_env = parse_env(env)?;
    let mut cmd = Command::new(cmd);
    for (name, value) in parsed_env {
        cmd.env(name, value);
    }
    let mut child = cmd
        .arg("test")
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
    child.wait().await?;
    let lines = stdout.split_terminator('\n');
    let mut passed = 0;
    let mut failed = 0;
    for line in lines {
        let test_suite_result = match serde_json::from_str::<TestSuiteResult>(line) {
            Ok(res) => res,
            Err(_) => continue,
        };
        passed += test_suite_result.passed;
        failed += test_suite_result.failed;
    }
    if passed == 0 && failed == 0 {
        return Ok(0f32);
    }

    Ok(passed as f32 / (passed as f32 + failed as f32))
}

async fn jest_runner(
    override_cmd: &Option<String>,
    override_args: &Option<Vec<String>>,
    repo_path: &Path,
) -> anyhow::Result<f32> {
    let cmd = if let Some(cmd) = override_cmd {
        cmd
    } else {
        "npm"
    };
    let default_args = vec!["run".to_owned(), "test".to_owned()];
    let args = if let Some(args) = override_args {
        args
    } else {
        &default_args
    };
    debug!("running jest tests: {cmd} {}", args.join(" "));
    let mut child = Command::new(cmd)
        .args(args)
        .current_dir(repo_path)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stderr = String::new();
    child
        .stderr
        .take()
        .ok_or(anyhow!("failed to take stderr"))?
        .read_to_string(&mut stderr)
        .await?;
    child.wait().await?;
    let lines = stderr.split_terminator('\n');
    let mut passed = 0f32;
    let mut failed = 0f32;
    for line in lines {
        if line.contains("Tests:") {
            let words = line.trim().split(' ').collect::<Vec<&str>>();
            let mut prev = words[0];
            for word in words {
                if word.contains("passed") {
                    passed = prev.parse::<u32>()? as f32;
                } else if word.contains("failed") {
                    failed = prev.parse::<u32>()? as f32;
                }
                prev = word;
            }
        }
    }
    if passed == 0f32 && failed == 0f32 {
        return Ok(0f32);
    }

    Ok(passed / (passed + failed))
}

async fn vitest_runner(
    override_cmd: &Option<String>,
    override_args: &Option<Vec<String>>,
    repo_path: &Path,
) -> anyhow::Result<f32> {
    let cmd = if let Some(cmd) = override_cmd {
        cmd
    } else {
        "npm"
    };
    let default_args = vec!["run".to_owned(), "test".to_owned()];
    let args = if let Some(args) = override_args {
        args
    } else {
        &default_args
    };
    debug!("running vitest tests: {cmd} {}", args.join(" "));
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
    child.wait().await?;
    let lines = stdout.split_terminator('\n');
    let mut passed = 0f32;
    let mut failed = 0f32;
    for line in lines {
        if line.contains("Tests") {
            let words = line.trim().split(' ').collect::<Vec<&str>>();
            let mut prev = words[0];
            for word in words {
                if word.contains("passed") {
                    passed = prev.parse::<u32>()? as f32;
                } else if word.contains("failed") {
                    failed = prev.parse::<u32>()? as f32;
                }
                prev = word;
            }
        }
    }
    if passed == 0f32 && failed == 0f32 {
        return Ok(0f32);
    }

    Ok(passed / (passed + failed))
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Runner {
    Cargo,
    Jest,
    Pytest,
    Vitest,
}

pub async fn run_test(
    runner: Runner,
    override_cmd: &Option<String>,
    override_args: &Option<Vec<String>>,
    extra_args: &mut Vec<String>,
    env: &Option<Vec<String>>,
    repo_path: &Path,
) -> anyhow::Result<f32> {
    match runner {
        Runner::Cargo => cargo_runner(override_cmd, extra_args, env, repo_path).await,
        Runner::Jest => jest_runner(override_cmd, override_args, repo_path).await,
        Runner::Pytest => pytest_runner(override_cmd, extra_args, repo_path).await,
        Runner::Vitest => vitest_runner(override_cmd, override_args, repo_path).await,
    }
}
