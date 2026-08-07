$ErrorActionPreference = "Stop"
$Config = "config\project.yaml"
$Steps = @(
  "00_extract_archives.py",
  "01_audit_raw_data.py",
  "02_select_scope.py",
  "03_build_curated_panel.py",
  "04_statistical_analysis.py",
  "05_build_feature_dataset.py",
  "05b_feature_diagnostics.py",
  "05c_build_family_weights.py",
  "06_backtest_and_select.py",
  "06b_statistical_models_benchmark.py",
  "07_evaluate_holdout.py",
  "08_train_final_and_predict.py",
  "09_inventory_simulation.py",
  "10_explain_model.py",
  "11_generate_executive_report.py",
  "12_monitoring_snapshot.py"
)
foreach ($Step in $Steps) {
  Write-Host "`n>>> $Step" -ForegroundColor Cyan
  python "scripts\$Step" --config $Config
}
pytest
Write-Host "`nPipeline terminé. Consulte reports\EXECUTIVE_REPORT.md" -ForegroundColor Green
