from callcenter_ynew_backcast_ensemble import BackcastConfig, BackcastEnsemble

config = BackcastConfig(
    n_splits=3,
    test_size=21,
    gap=7,
    tsfresh_windows=(28, 56),
    top_k_features=140,
    use_prophet=True,
)

runner = BackcastEnsemble(config)
df = runner.load_csv("your_data.csv")
runner.fit(df)
runner.save_outputs("./backcast_output")

print(runner.cv_scores_)
print(runner.ensemble_score_)
print(runner.get_weight_table())
print(runner.result_.head())
